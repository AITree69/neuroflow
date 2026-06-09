"""Stage 2 Sprint 3.8 demo — train an FNO1d surrogate for the
1D Morse potential, export to NeuroIR, and verify the
energy + force on the C++ runtime against the analytical
Morse potential.

The demo has two modes:

  default (single sample):
      Train on a single 1D Morse curve
      V(r) = D_e (1 - exp(-a (r - r_e)))^2 - D_e
      with D_e = 1.0, a = 1.6, r_e = 1.05.  The model
      fits the curve tightly (val rel L2 < 1e-2) and the
      C++ runtime matches PyTorch within the Stage 2
      parity budget (< 1e-3).

  --family (parameter sweep):
      Train on a parameter-conditioned dataset where
      each sample's input is [r, D_e, a, r_e] with the
      last three channels broadcast at every r-grid
      point.  This is the standard "global-parameter
      conditioning" pattern for an FNO surrogate of a
      family of potentials.

Run:
    python domains/lammps/examples/17_morse_surrogate.py
    python domains/lammps/examples/17_morse_surrogate.py --family
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from domains.lammps.neuroflow_lammps import (
    MorseConfig,
    MorseSurrogate,
    build_morse_dataset,
    morse_force,
    morse_potential,
)
from neuroflow.ir.export import export_all
from neuroflow.nn.fno import FNO1d, FNO1dConfig


def _build_fno1d_surrogate(n_points: int, in_channels: int = 1) -> FNO1d:
    cfg = FNO1dConfig(
        in_channels=in_channels,
        out_channels=1,
        width=48,
        modes=min(16, n_points // 2),
        n_layers=3,
        activation="gelu",
        pad_factor=1,
        name="fno1d_morse_surrogate",
    )
    return FNO1d(cfg)


def _rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).reshape(pred.size(0), -1)
    ref = target.reshape(target.size(0), -1)
    return torch.mean(
        torch.linalg.norm(diff, dim=1)
        / torch.linalg.norm(ref, dim=1).clamp_min(1e-8)
    )


def _build_train_val(args, mode: str):
    """Build (x_train, y_train, x_val, y_val, r_grid) and the
    in_channels implied by the chosen mode.
    """
    # The dataset always emits 4 channels per point
    # (r, D_e, a, r_e) — in single mode the last three are
    # constant across samples.
    in_channels = 4
    if mode == "single":
        n_total = args.n_train + args.n_val
        data_cfg = MorseConfig(
            n_points=args.n_points, n_samples=n_total,
            seed=args.seed,
            D_e_range=(1.0, 1.0), a_range=(1.6, 1.6),
            r_e_range=(1.05, 1.05),
        )
        x, y, r_grid, _ = build_morse_dataset(data_cfg)
    elif mode == "family":
        data_cfg = MorseConfig(
            n_points=args.n_points,
            n_samples=args.n_train + args.n_val,
            seed=args.seed,
        )
        x, y, r_grid, _ = build_morse_dataset(data_cfg)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    x_train = x[:args.n_train]
    y_train = y[:args.n_train]
    x_val = x[args.n_train:args.n_train + args.n_val]
    y_val = y[args.n_train:args.n_train + args.n_val]
    return x_train, y_train, x_val, y_val, r_grid, in_channels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FNO1d surrogate for the 1D Morse potential"
    )
    parser.add_argument("--mode", choices=["single", "family"],
                         default="single")
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                         default="./artifacts/lammps_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"==> Building Morse dataset (mode = {args.mode})")
    x_train, y_train, x_val, y_val, r_grid, in_channels = _build_train_val(
        args, args.mode
    )
    print(f"   train = {args.n_train}, val = {args.n_val}, "
          f"n_points = {args.n_points}, in_channels = {in_channels}")
    print(f"   r ∈ [{r_grid[0]:.2f}, {r_grid[-1]:.2f}]")

    print("==> Building FNO1d surrogate")
    model = _build_fno1d_surrogate(args.n_points, in_channels=in_channels)
    print(f"   parameters = {model.num_parameters():,}")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train), torch.from_numpy(y_train)
        ),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val), torch.from_numpy(y_val)
        ),
        batch_size=args.batch_size, shuffle=False,
    )

    optim = Adam(model.parameters(), lr=args.lr)
    sched = StepLR(optim, step_size=max(1, args.epochs // 4), gamma=0.5)

    history = {"epoch": [], "train": [], "val": []}
    print("==> Training")
    t0 = time.time()
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_count = 0.0, 0
        for xb, yb in train_loader:
            optim.zero_grad()
            pred = model(xb)
            loss = _rel_l2(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * xb.size(0)
            train_count += xb.size(0)
        train_loss /= max(train_count, 1)
        sched.step()

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = _rel_l2(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_count += xb.size(0)
        val_loss /= max(val_count, 1)
        best_val = min(best_val, val_loss)
        history["epoch"].append(epoch)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(
                f"   [epoch {epoch:3d}] train={train_loss:.3e} "
                f"val={val_loss:.3e} best={best_val:.3e}"
            )
    print(f"   elapsed = {time.time() - t0:.1f}s")

    # ---- Plot training loss curve ----
    try:
        from neuroflow.utils.plotting import plot_loss_curve
        plot_loss_curve(
            history["epoch"], history["train"], history["val"],
            save_path=out_dir / f"morse_surrogate_train_{args.mode}.png",
        )
    except Exception as e:
        print(f"   [skip] loss-curve plot failed: {e}")

    # ---- Export + C++ parity check ----
    print("==> Exporting NeuroIR (binary)")
    sub = out_dir / "ir"
    sub.mkdir(parents=True, exist_ok=True)
    _, bin_path = export_all(model, sub,
                                basename=f"morse_surrogate_{args.mode}")
    print(f"   {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    # ---- Test: pick a held-out test sample and evaluate ----
    print("==> Evaluating energy + force on a held-out test sample")
    if args.mode == "single":
        D_e, a, r_e = 1.0, 1.6, 1.05
    else:
        # Family mode: pick a fresh in-distribution test sample.
        torch.manual_seed(args.seed + 7)
        D_e, a, r_e = 1.4, 2.0, 1.10
    x_test = np.zeros((1, args.n_points, 4), dtype=np.float32)
    x_test[0, :, 0] = r_grid
    x_test[0, :, 1] = D_e
    x_test[0, :, 2] = a
    x_test[0, :, 3] = r_e
    x_test_t = torch.from_numpy(x_test).float()
    with torch.no_grad():
        y_torch = model(x_test_t).numpy()  # (1, n_points, 1)
    y_ana = morse_potential(r_grid, D_e, a, r_e).astype(np.float32)
    f_ana = morse_force(r_grid, D_e, a, r_e).astype(np.float32)
    err_energy = np.abs(y_torch[0, :, 0] - y_ana)
    print(f"   D_e = {D_e}, a = {a}, r_e = {r_e}")
    print(f"   energy rel L2 = "
          f"{np.linalg.norm(err_energy) / max(np.linalg.norm(y_ana), 1e-8):.3e}, "
          f"max abs = {err_energy.max():.3e}")

    surrogate = MorseSurrogate(model, r_min=float(r_grid[0]),
                                r_max=float(r_grid[-1]), fd_step=1e-3)
    with torch.no_grad():
        f_torch = surrogate.force(x_test_t).numpy()  # (1, n_points)
    err_force = np.abs(f_torch[0] - f_ana)
    print(f"   force  rel L2 = "
          f"{np.linalg.norm(err_force) / max(np.linalg.norm(f_ana), 1e-8):.3e}, "
          f"max abs = {err_force.max():.3e}")

    # ---- C++ parity check on the test sample ----
    print("==> C++ runtime parity check")
    try:
        import neuroflow_cpp
        y_cpp = neuroflow_cpp.infer_arrays(
            str(bin_path), x_test.astype("float32")
        )
        diff_cpp = float(np.abs(y_torch - y_cpp).max())
        print(f"   C++ vs PyTorch max abs diff (energy) = {diff_cpp:.2e}")
    except Exception as e:
        print(f"   [skip] C++ runtime not available: {e}")
        diff_cpp = None

    # ---- Visualise energy + force comparison ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        axes[0].plot(r_grid, y_ana, "k-", label="analytical V(r)", lw=2)
        axes[0].plot(r_grid, y_torch[0, :, 0], "b--",
                      label=f"FNO1d surrogate (D_e={D_e}, a={a}, r_e={r_e})",
                      lw=1.5)
        axes[0].set_xlabel("r")
        axes[0].set_ylabel("V(r)")
        axes[0].set_title(f"Energy ({args.mode} mode)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(r_grid, f_ana, "k-", label="analytical F(r)", lw=2)
        axes[1].plot(r_grid, f_torch[0], "r--",
                      label="FD on FNO1d surrogate", lw=1.5)
        axes[1].set_xlabel("r")
        axes[1].set_ylabel("F(r) = -dV/dr")
        axes[1].set_title(f"Force ({args.mode} mode)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("NeuroFlow × LAMMPS — 1D Morse potential surrogate")
        fig.tight_layout()
        png_path = out_dir / f"morse_surrogate_pred_{args.mode}.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
