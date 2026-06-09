"""Train a tiny FNO1d surrogate for the LAMMPS shim
demo.  The model takes a per-atom 1D signal of length
`n_points=1` with 3 channels (x, y, z) and outputs a
scalar energy.  The "data" is a synthetic Morse-like
potential V(r) = D * (1 - exp(-a * r))^2 - D, where
r = sqrt(x^2 + y^2 + z^2).

This is a stand-in for a real MD force field — the
shim is the Stage 3 deliverable, the surrogate is
just enough to make the shim testable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from neuroflow.ir.export import export_all
from neuroflow.nn.fno import FNO1d, FNO1dConfig


def _morse_at_position(x, D=1.0, a=1.5):
    """Morse potential V(r) = D * (1 - exp(-a * r))^2 - D
    where r = sqrt(x^2 + y^2 + z^2)."""
    r = np.sqrt((x ** 2).sum(axis=-1))
    return D * (1.0 - np.exp(-a * r)) ** 2 - D


def _build_fno1d(n_points: int = 8, width: int = 32,
                  n_layers: int = 3) -> FNO1d:
    cfg = FNO1dConfig(
        in_channels=3,         # x, y, z
        out_channels=1,
        width=width,
        modes=min(4, n_points // 2),
        n_layers=n_layers,
        activation="gelu",
        pad_factor=1,
        name="fno1d_morse_3d",
    )
    return FNO1d(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--out-dir", default="./artifacts/lammps_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    # Generate synthetic data: random 3D points, target
    # is Morse potential.  The per-atom "signal" is
    # a 1D sequence of `n_points` copies of the
    # (x, y, z) position — this gives the FNO a real
    # spatial axis to work with.  In the shim, we
    # broadcast the per-atom feature the same way.
    n_points = 8
    print("==> Generating synthetic Morse dataset")
    n_total = args.n_train + args.n_val
    x_single = np.random.uniform(-1.5, 1.5, size=(n_total, 3)).astype(
        np.float32)  # (n, 3)
    # Broadcast to (n, n_points, 3) by replicating the
    # single 3-vec along the spatial axis.
    x = np.broadcast_to(x_single[:, None, :],
                         (n_total, n_points, 3)).copy()
    y_single = _morse_at_position(x_single).astype(np.float32)  # (n,)
    # Broadcast to (n, n_points, 1).
    y = np.broadcast_to(y_single[:, None, None],
                         (n_total, n_points, 1)).copy()
    x_train, x_val = x[:args.n_train], x[args.n_train:]
    y_train, y_val = y[:args.n_train], y[args.n_train:]
    print(f"   train = {args.n_train}, val = {args.n_val}, "
          f"n_points = {n_points}, V range = [{y.min():.3f}, {y.max():.3f}]")

    print("==> Training FNO1d (3 input channels, 1 output)")
    model = _build_fno1d()
    print(f"   parameters = {model.num_parameters():,}")
    optim = Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train),
                       torch.from_numpy(y_train)),
        batch_size=32, shuffle=True, drop_last=True,
    )
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss, n = 0.0, 0
        for xb, yb in train_loader:
            optim.zero_grad()
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= max(n, 1)
        model.eval()
        with torch.no_grad():
            val_loss = ((model(x_val_t) - y_val_t) ** 2).mean().item()
        best_val = min(best_val, val_loss)
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(f"   [epoch {epoch:3d}] train={train_loss:.3e} "
                  f"val={val_loss:.3e} best={best_val:.3e}")
    print(f"   best val MSE = {best_val:.3e}")

    # Export to .nneuroir
    sub = out_dir / "ir"
    sub.mkdir(parents=True, exist_ok=True)
    _, bin_path = export_all(model, sub, basename="fno1d_morse_3d")
    print(f"==> Wrote {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")
    return bin_path


if __name__ == "__main__":
    main()
