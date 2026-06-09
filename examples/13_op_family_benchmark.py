"""Stage 2 Sprint 3.3 — Cross-Operator Family Benchmark on Burgers 1D.

This script trains three operator families on the *same* task
(``Burgers1dDataset`` with ``t_in = t_out = 1`` — predict the next
time step of a 1D Burgers trajectory given the current state) and
records a side-by-side comparison:

    op family  |  #params  |  val rel L2  |  C++ vs PyTorch  |  C++ latency (ms)
    -----------+-----------+--------------+------------------+--------------------
    FNO1d      |  ...
    TokenMixer |  ...
    GraphOp    |  ...

The script writes:
  - artifacts/benchmark/burgers1d_benchmark.csv   (raw table)
  - artifacts/benchmark/burgers1d_benchmark.png   (4-panel bar chart)
  - artifacts/benchmark/<op>_benchmark/...        (per-op exported IR + .pt)

The Burgers 1-step task was chosen because every op family accepts
the same input shape ``(batch, n_points, in_dim) = (b, 64, 1)`` and
emits the same output shape.  This makes the comparison meaningful
(they differ in *how* they model spatial dependencies, not in what
they ingest or emit).

Run:
    python examples/13_op_family_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Make the project root importable so the locally-built `neuroflow_cpp`
# pybind module is discoverable (it lives at D:/minimax_proj/neuroflow_cpp.cp312-win_amd64.pyd).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from neuroflow.data.burgers import Burgers1dConfig, Burgers1dDataset
from neuroflow.ir.export import export_all
from neuroflow.nn.fno import FNO1d, FNO1dConfig
from neuroflow.nn.graph_op import GraphOp, GraphOpConfig
from neuroflow.nn.tokenmixer import TokenMixer, TokenMixerConfig
from neuroflow.utils.plotting import plot_loss_curve


def _build_fno1d(n_points: int) -> FNO1d:
    cfg = FNO1dConfig(
        in_channels=1,
        out_channels=1,
        width=32,
        modes=min(16, n_points // 2),
        n_layers=2,
        activation="gelu",
        pad_factor=1,
        name="fno1d_benchmark",
    )
    return FNO1d(cfg)


def _build_tokenmixer(n_points: int) -> TokenMixer:
    cfg = TokenMixerConfig(
        in_dim=1,
        out_dim=1,
        n_points=n_points,
        n_patches=8 if n_points % 8 == 0 else 4,
        latent_dim=32,
        n_heads=4,
        n_layers=1,
        activation="gelu",
        name="tokenmixer_benchmark",
    )
    return TokenMixer(cfg)


def _build_graphop(n_points: int) -> GraphOp:
    cfg = GraphOpConfig(
        in_dim=1,
        out_dim=1,
        n_nodes=n_points,
        hidden_dim=32,
        n_layers=1,
        activation="gelu",
        name="graphop_benchmark",
    )
    return GraphOp(cfg)


def _train_and_eval(
    op_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> tuple[float, dict[str, list[float]]]:
    """Train `model` and return (val_rel_l2, history)."""
    model = model.to(device)
    optim = Adam(model.parameters(), lr=lr)
    sched = StepLR(optim, step_size=max(1, epochs // 4), gamma=0.5)

    def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).reshape(pred.size(0), -1)
        ref = target.reshape(target.size(0), -1)
        return torch.mean(
            torch.linalg.norm(diff, dim=1)
            / torch.linalg.norm(ref, dim=1).clamp_min(1e-8)
        )

    history = {"epoch": [], "train": [], "val": []}
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss, train_count = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = rel_l2(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * x.size(0)
            train_count += x.size(0)
        train_loss /= max(train_count, 1)
        sched.step()

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = rel_l2(pred, y)
                val_loss += loss.item() * x.size(0)
                val_count += x.size(0)
        val_loss /= max(val_count, 1)
        history["epoch"].append(epoch)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        best_val = min(best_val, val_loss)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(
                f"   [{op_name:11s}] epoch {epoch:3d}  "
                f"train={train_loss:.3e}  val={val_loss:.3e}  "
                f"best={best_val:.3e}"
            )
    return best_val, history


def _cpp_parity_and_latency(
    op_name: str,
    model: torch.nn.Module,
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    out_dir: Path,
    n_bench: int,
) -> tuple[float | None, float | None]:
    """Export the model, run the C++ pybind, return (max_abs_diff, ms_per_call)."""
    try:
        import neuroflow_cpp  # noqa: F401
    except Exception as e:
        print(f"   [{op_name:11s}] C++ extension not available: {e}")
        return None, None

    sub = out_dir / f"{op_name}_benchmark"
    sub.mkdir(parents=True, exist_ok=True)
    _, bin_path = export_all(model, sub, basename=op_name)

    y_torch = model(torch.from_numpy(x_sample)).detach().cpu().numpy().astype("float32")
    y_cpp = neuroflow_cpp.infer_arrays(str(bin_path), x_sample.astype("float32"))
    diff = float(np.abs(y_torch - y_cpp).max())

    # Warm-up
    for _ in range(3):
        neuroflow_cpp.infer_arrays(str(bin_path), x_sample.astype("float32"))
    t0 = time.perf_counter()
    for _ in range(n_bench):
        neuroflow_cpp.infer_arrays(str(bin_path), x_sample.astype("float32"))
    ms_per_call = (time.perf_counter() - t0) / n_bench * 1000.0

    return diff, ms_per_call


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-op benchmark on Burgers 1D")
    parser.add_argument("--n-train-trajs", type=int, default=80)
    parser.add_argument("--n-val-trajs", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bench", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="./artifacts/benchmark")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print("==> Building Burgers 1D 1-step dataset (t_in = t_out = 1)")
    cfg = Burgers1dConfig(
        n_points=args.n_points,
        n_tsteps=20,
        nu=args.nu,
        dt=args.dt,
    )
    train_set = Burgers1dDataset(
        n_samples=args.n_train_trajs, cfg=cfg, t_in=1, t_out=1,
        seed=args.seed, trajectory_stride=1,
    )
    val_set = Burgers1dDataset(
        n_samples=args.n_val_trajs, cfg=cfg, t_in=1, t_out=1,
        seed=args.seed + 1, trajectory_stride=1,
    )
    print(
        f"   n_points = {args.n_points}, train = {len(train_set)}, "
        f"val = {len(val_set)}"
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False
    )

    # A held-out sample for the C++ parity check.
    x_sample, y_sample = val_set[0]
    x_sample = x_sample.unsqueeze(0).numpy().astype("float32")  # (1, n, 1)

    # 3 op families (DeepONet is excluded: its branch/trunk split does
    # not naturally fit a 1-step spatial prediction task).
    op_factories: "OrderedDict[str, callable]" = OrderedDict([
        ("FNO1d", _build_fno1d),
        ("TokenMixer", _build_tokenmixer),
        ("GraphOp", _build_graphop),
    ])

    rows: list[dict[str, object]] = []
    for op_name, factory in op_factories.items():
        print(f"\n==> Training {op_name}")
        torch.manual_seed(args.seed)
        model = factory(args.n_points)
        n_params = model.num_parameters()
        print(f"   parameters = {n_params:,}")

        best_val, history = _train_and_eval(
            op_name, model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device,
        )
        print(f"   best val rel L2 = {best_val:.3e}")

        # Per-op loss curve
        plot_loss_curve(
            history["epoch"], history["train"], history["val"],
            save_path=out_dir / f"{op_name.lower()}_benchmark_train.png",
        )

        # C++ parity + latency.
        diff, ms = _cpp_parity_and_latency(
            op_name, model, x_sample, y_sample, out_dir, args.n_bench
        )
        if diff is not None:
            print(
                f"   C++ vs PyTorch max abs diff = {diff:.2e}; "
                f"latency = {ms:.3f} ms / call"
            )
        rows.append(OrderedDict([
            ("op", op_name),
            ("n_params", n_params),
            ("val_rel_l2", best_val),
            ("cpp_max_abs_diff", diff if diff is not None else float("nan")),
            ("cpp_ms_per_call", ms if ms is not None else float("nan")),
        ]))

    # ---- Write CSV ----
    csv_path = out_dir / "burgers1d_benchmark.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n==> Wrote CSV: {csv_path}")

    # ---- 4-panel figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["op"] for r in rows]
        params = [r["n_params"] for r in rows]
        val_l2 = [r["val_rel_l2"] for r in rows]
        cpp_diff = [r["cpp_max_abs_diff"] for r in rows]
        cpp_ms = [r["cpp_ms_per_call"] for r in rows]

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        x = np.arange(len(names))
        bar_kw = dict(color=["#4c72b0", "#dd8452", "#55a467"])

        axes[0].bar(x, params, **bar_kw)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=15)
        axes[0].set_ylabel("# parameters")
        axes[0].set_title("Capacity")
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].bar(x, val_l2, **bar_kw)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15)
        axes[1].set_ylabel("val rel L2")
        axes[1].set_title("Accuracy (lower = better)")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3, axis="y")

        axes[2].bar(x, cpp_diff, **bar_kw)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=15)
        axes[2].set_ylabel("C++ vs PyTorch max abs diff")
        axes[2].set_title("C++ parity (target < 1e-3)")
        axes[2].axhline(1e-3, color="red", linestyle="--", alpha=0.6,
                        label="1e-3 target")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis="y")

        axes[3].bar(x, cpp_ms, **bar_kw)
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(names, rotation=15)
        axes[3].set_ylabel("C++ latency (ms / call)")
        axes[3].set_title("Inference cost (lower = better)")
        axes[3].grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "NeuroFlow cross-op benchmark — Burgers 1D, 1-step prediction "
            f"(n_points={args.n_points})",
            fontsize=11,
        )
        fig.tight_layout()
        png_path = out_dir / "burgers1d_benchmark.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    # ---- Console summary table ----
    print("\n==> Summary")
    print(f"   {'op':<11s} {'#params':>10s}  {'val rel L2':>11s}  "
          f"{'C++ diff':>10s}  {'C++ ms/call':>11s}")
    for r in rows:
        diff_str = f"{r['cpp_max_abs_diff']:.2e}" if not np.isnan(r["cpp_max_abs_diff"]) else "n/a"
        ms_str = f"{r['cpp_ms_per_call']:.3f}" if not np.isnan(r["cpp_ms_per_call"]) else "n/a"
        print(
            f"   {r['op']:<11s} {int(r['n_params']):>10d}  "
            f"{r['val_rel_l2']:>11.3e}  {diff_str:>10s}  {ms_str:>11s}"
        )

    print("==> Done.")


if __name__ == "__main__":
    main()
