"""Stage 2 Sprint 3.4 — Multi-seed cross-op benchmark on Burgers 1D.

Reuses the single-seed benchmark from example 13 and repeats it
across 5 random seeds.  We report mean ± std validation relative
L2 for each op family, plus a per-seed CSV.

The Stage 1 paper's "Threats to Validity" section already lists
"single seed" as a caveat; this script is the Stage 2 closure
of that caveat for the cross-op benchmark.

Run:
    python examples/14_multi_seed_benchmark.py

Outputs (under ./artifacts/benchmark/):
    - burgers1d_benchmark_per_seed.csv   (15 rows: 5 seeds × 3 ops)
    - burgers1d_benchmark_summary.csv    (3 rows: one per op, mean ± std)
    - burgers1d_benchmark_multi_seed.png (3-panel bar chart with error bars)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

# Make the project root importable so the locally-built `neuroflow_cpp`
# pybind module is discoverable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from neuroflow.data.burgers import Burgers1dConfig, Burgers1dDataset
from neuroflow.nn.fno import FNO1d, FNO1dConfig
from neuroflow.nn.graph_op import GraphOp, GraphOpConfig
from neuroflow.nn.tokenmixer import TokenMixer, TokenMixerConfig


# --- op factories (mirror example 13) ---------------------------------------

def _build_fno1d(n_points: int) -> FNO1d:
    cfg = FNO1dConfig(
        in_channels=1,
        out_channels=1,
        width=32,
        modes=min(16, n_points // 2),
        n_layers=2,
        activation="gelu",
        pad_factor=1,
        name="fno1d_multiseed",
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
        name="tokenmixer_multiseed",
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
        name="graphop_multiseed",
    )
    return GraphOp(cfg)


def _make_dataloaders(
    n_points: int,
    n_train_trajs: int,
    n_val_trajs: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    cfg = Burgers1dConfig(n_points=n_points, n_tsteps=20, nu=0.01, dt=0.01)
    train_set = Burgers1dDataset(
        n_samples=n_train_trajs, cfg=cfg, t_in=1, t_out=1,
        seed=seed, trajectory_stride=1,
    )
    val_set = Burgers1dDataset(
        n_samples=n_val_trajs, cfg=cfg, t_in=1, t_out=1,
        seed=seed + 1, trajectory_stride=1,
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
    )


def _train_one(
    op_name: str,
    seed: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> float:
    """Train `model` with the given seed and return best validation rel L2."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    optim = Adam(model.parameters(), lr=lr)
    sched = StepLR(optim, step_size=max(1, epochs // 4), gamma=0.5)

    def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).reshape(pred.size(0), -1)
        ref = target.reshape(target.size(0), -1)
        return torch.mean(
            torch.linalg.norm(diff, dim=1)
            / torch.linalg.norm(ref, dim=1).clamp_min(1e-8)
        )

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optim.zero_grad()
            pred = model(x)
            loss = rel_l2(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = rel_l2(pred, y)
                val_loss += loss.item() * x.size(0)
                val_count += x.size(0)
        val_loss /= max(val_count, 1)
        best_val = min(best_val, val_loss)
    return best_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed cross-op benchmark on Burgers 1D")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--n-train-trajs", type=int, default=80)
    parser.add_argument("--n-val-trajs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=str, default="./artifacts/benchmark")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    op_factories: "OrderedDict[str, callable]" = OrderedDict([
        ("FNO1d", _build_fno1d),
        ("TokenMixer", _build_tokenmixer),
        ("GraphOp", _build_graphop),
    ])

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    print(f"==> Running {len(seeds)} seeds × {len(op_factories)} ops "
          f"= {len(seeds) * len(op_factories)} runs")

    per_seed: list[dict[str, object]] = []
    for op_name, factory in op_factories.items():
        n_params = factory(args.n_points).num_parameters()
        for seed in seeds:
            print(f"\n--- {op_name} | seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_loader, val_loader = _make_dataloaders(
                n_points=args.n_points,
                n_train_trajs=args.n_train_trajs,
                n_val_trajs=args.n_val_trajs,
                batch_size=args.batch_size,
                seed=seed,
            )
            torch.manual_seed(seed)  # re-seed for the model
            model = factory(args.n_points)
            t0 = time.time()
            best_val = _train_one(
                op_name, seed, model, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr,
            )
            dt = time.time() - t0
            print(f"   best val rel L2 = {best_val:.3e}  (took {dt:.1f}s, "
                  f"n_params = {n_params:,})")
            per_seed.append({
                "op": op_name,
                "seed": seed,
                "n_params": n_params,
                "val_rel_l2": best_val,
            })

    # ---- per-seed CSV ----
    per_seed_csv = out_dir / "burgers1d_benchmark_per_seed.csv"
    with per_seed_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        writer.writeheader()
        for row in per_seed:
            writer.writerow(row)
    print(f"\n==> Wrote per-seed CSV: {per_seed_csv}")

    # ---- summary CSV (mean / std / min / max) ----
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in per_seed:
        grouped[row["op"]].append(float(row["val_rel_l2"]))

    summary: list[dict[str, object]] = []
    print(f"\n==> Summary (over {len(seeds)} seeds)")
    print(f"   {'op':<11s} {'#params':>8s}  {'mean rel L2':>13s}  "
          f"{'std':>10s}  {'min':>10s}  {'max':>10s}")
    for op_name in op_factories.keys():
        vals = grouped[op_name]
        params = next(r["n_params"] for r in per_seed if r["op"] == op_name)
        m, s = float(np.mean(vals)), float(np.std(vals, ddof=0))
        lo, hi = float(np.min(vals)), float(np.max(vals))
        summary.append({
            "op": op_name,
            "n_params": params,
            "val_rel_l2_mean": m,
            "val_rel_l2_std": s,
            "val_rel_l2_min": lo,
            "val_rel_l2_max": hi,
        })
        print(f"   {op_name:<11s} {int(params):>8d}  {m:>13.3e}  {s:>10.2e}  "
              f"{lo:>10.3e}  {hi:>10.3e}")

    summary_csv = out_dir / "burgers1d_benchmark_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(f"==> Wrote summary CSV: {summary_csv}")

    # ---- 3-panel figure (mean ± std) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [s["op"] for s in summary]
        params = [s["n_params"] for s in summary]
        means = [s["val_rel_l2_mean"] for s in summary]
        stds = [s["val_rel_l2_std"] for s in summary]

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        bar_kw = dict(color=["#4c72b0", "#dd8452", "#55a467"], capsize=4)

        x = np.arange(len(names))

        # 1. Parameter count (no error bar, deterministic).
        axes[0].bar(x, params, **bar_kw)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=15)
        axes[0].set_ylabel("# parameters")
        axes[0].set_title("Capacity")
        axes[0].grid(True, alpha=0.3, axis="y")

        # 2. Val rel L2 mean ± std.
        axes[1].bar(x, means, yerr=stds, **bar_kw)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15)
        axes[1].set_ylabel("val rel L2 (mean ± std)")
        axes[1].set_title(f"Accuracy over {len(seeds)} seeds (log)")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3, axis="y")

        # 3. Coefficient of variation (std / mean): how *stable* each op is.
        cvs = [s / m if m > 0 else 0.0 for s, m in zip(stds, means)]
        axes[2].bar(x, cvs, **bar_kw)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=15)
        axes[2].set_ylabel("std / mean")
        axes[2].set_title("Seed sensitivity (lower = more stable)")
        axes[2].grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"Multi-seed cross-op benchmark — Burgers 1D, 1-step prediction "
            f"({len(seeds)} seeds, n_points={args.n_points})",
            fontsize=11,
        )
        fig.tight_layout()
        png_path = out_dir / "burgers1d_benchmark_multi_seed.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
