"""Stage 2 Sprint 3.5 — Cross-Operator Family Benchmark on 2D Poisson.

Mirrors examples/14 (1D Burgers multi-seed) but on the 2D Poisson
task using the existing `Heat2dDataset`.  Trains three operator
families on the same task and reports multi-seed val rel L2:

    FNO2d       (spectral, NeuroIR v0.2.0)
    TokenMixer2D (attention over flattened patches, new in Sprint 3.5)
    GraphOp2D    (8-conn grid message passing, new in Sprint 3.5)

This is the **Python-only** counterpart of the 1D cross-op
benchmark; the C++ parity path for TokenMixer2D / GraphOp2D
is queued for Sprint 3.6.

Run:
    python examples/15_2d_cross_op_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

# Make the project root importable so any locally-built
# `neuroflow_cpp` pybind module is discoverable (not used here
# yet but kept for symmetry with the 1D benchmark).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from neuroflow.data.heat2d import Heat2dConfig, Heat2dDataset
from neuroflow.nn.fno2d import FNO2d, FNO2dConfig
from neuroflow.nn.graph_op2d import GraphOp2D, GraphOp2DConfig
from neuroflow.nn.tokenmixer2d import TokenMixer2D, TokenMixer2DConfig


# --- op factories (Stage 2 demo defaults) ---------------------------------

def _build_fno2d(h: int, w: int) -> FNO2d:
    cfg = FNO2dConfig(
        in_channels=1,
        out_channels=1,
        width=24,
        modes_h=min(8, h // 2),
        modes_w=min(8, w // 2),
        n_layers=2,
        activation="gelu",
        pad_factor=1,
        name="fno2d_2dbench",
    )
    return FNO2d(cfg)


def _build_tokenmixer2d(h: int, w: int) -> TokenMixer2D:
    n_points = h * w
    # Pick the largest power of two <= n_points that is a valid
    # divisor (for the mean-pool slice). Fall back to a small value.
    n_patches = 1
    for cand in (256, 128, 64, 32, 16, 8, 4, 2):
        if cand <= n_points and n_points % cand == 0:
            n_patches = cand
            break
    if n_patches == 1:
        n_patches = 4  # safety: pick a tiny value
    cfg = TokenMixer2DConfig(
        in_dim=1, out_dim=1, h=h, w=w,
        n_patches=n_patches,
        latent_dim=32, n_heads=4, n_layers=1,
        activation="gelu",
        name="tokenmixer2d_2dbench",
    )
    return TokenMixer2D(cfg)


def _build_graphop2d(h: int, w: int) -> GraphOp2D:
    cfg = GraphOp2DConfig(
        in_dim=1, out_dim=1, h=h, w=w,
        hidden_dim=32, n_layers=1,
        activation="gelu",
        name="graphop2d_2dbench",
    )
    return GraphOp2D(cfg)


def _make_dataloaders(
    h: int, w: int, n_train: int, n_val: int, batch_size: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    cfg = Heat2dConfig(h=h, w=w, seed=seed)
    train_set = Heat2dDataset(n_samples=n_train, cfg=cfg, seed=seed)
    val_set = Heat2dDataset(n_samples=n_val, cfg=cfg, seed=seed + 1)
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
    parser = argparse.ArgumentParser(
        description="Multi-seed cross-op benchmark on 2D Poisson (Python-only)"
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--w", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=str, default="./artifacts/benchmark")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    op_factories: "OrderedDict[str, callable]" = OrderedDict([
        ("FNO2d", _build_fno2d),
        ("TokenMixer2D", _build_tokenmixer2d),
        ("GraphOp2D", _build_graphop2d),
    ])

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    print(f"==> 2D Poisson cross-op benchmark: {len(seeds)} seeds × "
          f"{len(op_factories)} ops = {len(seeds) * len(op_factories)} runs "
          f"on a {args.h}×{args.w} grid")

    per_seed: list[dict[str, object]] = []
    for op_name, factory in op_factories.items():
        n_params = factory(args.h, args.w).num_parameters()
        for seed in seeds:
            print(f"\n--- {op_name} | seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_loader, val_loader = _make_dataloaders(
                args.h, args.w, args.n_train, args.n_val,
                args.batch_size, seed,
            )
            torch.manual_seed(seed)  # re-seed for the model
            model = factory(args.h, args.w)
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
    per_seed_csv = out_dir / "poisson2d_benchmark_per_seed.csv"
    with per_seed_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        writer.writeheader()
        for row in per_seed:
            writer.writerow(row)
    print(f"\n==> Wrote per-seed CSV: {per_seed_csv}")

    # ---- summary ----
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in per_seed:
        grouped[row["op"]].append(float(row["val_rel_l2"]))

    summary: list[dict[str, object]] = []
    print(f"\n==> Summary (over {len(seeds)} seeds, 2D Poisson, "
          f"{args.h}×{args.w})")
    print(f"   {'op':<14s} {'#params':>10s}  {'mean rel L2':>13s}  "
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
        print(f"   {op_name:<14s} {int(params):>10d}  {m:>13.3e}  {s:>10.2e}  "
              f"{lo:>10.3e}  {hi:>10.3e}")

    summary_csv = out_dir / "poisson2d_benchmark_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(f"==> Wrote summary CSV: {summary_csv}")

    # ---- figure (3-panel: capacity / accuracy / seed-stability) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [s["op"] for s in summary]
        params = [s["n_params"] for s in summary]
        means = [s["val_rel_l2_mean"] for s in summary]
        stds = [s["val_rel_l2_std"] for s in summary]
        cvs = [s / m if m > 0 else 0.0 for s, m in zip(stds, means)]

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        bar_kw = dict(color=["#4c72b0", "#dd8452", "#55a467"], capsize=4)
        x = np.arange(len(names))

        axes[0].bar(x, params, **bar_kw)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=15)
        axes[0].set_ylabel("# parameters")
        axes[0].set_title("Capacity")
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].bar(x, means, yerr=stds, **bar_kw)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15)
        axes[1].set_ylabel("val rel L2 (mean ± std)")
        axes[1].set_title(f"Accuracy over {len(seeds)} seeds (log)")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3, axis="y")

        axes[2].bar(x, cvs, **bar_kw)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=15)
        axes[2].set_ylabel("std / mean")
        axes[2].set_title("Seed sensitivity (lower = more stable)")
        axes[2].grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"Multi-seed cross-op benchmark — 2D Poisson, "
            f"{args.h}×{args.w} grid ({len(seeds)} seeds)",
            fontsize=11,
        )
        fig.tight_layout()
        png_path = out_dir / "poisson2d_benchmark_multi_seed.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
