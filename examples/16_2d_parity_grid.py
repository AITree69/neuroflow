"""Stage 2 Sprint 3.7 — Extended C++ parity grid for the 2D operator
families.

Sweep across grid size, number of patches, number of attention
heads, hidden width, and number of transformer / GCN layers.
For each (TokenMixer2D or GraphOp2D) configuration we:
  1. Build the model with a deterministic seed.
  2. Export to NeuroIR.
  3. Run a single held-out sample through both the PyTorch
     reference and the C++ runtime (`neuroflow_cpp.infer_arrays`).
  4. Record max-abs-diff, C++ per-call latency, and
     parameter count.

Outputs (under ./artifacts/benchmark/):
  - 2d_parity_grid_per_config.csv
  - 2d_parity_grid_summary.csv (per-op mean / max / std of diff
    across all configs)
  - 2d_parity_grid.png  (heatmap of max-abs-diff by op × config)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from neuroflow.ir.export import export_all
from neuroflow.nn.graph_op2d import GraphOp2D, GraphOp2DConfig
from neuroflow.nn.tokenmixer2d import TokenMixer2D, TokenMixer2DConfig


def _build_tokenmixer2d(h: int, w: int, n_patches: int,
                         latent_dim: int, n_heads: int, n_layers: int) -> TokenMixer2D:
    n_points = h * w
    # Pick a valid divisor if the requested n_patches is invalid.
    if n_points % n_patches != 0:
        for cand in (256, 128, 64, 32, 16, 8, 4, 2, 1):
            if n_points % cand == 0:
                n_patches = cand
                break
    cfg = TokenMixer2DConfig(
        in_dim=1, out_dim=1, h=h, w=w,
        n_patches=n_patches, latent_dim=latent_dim,
        n_heads=n_heads, n_layers=n_layers,
        activation="gelu", name="tm2d_grid",
    )
    return TokenMixer2D(cfg)


def _build_graphop2d(h: int, w: int, hidden_dim: int,
                      n_layers: int) -> GraphOp2D:
    cfg = GraphOp2DConfig(
        in_dim=1, out_dim=1, h=h, w=w,
        hidden_dim=hidden_dim, n_layers=n_layers,
        activation="gelu", name="gcn2d_grid",
    )
    return GraphOp2D(cfg)


def _grid_configs() -> "list[tuple[str, dict]]":
    """Generate the (op, config) sweep.

    Each row is a dict with the kwargs needed by the corresponding
    factory.  The sweep is intentionally small (a few dozen
    configurations total) to keep the wall-clock under a few
    minutes; configurations are chosen to cover the dimensions
    of interest (spatial size, latent width, attention
    structure, depth).
    """
    grid: list[tuple[str, dict]] = []
    # TokenMixer2D sweep:
    # (h, w, n_patches, latent_dim, n_heads, n_layers)
    for h, w in [(8, 8), (16, 16), (32, 32)]:
        for n_patches in [4, 16]:
            for latent_dim, n_heads in [(16, 2), (32, 4)]:
                grid.append((
                    "TokenMixer2D",
                    dict(h=h, w=w, n_patches=n_patches,
                         latent_dim=latent_dim, n_heads=n_heads,
                         n_layers=1),
                ))
    # GraphOp2D sweep:
    for h, w in [(8, 8), (16, 16), (32, 32)]:
        for hidden_dim in [8, 16, 32]:
            grid.append((
                "GraphOp2D",
                dict(h=h, w=w, hidden_dim=hidden_dim, n_layers=1),
            ))
    return grid


def _run_one(op: str, kwargs: dict, n_bench: int) -> dict:
    torch.manual_seed(0)
    if op == "TokenMixer2D":
        model = _build_tokenmixer2d(**kwargs)
    else:
        model = _build_graphop2d(**kwargs)
    model.eval()
    h, w = kwargs["h"], kwargs["w"]
    x = torch.randn(2, h, w, 1)
    y_torch = model(x).detach().cpu().numpy().astype("float32")

    import neuroflow_cpp
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _, bin_path = export_all(model, td, op)
        y_cpp = neuroflow_cpp.infer_arrays(str(bin_path),
                                            x.numpy().astype("float32"))
        # Warm up + measure latency.
        for _ in range(3):
            neuroflow_cpp.infer_arrays(str(bin_path),
                                        x.numpy().astype("float32"))
        t0 = time.perf_counter()
        for _ in range(n_bench):
            neuroflow_cpp.infer_arrays(str(bin_path),
                                        x.numpy().astype("float32"))
        ms = (time.perf_counter() - t0) / n_bench * 1000.0

    diff = float(np.abs(y_torch - y_cpp).max())
    return {
        "op": op,
        "h": h,
        "w": w,
        "n_patches": kwargs.get("n_patches", 0),
        "latent_dim": kwargs.get("latent_dim", 0),
        "n_heads": kwargs.get("n_heads", 0),
        "hidden_dim": kwargs.get("hidden_dim", 0),
        "n_layers": kwargs.get("n_layers", 1),
        "n_params": model.num_parameters(),
        "max_abs_diff": diff,
        "cpp_ms_per_call": ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D operator-family C++ parity grid sweep"
    )
    parser.add_argument("--n-bench", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="./artifacts/benchmark")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = _grid_configs()
    print(f"==> Running {len(configs)} configurations")

    rows: list[dict] = []
    for i, (op, kwargs) in enumerate(configs, 1):
        try:
            r = _run_one(op, kwargs, args.n_bench)
        except Exception as e:
            print(f"  [{i:3d}] {op:14s} {kwargs}  FAILED: {e}")
            continue
        print(
            f"  [{i:3d}] {op:14s} h=w={r['h']:2d}/{r['w']:2d}  "
            f"params={r['n_params']:>7d}  diff={r['max_abs_diff']:.2e}  "
            f"lat={r['cpp_ms_per_call']:.3f} ms"
        )
        rows.append(r)

    per_csv = out_dir / "2d_parity_grid_per_config.csv"
    with per_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n==> Wrote per-config CSV: {per_csv}")

    # ---- summary per op ----
    grouped: dict[str, list[float]] = defaultdict(list)
    grouped_lat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        grouped[r["op"]].append(r["max_abs_diff"])
        grouped_lat[r["op"]].append(r["cpp_ms_per_call"])
    summary: list[dict] = []
    print(f"\n==> Summary over {len(rows)} configurations")
    print(f"   {'op':<14s}  {'mean diff':>11s}  {'max diff':>11s}  "
          f"{'std':>10s}  {'mean lat (ms)':>15s}")
    for op in ("TokenMixer2D", "GraphOp2D"):
        if op not in grouped:
            continue
        diffs = grouped[op]
        lats = grouped_lat[op]
        summary.append({
            "op": op,
            "n_configs": len(diffs),
            "mean_max_abs_diff": float(np.mean(diffs)),
            "max_max_abs_diff": float(np.max(diffs)),
            "std_max_abs_diff": float(np.std(diffs, ddof=0)),
            "mean_cpp_ms_per_call": float(np.mean(lats)),
        })
        print(
            f"   {op:<14s}  {np.mean(diffs):>11.2e}  {np.max(diffs):>11.2e}  "
            f"{np.std(diffs, ddof=0):>10.2e}  {np.mean(lats):>15.3f}"
        )

    sum_csv = out_dir / "2d_parity_grid_summary.csv"
    with sum_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for r in summary:
            writer.writerow(r)
    print(f"==> Wrote summary CSV: {sum_csv}")

    # ---- heatmap figure (parity by grid size per op) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

        for ax_i, op in enumerate(("TokenMixer2D", "GraphOp2D")):
            ax = axes[ax_i]
            sub = [r for r in rows if r["op"] == op]
            if not sub:
                ax.set_visible(False)
                continue
            # Group by (h, w) and plot max-abs-diff bars split by config.
            # Sort configs by (h, w) then n_patches (TM) or hidden_dim (GCN).
            sub.sort(key=lambda r: (r["h"], r["n_patches"] or r["hidden_dim"]))
            x_labels = [f"{r['h']:2d}x{r['w']:2d}/p{r['n_patches'] or r['hidden_dim']:>2d}"
                          for r in sub]
            diffs = [r["max_abs_diff"] for r in sub]
            x = np.arange(len(sub))
            ax.bar(x, diffs, color=("#4c72b0" if op == "TokenMixer2D" else "#dd8452"))
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=70, fontsize=7)
            ax.set_yscale("log")
            ax.axhline(1e-3, color="red", linestyle="--", alpha=0.6,
                       label="1e-3 target")
            ax.set_ylabel("C++ vs PyTorch max abs diff")
            ax.set_title(f"{op} ({len(sub)} configs)")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"2D operator-family C++ parity grid sweep "
            f"({len(rows)} configurations, target < 1e-3)",
            fontsize=11,
        )
        fig.tight_layout()
        png_path = out_dir / "2d_parity_grid.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
