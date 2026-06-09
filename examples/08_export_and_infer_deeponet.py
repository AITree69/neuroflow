"""Stage 2 Sprint 2 demo (4/4): C++ inference of the exported DeepONet NeuroIR.

Mirrors examples/04/06 for DeepONet, with the difference that DeepONet
takes TWO input arrays (u for the branch net, y for the trunk net).

Run:
    python examples/08_export_and_infer_deeponet.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from neuroflow.ir import NeuroIRSpec
from neuroflow.ir.load import load_neuroir, predict_with_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 DeepONet inference + benchmark")
    parser.add_argument("--ir", type=str, default="./artifacts/deeponet_integral.neuroir",
                        help="JSON IR (for PyTorch ref + Python IR roundtrip)")
    parser.add_argument("--bin", type=str, default="./artifacts/deeponet_integral.nneuroir",
                        help="Binary IR (for C++ pybind11 inference)")
    parser.add_argument("--n-sensor", type=int, default=100)
    parser.add_argument("--n-query", type=int, default=50)
    parser.add_argument("--n-bench", type=int, default=20)
    args = parser.parse_args()

    ir_path = Path(args.ir)
    if not ir_path.exists():
        raise FileNotFoundError(
            f"{ir_path} not found — run 07_train_deeponet.py first."
        )

    print(f"==> Loading NeuroIR (JSON): {ir_path}")
    spec = NeuroIRSpec.load(ir_path)
    print(f"   op = {spec.op}, version = {spec.version}")
    print(f"   config = {spec.config}")
    print(f"   #weights = {len(spec.weights)}")

    pt_model = load_neuroir(spec)
    pt_model.eval()

    rng = np.random.default_rng(42)
    # Branch input is the stacked [s_i, u(s_i)] feature (in_branch=2).
    s_grid = (np.arange(args.n_sensor) + 0.5) / args.n_sensor
    u_vals = rng.standard_normal(args.n_sensor).astype(np.float32)
    u_np = np.stack([s_grid, u_vals], axis=-1).reshape(1, args.n_sensor, 2).astype(np.float32)
    y_np = np.linspace(0.0, 1.0, args.n_query, endpoint=False, dtype=np.float32).reshape(1, args.n_query, 1)
    u_pt = torch.from_numpy(u_np)
    y_pt = torch.from_numpy(y_np)

    print("==> PyTorch reference inference")
    with torch.no_grad():
        out_pt = pt_model(u_pt, y_pt).numpy()
    print(f"   output shape = {out_pt.shape}, mean = {out_pt.mean():.4f}")

    print("==> Pure-Python IR inference (roundtrip)")
    out_py = predict_with_spec(spec, u_np, y_np)
    diff_py = np.abs(out_pt - out_py).max()
    print(f"   max abs diff vs PyTorch = {diff_py:.2e}")

    print("==> C++ runtime inference")
    y_cpp = None
    cpp_latency_ms = None
    bin_path = Path(args.bin)
    if not bin_path.exists():
        print(f"   [skip] binary IR not found: {bin_path}")
    else:
        try:
            from neuroflow_cpp import infer_deeponet_arrays  # type: ignore[import-not-found]
        except Exception as e:
            print(f"   [skip] C++ extension not available: {e}")
        else:
            y_cpp = infer_deeponet_arrays(str(bin_path), u_np, y_np)
            diff_cpp = np.abs(out_pt - y_cpp).max()
            print(f"   C++ output shape = {y_cpp.shape}, mean = {y_cpp.mean():.4f}")
            print(f"   max abs diff vs PyTorch = {diff_cpp:.2e}")

            for _ in range(5):
                infer_deeponet_arrays(str(bin_path), u_np, y_np)
            t0 = time.perf_counter()
            for _ in range(args.n_bench):
                infer_deeponet_arrays(str(bin_path), u_np, y_np)
            cpp_latency_ms = (time.perf_counter() - t0) / args.n_bench * 1000
        print(
            f"   C++ latency (batch=1, {args.n_sensor} sensors, "
            f"{args.n_query} queries) = {cpp_latency_ms:.3f} ms"
        )

    print("==> Pure-Python latency benchmark")
    for _ in range(3):
        predict_with_spec(spec, u_np, y_np)
    t0 = time.perf_counter()
    for _ in range(args.n_bench):
        predict_with_spec(spec, u_np, y_np)
    py_latency_ms = (time.perf_counter() - t0) / args.n_bench * 1000
    print(
        f"   Python latency (batch=1, {args.n_sensor} sensors, "
        f"{args.n_query} queries) = {py_latency_ms:.3f} ms"
    )

    if cpp_latency_ms is not None:
        speedup = py_latency_ms / cpp_latency_ms
        print(f"   C++ speedup vs Python = {speedup:.1f}x")

    print("==> Done.")


if __name__ == "__main__":
    main()
