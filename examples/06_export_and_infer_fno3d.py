"""Stage 2 Sprint 2 demo (2/2): C++ inference of the exported FNO3d NeuroIR.

Mirrors examples/04_export_and_infer_fno2d.py for the 3D operator.

Run:
    python examples/06_export_and_infer_fno3d.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sure the project root is on sys.path so the C++ extension is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from neuroflow.ir import NeuroIRSpec
from neuroflow.ir.load import load_neuroir, predict_with_spec
from neuroflow.utils.plotting import plot_2d_field_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 FNO3d inference + benchmark")
    parser.add_argument("--ir", type=str, default="./artifacts/fno3d_poisson.neuroir")
    parser.add_argument("--bin", type=str, default="./artifacts/fno3d_poisson.nneuroir")
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--w", type=int, default=16)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--n-bench", type=int, default=10)
    args = parser.parse_args()

    ir_path = Path(args.ir)
    if not ir_path.exists():
        raise FileNotFoundError(
            f"{ir_path} not found — run 05_train_fno3d.py first."
        )

    print(f"==> Loading NeuroIR (JSON): {ir_path}")
    spec = NeuroIRSpec.load(ir_path)
    print(f"   op = {spec.op}, version = {spec.version}")
    print(f"   config = {spec.config}")
    print(f"   #weights = {len(spec.weights)}")

    pt_model = load_neuroir(spec)
    pt_model.eval()

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((1, args.h, args.w, args.d, 1)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    print("==> PyTorch reference inference")
    with torch.no_grad():
        y_pt = pt_model(x_pt).numpy()
    print(f"   output shape = {y_pt.shape}, mean = {y_pt.mean():.4f}")

    print("==> Pure-Python IR inference (roundtrip)")
    y_py = predict_with_spec(spec, x_np)
    diff_py = np.abs(y_pt - y_py).max()
    print(f"   max abs diff vs PyTorch = {diff_py:.2e}")

    print("==> C++ runtime inference")
    bin_path = Path(args.bin)
    y_cpp = None
    cpp_latency_ms = None
    if not bin_path.exists():
        print(f"   [skip] binary IR not found: {bin_path}")
    else:
        try:
            from neuroflow_cpp import infer as cpp_infer  # type: ignore[import-not-found]
        except Exception as e:
            print(f"   [skip] C++ extension not available: {e}")
        else:
            tmp_in = Path("./artifacts/_tmp_in_fno3d.npy")
            tmp_out = Path("./artifacts/_tmp_out_fno3d.npy")
            np.save(tmp_in, x_np)
            cpp_infer(str(bin_path), str(tmp_in), str(tmp_out))
            y_cpp = np.load(tmp_out)
            diff_cpp = np.abs(y_pt - y_cpp).max()
            print(f"   C++ output shape = {y_cpp.shape}, mean = {y_cpp.mean():.4f}")
            print(f"   max abs diff vs PyTorch = {diff_cpp:.2e}")

            for _ in range(3):
                cpp_infer(str(bin_path), str(tmp_in), str(tmp_out))
            t0 = time.perf_counter()
            for _ in range(args.n_bench):
                cpp_infer(str(bin_path), str(tmp_in), str(tmp_out))
            cpp_latency_ms = (time.perf_counter() - t0) / args.n_bench * 1000
            print(
                f"   C++ latency (batch=1, {args.h}x{args.w}x{args.d}) = "
                f"{cpp_latency_ms:.3f} ms"
            )

    print("==> Pure-Python latency benchmark")
    for _ in range(3):
        predict_with_spec(spec, x_np)
    t0 = time.perf_counter()
    for _ in range(args.n_bench):
        predict_with_spec(spec, x_np)
    py_latency_ms = (time.perf_counter() - t0) / args.n_bench * 1000
    print(f"   Python latency (batch=1, {args.h}x{args.w}x{args.d}) = {py_latency_ms:.3f} ms")

    if cpp_latency_ms is not None:
        speedup = py_latency_ms / cpp_latency_ms
        print(f"   C++ speedup vs Python = {speedup:.1f}x")

    if y_cpp is not None:
        diff_cpp = float(np.abs(y_pt - y_cpp).max())
        title = f"FNO3d: PyTorch vs C++ (max abs diff = {diff_cpp:.2e}, mid-D)"
        cmp_pred = y_cpp[0, ..., args.d // 2, 0]
    else:
        title = "FNO3d: PyTorch vs Python-IR (roundtrip, mid-D)"
        cmp_pred = y_py[0, ..., args.d // 2, 0]
    plot_2d_field_comparison(
        y_pt[0, ..., args.d // 2, 0],
        cmp_pred,
        title=title,
        save_path="./artifacts/fno3d_poisson_roundtrip.png",
    )
    print("==> Done.")


if __name__ == "__main__":
    main()
