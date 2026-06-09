"""Stage 1 demo (2/2): C++ inference of the exported NeuroIR.

This script:
    1. Loads the .neuroir (JSON) from disk.
    2. Tries to call the compiled C++ runtime via pybind11.
    3. Falls back to pure-Python inference if the C++ extension is not built.
    4. Compares C++ output to PyTorch output (roundtrip precision check).
    5. Runs a small latency benchmark.

Run:
    python examples/02_export_and_infer.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sure the project root is on sys.path so the C++ extension
# (copied next to README.md as `neuroflow_cpp.pyd`) is importable
# regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from neuroflow.ir import NeuroIRSpec
from neuroflow.ir.load import load_neuroir, predict_with_spec
from neuroflow.utils.plotting import plot_burgers_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 C++ inference + benchmark")
    parser.add_argument("--ir", type=str, default="./artifacts/fno1d_burgers.neuroir")
    parser.add_argument("--bin", type=str, default="./artifacts/fno1d_burgers.nneuroir")
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--n-bench", type=int, default=50)
    args = parser.parse_args()

    ir_path = Path(args.ir)
    if not ir_path.exists():
        raise FileNotFoundError(
            f"{ir_path} not found — run 01_train_burgers1d.py first."
        )

    print(f"==> Loading NeuroIR (JSON): {ir_path}")
    spec = NeuroIRSpec.load(ir_path)
    print(f"   op = {spec.op}, version = {spec.version}")
    print(f"   config = {spec.config}")
    print(f"   #weights = {len(spec.weights)}")

    # Reconstruct PyTorch model
    pt_model = load_neuroir(spec)
    pt_model.eval()

    # Build a sample input
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((1, args.n_points, args.t_in)).astype(np.float32)
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
    if not bin_path.exists():
        print(f"   [skip] binary IR not found: {bin_path}")
        y_cpp = None
        cpp_latency_ms = None
    else:
        y_cpp = None
        cpp_latency_ms = None
        try:
            from neuroflow_cpp import infer as cpp_infer  # type: ignore[import-not-found]
        except Exception as e:
            print(f"   [skip] C++ extension not available: {e}")
            print("   To enable: build the C++ runtime (see cpp/CMakeLists.txt).")
        else:
            tmp_in = Path("./artifacts/_tmp_in.npy")
            tmp_out = Path("./artifacts/_tmp_out.npy")
            tmp_in.parent.mkdir(parents=True, exist_ok=True)
            np.save(tmp_in, x_np)
            cpp_infer(str(bin_path), str(tmp_in), str(tmp_out))
            y_cpp = np.load(tmp_out)
            diff_cpp = np.abs(y_pt - y_cpp).max()
            print(f"   C++ output shape = {y_cpp.shape}, mean = {y_cpp.mean():.4f}")
            print(f"   max abs diff vs PyTorch = {diff_cpp:.2e}")

            # Latency benchmark
            for _ in range(5):
                cpp_infer(str(bin_path), str(tmp_in), str(tmp_out))
            t0 = time.perf_counter()
            for _ in range(args.n_bench):
                cpp_infer(str(bin_path), str(tmp_in), str(tmp_out))
            cpp_latency_ms = (time.perf_counter() - t0) / args.n_bench * 1000
            print(f"   C++ latency (batch=1, n={args.n_points}) = {cpp_latency_ms:.3f} ms")

    print("==> Pure-Python latency benchmark")
    for _ in range(3):
        predict_with_spec(spec, x_np)
    t0 = time.perf_counter()
    for _ in range(args.n_bench):
        predict_with_spec(spec, x_np)
    py_latency_ms = (time.perf_counter() - t0) / args.n_bench * 1000
    print(f"   Python latency (batch=1, n={args.n_points}) = {py_latency_ms:.3f} ms")

    if cpp_latency_ms is not None:
        speedup = py_latency_ms / cpp_latency_ms
        print(f"   C++ speedup vs Python = {speedup:.1f}x")

    # Plot a sample
    x_grid = np.linspace(0.0, 2 * np.pi, args.n_points, endpoint=False)
    y_pt_squeezed = y_pt[0].T  # (n, t_out) -> (t_out, n)
    plot_burgers_prediction(
        x_grid,
        y_pt_squeezed,
        y_py[0].T,
        title="FNO1d: PyTorch vs Python-IR (roundtrip)",
        save_path="./artifacts/fno1d_burgers_roundtrip.png",
    )
    print("==> Done.")


if __name__ == "__main__":
    main()
