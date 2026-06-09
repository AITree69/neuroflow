"""Stage 2 Sprint 3 demo (2/2): C++ inference of the exported TokenMixer
NeuroIR.

Mirrors examples/02 / 04 / 06 / 08, but for the TokenMixer (Transolver-style)
operator. Loads the .neuroir (JSON) for the PyTorch reference + pure-NumPy
roundtrip, then loads the .nneuroir (binary) into the C++ runtime and
verifies parity.

Run:
    python examples/10_export_and_infer_tokenmixer.py
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

from neuroflow.data.token_mixer_demo import TokenMixerDemo1dConfig, TokenMixerDemo1dDataset
from neuroflow.ir import NeuroIRSpec
from neuroflow.ir.load import load_neuroir, predict_with_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 TokenMixer inference + benchmark")
    parser.add_argument("--ir", type=str, default="./artifacts/tokenmixer_demo.neuroir",
                        help="JSON IR (for PyTorch ref + Python IR roundtrip)")
    parser.add_argument("--bin", type=str, default="./artifacts/tokenmixer_demo.nneuroir",
                        help="Binary IR (for C++ pybind11 inference)")
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--n-bench", type=int, default=20)
    args = parser.parse_args()

    ir_path = Path(args.ir)
    if not ir_path.exists():
        raise FileNotFoundError(
            f"{ir_path} not found — run 09_train_tokenmixer.py first."
        )

    print(f"==> Loading NeuroIR (JSON): {ir_path}")
    spec = NeuroIRSpec.load(ir_path)
    print(f"   op = {spec.op}, version = {spec.version}")
    print(f"   config = {spec.config}")
    print(f"   #weights = {len(spec.weights)}")

    pt_model = load_neuroir(spec)
    pt_model.eval()

    # Build a deterministic batch from the demo dataset (val sample 0).
    val_set = TokenMixerDemo1dDataset(
        n_samples=1, cfg=TokenMixerDemo1dConfig(n_points=args.n_points, seed=99), seed=99
    )
    x, _ = val_set[0]
    x_np = x.unsqueeze(0).numpy().astype(np.float32)  # (1, n_points, 2)

    print("==> PyTorch reference inference")
    import torch
    with torch.no_grad():
        out_pt = pt_model(x.unsqueeze(0)).numpy()
    print(f"   output shape = {out_pt.shape}, mean = {out_pt.mean():.4f}")

    print("==> Pure-Python IR inference (roundtrip)")
    out_py = predict_with_spec(spec, x_np)
    diff_py = np.abs(out_pt - out_py).max()
    print(f"   max abs diff vs PyTorch = {diff_py:.2e}")

    print("==> C++ runtime inference")
    cpp_latency_ms = None
    bin_path = Path(args.bin)
    if not bin_path.exists():
        print(f"   [skip] binary IR not found: {bin_path}")
    else:
        try:
            from neuroflow_cpp import infer_arrays  # type: ignore[import-not-found]
        except Exception as e:
            print(f"   [skip] C++ extension not available: {e}")
        else:
            y_cpp = infer_arrays(str(bin_path), x_np)
            diff_cpp = np.abs(out_pt - y_cpp).max()
            print(f"   C++ output shape = {y_cpp.shape}, mean = {y_cpp.mean():.4f}")
            print(f"   max abs diff vs PyTorch = {diff_cpp:.2e}")

            for _ in range(5):
                infer_arrays(str(bin_path), x_np)
            t0 = time.perf_counter()
            for _ in range(args.n_bench):
                infer_arrays(str(bin_path), x_np)
            cpp_latency_ms = (time.perf_counter() - t0) / args.n_bench * 1000
        print(
            f"   C++ latency (batch=1, n_points={args.n_points}, "
            f"latent={spec.config['latent_dim']}) = {cpp_latency_ms:.3f} ms"
        )

    print("==> Pure-Python latency benchmark")
    for _ in range(3):
        predict_with_spec(spec, x_np)
    t0 = time.perf_counter()
    for _ in range(args.n_bench):
        predict_with_spec(spec, x_np)
    py_latency_ms = (time.perf_counter() - t0) / args.n_bench * 1000
    print(
        f"   Python latency (batch=1, n_points={args.n_points}, "
        f"latent={spec.config['latent_dim']}) = {py_latency_ms:.3f} ms"
    )

    if cpp_latency_ms is not None:
        speedup = py_latency_ms / cpp_latency_ms
        print(f"   C++ speedup vs Python = {speedup:.1f}x")

    print("==> Done.")


if __name__ == "__main__":
    main()
