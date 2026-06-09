"""Sprint 3.28 -- Inline INT8 GEMM production-kernel bench.

Compares the FNO1d::Forward wall-time with the inline INT8
GEMM path (per-channel W + per-tensor A) vs the FP32-only
path.  This is the production-kernel counterpart to the
Sprint 3.26 bench (which only timed the standalone INT8
GEMM kernel).  Here the per-layer Linear is the hot loop;
with the inline path it goes through
`int8_gemm::LinearForward` (VNNI / AVX2 / scalar), without
it goes through the FP32 Eigen path.

Usage:
    python examples/27_fno1d_int8_gemm_inline_bench.py
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from neuroflow.ir.export import export_all
from neuroflow.nn.fno import FNO1d, FNO1dConfig
from neuroflow.quant import quantise_model, quant_to_ir


def _time_run(rt, x, y, n_iters: int) -> float:
    """Return mean time per call in microseconds."""
    # Warm up.
    for _ in range(3):
        rt.run(x, y)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        rt.run(x, y)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1e6  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n", type=int, default=256,
                    help="input length (must be power of 2)")
    ap.add_argument("--n_iters", type=int, default=200)
    ap.add_argument("--per_channel", action="store_true", default=True,
                    help="per-channel W + per-tensor A (the INT8 GEMM "
                         "path's natural fit; default on)")
    ap.add_argument("--out_dir", type=str,
                    default="artifacts/bench_fno1d_int8_gemm_inline")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build + calibrate a small FNO1d.  Width / modes / n_layers
    # are user-controlled so you can crank width to see the
    # speedup scale.
    torch.manual_seed(0)
    cfg = FNO1dConfig(
        in_channels=1, out_channels=1,
        width=args.width, modes=args.modes,
        n_layers=args.n_layers, activation="gelu",
    )
    model = FNO1d(cfg).eval()
    with torch.no_grad():
        x_calib = torch.randn(8, args.n, 1) * 0.5
    quantised = quantise_model(
        model, [x_calib],
        per_channel_weights=args.per_channel,
        per_token_activations=False,
    )
    quant_ir = quant_to_ir(quantised)

    # Export two IRs: a FP32 reference (no qparams) and a
    # calibrated IR (per-channel W + per-tensor A).
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _, fp32_path = export_all(model, tmpdir,
                                  basename="fno1d_fp32")
        _, int8_path = export_all(model, tmpdir,
                                  basename="fno1d_int8",
                                  quant=quant_ir)
        fp32_ir = out_dir / fp32_path.name
        int8_ir = out_dir / int8_path.name
        import shutil
        shutil.copy2(fp32_path, fp32_ir)
        shutil.copy2(int8_path, int8_ir)

    import neuroflow_cpp

    # Input
    x_np = np.random.default_rng(0).uniform(
        -0.5, 0.5, size=(1, args.n, 1)).astype(np.float32)
    y_shape = (1, args.n, 1)
    y_fp32 = np.zeros(y_shape, dtype=np.float32)
    y_int8 = np.zeros(y_shape, dtype=np.float32)

    # FP32 path
    rt_fp32 = neuroflow_cpp.InferenceRuntime(str(fp32_ir))
    t_fp32 = _time_run(rt_fp32, x_np, y_fp32, args.n_iters)

    # INT8 GEMM path
    rt_int8 = neuroflow_cpp.InferenceRuntime(str(int8_ir))
    rt_int8.enable_int8_gemm()
    assert rt_int8.is_int8_gemm_enabled(), (
        "INT8 GEMM didn't take effect -- per-channel W "
        "qparams may be missing from the IR"
    )
    t_int8 = _time_run(rt_int8, x_np, y_int8, args.n_iters)

    speedup = t_fp32 / t_int8
    max_abs = float(np.abs(y_fp32 - y_int8).max())

    print(f"FNO1d inline INT8 GEMM bench "
          f"(w={args.width}, modes={args.modes}, "
          f"L={args.n_layers}, n={args.n}, "
          f"n_iters={args.n_iters})")
    print(f"  FP32 (Eigen Linear):    {t_fp32:8.2f} us / call")
    print(f"  INT8 GEMM (inline):     {t_int8:8.2f} us / call")
    print(f"  Speedup:                {speedup:.2f}x")
    print(f"  Max abs diff (INT8 vs FP32): {max_abs:.3e}")
    print()
    print("Note: the per-layer Linear is one of several ops in")
    print("FNO1d::Forward (also FFT, bias-add, activation,")
    print("permute).  The total speedup on the full pipeline is")
    print("bounded by Amdahl's law: if the Linear is X% of the")
    print("forward time, the full speedup is at most 1/(1-X+X/S)")
    print("where S is the kernel speedup (~5-6x for VNNI/AVX2).")
    print("Crank --width to push X higher and see the speedup")
    print("approach the kernel limit.")

    out_txt = out_dir / f"fno1d_int8_gemm_inline_w{args.width}_L{args.n_layers}.txt"
    with open(out_txt, "w") as f:
        f.write(
            f"FNO1d inline INT8 GEMM bench "
            f"(w={args.width}, modes={args.modes}, "
            f"L={args.n_layers}, n={args.n}, "
            f"n_iters={args.n_iters})\n"
            f"  FP32 (Eigen Linear):    {t_fp32:8.2f} us / call\n"
            f"  INT8 GEMM (inline):     {t_int8:8.2f} us / call\n"
            f"  Speedup:                {speedup:.2f}x\n"
            f"  Max abs diff (INT8 vs FP32): {max_abs:.3e}\n"
        )
    print(f"\nResults saved to {out_txt}")


if __name__ == "__main__":
    main()
