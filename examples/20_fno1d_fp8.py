"""Sprint 3.15: FP8 (E4M3) activation quantisation demo.

Compares the FP8 W8A8 fake-quant path against the
INT8 W8A8 path on Burgers 1D FNO1d.  The headline
question: does FP8 close the per-token INT8 floor
on activations?

The C++ runtime v0.21.0 implements this via
`FNO1d::EnableFP8Activation` (post-`Linear` FP8
fake-quant, FP8 E4M3 symmetric, scale = max_abs /
448).  This Python script reproduces the same
round-trip in `FakeQuantLinear` via
`build_fake_quant_model(use_fp8_activations=True)`.

Outputs:
  - out_dir/ir/fno1d_int8_fp8.nneuroir — the
    exported NeuroIR v0.21.0 model with FP8
    activation qparams (NIRQ kind=3 entries).
  - out_dir/metrics_<scheme>.csv — relative L2 vs
    ground truth for FP32, INT8, FP8.
  - out_dir/fno1d_int8_fp8_pred.png — predictions
    vs ground truth + quantisation noise histogram.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (torch.linalg.norm(a - b) /
            torch.linalg.norm(b).clamp(min=1e-8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--n-calib", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default="./artifacts/fp8_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Data: 1D Burgers 1-step regression ----
    print("==> Building Burgers 1D 1-step dataset")
    from neuroflow.data.burgers import Burgers1dConfig, Burgers1dDataset
    cfg = Burgers1dConfig(
        n_points=args.n_points, n_tsteps=20, nu=0.01, dt=0.01,
    )
    train_set = Burgers1dDataset(
        n_samples=args.n_train, cfg=cfg, t_in=1, t_out=1,
        seed=args.seed, trajectory_stride=1,
    )
    val_set = Burgers1dDataset(
        n_samples=args.n_val, cfg=cfg, t_in=1, t_out=1,
        seed=args.seed + 1, trajectory_stride=1,
    )
    test_set = Burgers1dDataset(
        n_samples=args.n_test, cfg=cfg, t_in=1, t_out=1,
        seed=args.seed + 2, trajectory_stride=1,
    )

    def _to_xy(ds, n: int):
        xs, ys = [], []
        for i in range(n):
            x_i, y_i = ds[i]
            xs.append(x_i)
            ys.append(y_i)
        x = torch.stack(xs, dim=0)  # (B, n, 1)
        y = torch.stack(ys, dim=0)  # (B, n, 1)
        return x, y

    x_train, y_train = _to_xy(train_set, args.n_train)
    x_val, y_val = _to_xy(val_set, args.n_val)
    x_test, y_test = _to_xy(test_set, args.n_test)
    print(f"   train = {len(x_train)}, val = {len(x_val)}, "
          f"test = {len(x_test)}, n_points = {args.n_points}")

    # ---- 2. Model: small FNO1d ----
    print("==> Building FNO1d (2-layer, width=32, modes=16)")
    from neuroflow.nn.fno import FNO1d
    model = FNO1d(
        in_channels=1, out_channels=1, width=args.width,
        modes=args.modes, n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   params = {n_params}")

    # ---- 3. Train ----
    print("==> Training FP32 baseline")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        train_loss, train_count = 0.0, 0
        for i in range(0, len(x_train), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            yp = model(xb)
            loss = (yp - yb).pow(2).mean()
            loss.backward()
            opt.step()
            train_loss += float(loss) * len(idx)
            train_count += len(idx)
        train_loss /= max(train_count, 1)
        with torch.no_grad():
            val_loss = _rel_l2(model(x_val), y_val).item()
        best_val = min(best_val, val_loss)
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(f"   [epoch {epoch:3d}] train={train_loss:.3e} "
                  f"val={val_loss:.3e} best={best_val:.3e}")
    print(f"   elapsed = {time.time() - t0:.1f}s")

    # ---- 4. Calibrate (INT8 + FP8 together) ----
    print(f"==> Calibrating with {args.n_calib} samples (INT8 + FP8)")
    from neuroflow.quant import quantise_model
    calib_inputs = [x_val[i:i+1] for i in range(args.n_calib)]
    qm = quantise_model(model, calib_inputs,
                         per_channel_weights=False,
                         per_token_activations=False,
                         percentile=100.0, ema_decay=None,
                         fp8_activations=True)
    print(f"   INT8 act qparams: {len(qm.activation_qparams)} layers")
    print(f"   FP8  act qparams: {len(qm.fp8_qparams)} layers")
    # FP8 scales for the first and last linear layer
    for k in sorted(qm.fp8_qparams.keys())[:3]:
        print(f"     {k}: scale = {qm.fp8_qparams[k].scale:.3e}")
    for k in sorted(qm.fp8_qparams.keys())[-2:]:
        print(f"     {k}: scale = {qm.fp8_qparams[k].scale:.3e}")

    # ---- 5. Export to NeuroIR v0.21.0 ----
    print("==> Exporting NeuroIR v0.21.0 with NIRQ kind=3 (FP8) entries")
    from neuroflow.quant import quant_to_ir
    from neuroflow.ir.export import export_all
    sub = out_dir / "ir"
    sub.mkdir(parents=True, exist_ok=True)
    quant_ir = quant_to_ir(qm)
    _, bin_path = export_all(model, sub,
                                basename="fno1d_int8_fp8",
                                quant=quant_ir)
    print(f"   {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    # ---- 6. Three predictions: FP32 / INT8 (fake-quant) / FP8 (fake-quant) ----
    print("==> Building PyTorch fake-quant references")
    from neuroflow.quant import build_fake_quant_model
    fq_int8 = build_fake_quant_model(model, qm,
                                       use_fp8_activations=False)
    fq_fp8 = build_fake_quant_model(model, qm,
                                      use_fp8_activations=True)
    fq_int8.eval()
    fq_fp8.eval()
    with torch.no_grad():
        y_fp32 = model(x_test).numpy()
        y_int8 = fq_int8(x_test).numpy()
        y_fp8 = fq_fp8(x_test).numpy()
    y_true = y_test.numpy()

    def rel_l2(a, b):
        return float(np.linalg.norm(
            (a - b).reshape(len(a), -1), axis=1
        ).mean() / max(np.linalg.norm(
            b.reshape(len(b), -1), axis=1
        ).mean(), 1e-8))

    rel_fp32 = rel_l2(y_fp32, y_true)
    rel_int8 = rel_l2(y_int8, y_true)
    rel_fp8 = rel_l2(y_fp8, y_true)
    err_int8 = float(np.abs(y_fp32 - y_int8).max())
    err_fp8 = float(np.abs(y_fp32 - y_fp8).max())
    print(f"   FP32 rel L2 = {rel_fp32:.3e}")
    print(f"   INT8 rel L2 = {rel_int8:.3e}  (max abs err = {err_int8:.3e})")
    print(f"   FP8  rel L2 = {rel_fp8:.3e}  (max abs err = {err_fp8:.3e})")
    delta_vs_int8 = (rel_fp8 - rel_int8) / rel_int8 * 100
    print(f"   FP8 vs INT8: {delta_vs_int8:+.1f}% change in rel L2")

    # ---- 7. C++ runtime parity (if available) ----
    diff_cpp_fp8: float | None = None
    diff_cpp_int8: float | None = None
    try:
        import neuroflow_cpp
        y_cpp_int8 = neuroflow_cpp.infer_arrays(
            str(bin_path), x_test.numpy().astype("float32")
        )
        diff_cpp_int8 = float(np.abs(y_int8 - y_cpp_int8).max())
        diff_cpp_fp8 = float(np.abs(y_fp8 - y_cpp_int8).max())
        print(f"   C++ vs PyTorch INT8 max abs diff = {diff_cpp_int8:.2e}")
        print(f"   C++ vs PyTorch FP8  max abs diff = {diff_cpp_fp8:.2e}")
    except Exception as e:
        print(f"   [skip] C++ runtime not available: {e}")

    # ---- 8. Save metrics CSV ----
    csv_path = out_dir / "metrics_fp8_vs_int8.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"fp32_rel_l2,{rel_fp32}\n")
        f.write(f"int8_rel_l2,{rel_int8}\n")
        f.write(f"fp8_rel_l2,{rel_fp8}\n")
        f.write(f"int8_max_abs_err_vs_fp32,{err_int8}\n")
        f.write(f"fp8_max_abs_err_vs_fp32,{err_fp8}\n")
        f.write(f"fp8_vs_int8_rel_l2_change_pct,{delta_vs_int8}\n")
        if diff_cpp_int8 is not None:
            f.write(f"int8_cpp_vs_pytorch_max_abs_diff,{diff_cpp_int8}\n")
            f.write(f"fp8_cpp_vs_pytorch_max_abs_diff,{diff_cpp_fp8}\n")
    print(f"==> Wrote {csv_path}")

    # ---- 9. Visualisation ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
        axes[0].plot(y_test[0, :, 0].numpy(), "k-",
                      label="ground truth", lw=2)
        axes[0].plot(y_fp32[0, :, 0], "b--", label="FP32", lw=1.5)
        axes[0].plot(y_int8[0, :, 0], "r--",
                      label="INT8 (W8A8 fake-quant)", lw=1.5)
        axes[0].plot(y_fp8[0, :, 0], "g:",
                      label="FP8 (W8A8 fake-quant, E4M3 act)", lw=1.5)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(t+1, x)")
        axes[0].set_title("Predictions (test sample 0)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Quantisation error histogram
        err_i = (y_int8 - y_fp32).flatten()
        err_f = (y_fp8 - y_fp32).flatten()
        axes[1].hist(err_i, bins=60, color="C3", alpha=0.6,
                      label=f"INT8 err (max abs = {err_int8:.2e})")
        axes[1].hist(err_f, bins=60, color="C2", alpha=0.6,
                      label=f"FP8 err (max abs = {err_fp8:.2e})")
        axes[1].set_xlabel("Quantisation error vs FP32")
        axes[1].set_ylabel("count")
        axes[1].set_title(
            f"FP8 vs INT8 (FP8 change: {delta_vs_int8:+.1f}% rel L2)"
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(
            "NeuroFlow — FP8 (E4M3) W8A8 vs INT8 W8A8 on Burgers 1D"
        )
        fig.tight_layout()
        png_path = out_dir / "fno1d_int8_fp8_pred.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
