"""Sprint 3.19: FP8 generalisation to FNO2D.

Validates the Sprint 3.15 finding on a 2D operator.
If FP8 also reduces the INT8 W8A8 floor on FNO2D,
it confirms FP8 is a general technique for
spectral neural operators, not just a 1D
phenomenon.

Setup:
  - Train a small FNO2d on 2D heat / Poisson data
    (existing Sprint 3.5 dataset).
  - Calibrate INT8 + FP8 qparams via PTQ.
  - Compare FP32 / PTQ INT8 / PTQ FP8 on the
    test set.

The C++ FNO2d path supports FP8 activation
fake-quant in v0.25.0 (this sprint adds the FP8
fake-quant at the locs.<i> Linear output, mirroring
the FNO1d pattern).

Outputs:
  - out_dir/ir/fno2d_fp8.nneuroir (FP32 / FP8 / INT8)
  - out_dir/metrics_fno2d_fp8.csv (3-scheme)
  - out_dir/fno2d_fp8_pred.png
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
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=80)
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--n-calib", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--modes-h", type=int, default=8)
    parser.add_argument("--modes-w", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default="./artifacts/fno2d_fp8_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Data: 2D heat / Poisson ----
    print("==> Building Heat2d dataset (2D Poisson)")
    from neuroflow.data.heat2d import Heat2dConfig, Heat2dDataset
    cfg = Heat2dConfig(h=args.H, w=args.W, seed=args.seed)
    train_set = Heat2dDataset(n_samples=args.n_train, cfg=cfg)
    val_set = Heat2dDataset(n_samples=args.n_val, cfg=cfg)
    test_set = Heat2dDataset(n_samples=args.n_test, cfg=cfg)

    def _to_xy(ds, n: int):
        xs, ys = [], []
        for i in range(n):
            x_i, y_i = ds[i]
            xs.append(x_i)
            ys.append(y_i)
        x = torch.stack(xs, dim=0)  # (B, H, W, in_ch)
        y = torch.stack(ys, dim=0)  # (B, H, W, out_ch)
        return x, y

    x_train, y_train = _to_xy(train_set, args.n_train)
    x_val, y_val = _to_xy(val_set, args.n_val)
    x_test, y_test = _to_xy(test_set, args.n_test)
    print(f"   train = {len(x_train)}, val = {len(x_val)}, "
          f"test = {len(x_test)}, H={args.H}, W={args.W}")

    # ---- 2. Model: small FNO2d ----
    print("==> Building FNO2d (2-layer, width=16, modes=8)")
    from neuroflow.nn.fno2d import FNO2d, FNO2dConfig
    fno2d_cfg = FNO2dConfig(
        in_channels=1, out_channels=1, width=args.width,
        modes_h=args.modes_h, modes_w=args.modes_w,
        n_layers=args.n_layers, pad_factor=2,
        activation="gelu",
    )
    model = FNO2d(fno2d_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   params = {n_params}")

    # ---- 3. Train FP32 baseline ----
    print(f"==> Training FP32 for {args.epochs} epochs")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            loss = (model(xb) - yb).pow(2).mean()
            loss.backward()
            opt.step()
        if epoch % max(1, args.epochs // 5) == 0 \
                or epoch == args.epochs - 1:
            with torch.no_grad():
                val_loss = _rel_l2(model(x_val), y_val).item()
            print(f"   [epoch {epoch:3d}] val={val_loss:.3e}")
    print(f"   elapsed = {time.time() - t0:.1f}s")

    # ---- 4. Calibrate ----
    print(f"==> Calibrating with {args.n_calib} samples")
    from neuroflow.quant import quantise_model
    calib_inputs = [x_val[i:i + 1] for i in range(args.n_calib)]
    qm_int8 = quantise_model(model, calib_inputs,
                                fp8_activations=False)
    qm_fp8 = quantise_model(model, calib_inputs,
                              fp8_activations=True)

    # ---- 5. Export FP8 model to NeuroIR v0.25.0 ----
    print("==> Exporting NeuroIR v0.25.0 (FP8 activations)")
    from neuroflow.quant import quant_to_ir
    from neuroflow.ir.export import export_all
    sub = out_dir / "ir"
    sub.mkdir(parents=True, exist_ok=True)
    quant_ir = quant_to_ir(qm_fp8)
    _, bin_path = export_all(model, sub,
                                basename="fno2d_fp8",
                                quant=quant_ir)
    print(f"   {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    # ---- 6. Build fake-quant references ----
    print("==> Building PyTorch fake-quant references")
    from neuroflow.quant import build_fake_quant_model
    fq_int8 = build_fake_quant_model(model, qm_int8,
                                       use_fp8_activations=False)
    fq_fp8 = build_fake_quant_model(model, qm_fp8,
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
    print(f"   FP32         rel L2 = {rel_fp32:.3e}")
    print(f"   PTQ INT8     rel L2 = {rel_int8:.3e}")
    print(f"   PTQ FP8      rel L2 = {rel_fp8:.3e}")
    if rel_fp8 < rel_int8:
        delta = (rel_int8 - rel_fp8) / rel_int8 * 100
        print(f"   [+] FP8 helps: {delta:+.1f}% reduction over INT8 on FNO2d")

    # ---- 7. C++ runtime parity check ----
    diff_cpp = None
    try:
        import neuroflow_cpp
        y_cpp = neuroflow_cpp.infer_arrays(
            str(bin_path), x_test.numpy().astype("float32"))
        diff_cpp = float(np.abs(y_fp8 - y_cpp).max())
        print(f"   C++ vs PyTorch FP8 max abs diff = {diff_cpp:.2e}")
    except Exception as e:
        print(f"   [skip] C++ runtime not available: {e}")

    # ---- 8. Save metrics CSV ----
    csv_path = out_dir / "metrics_fno2d_fp8.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"fp32_rel_l2,{rel_fp32}\n")
        f.write(f"int8_rel_l2,{rel_int8}\n")
        f.write(f"fp8_rel_l2,{rel_fp8}\n")
        f.write(f"fp8_vs_int8_reduction_pct,"
                f"{(rel_int8 - rel_fp8) / rel_int8 * 100 if rel_int8 > 0 else 0}\n")
        if diff_cpp is not None:
            f.write(f"fp8_cpp_vs_pytorch_max_abs_diff,{diff_cpp}\n")
    print(f"==> Wrote {csv_path}")

    # ---- 9. Visualisation ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        # 2D heatmap of one test sample
        im = axes[0].imshow(y_test[0, :, :, 0].numpy(),
                              cmap="viridis")
        axes[0].set_title("Ground truth (test 0)")
        plt.colorbar(im, ax=axes[0])
        # Quantisation error
        err_int8 = (y_int8 - y_fp32)[0, :, :, 0]
        err_fp8 = (y_fp8 - y_fp32)[0, :, :, 0]
        vmax = max(np.abs(err_int8).max(), np.abs(err_fp8).max())
        im1 = axes[1].imshow(err_int8, cmap="RdBu", vmin=-vmax, vmax=vmax)
        axes[1].set_title(
            f"INT8 err (max abs = {np.abs(err_int8).max():.2e})")
        plt.colorbar(im1, ax=axes[1])
        fig.suptitle(
            f"NeuroFlow — FP8 (E4M3) W8A8 on FNO2D / 2D Poisson"
        )
        fig.tight_layout()
        png_path = out_dir / "fno2d_fp8_pred.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
