"""Sprint 3.16: Per-channel weight + FP8 activation combo.

Combines the two best schemes (per-channel INT8
weights + FP8 E4M3 activations) to see if they
stack.  This is the "best of both worlds" combo:
- Per-channel weights: per-output-channel INT8
  scale/zp removes the per-tensor weight quant
  floor (Sprint 3.10).
- FP8 E4M3 activations: wider dynamic range than
  INT8 (8 magnitudes per 2^e bucket) closes the
  per-tensor activation floor (Sprint 3.15).

The C++ runtime v0.22.0 supports this combo via
`FNO1d::EnablePerChannelWeightDequant` (per-channel
weights) + `FNO1d::EnableFP8Activation` (FP8
activations).  This example exercises the combo
end-to-end on Burgers 1D FNO1d and produces a
4-column comparison table:

  FP32 | INT8 per-tensor W+A | INT8 PC W + INT8 A | INT8 PC W + FP8 A

Outputs:
  - out_dir/ir/fno1d_pc_fp8.nneuroir — the v0.22.0
    IR with PC weights (NIRQ kind=1) + FP8 act
    (NIRQ kind=3).
  - out_dir/metrics_pc_fp8.csv — full 4-column
    comparison.
  - out_dir/fno1d_pc_fp8_pred.png — predictions +
    noise histograms for the 3 quantised schemes.
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
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--n-calib", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default="./artifacts/pc_fp8_demo")
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
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    x_train, y_train = _to_xy(train_set, args.n_train)
    x_val, y_val = _to_xy(val_set, args.n_val)
    x_test, y_test = _to_xy(test_set, args.n_test)

    # ---- 2. Model: small FNO1d ----
    print("==> Building FNO1d (2-layer, width=32, modes=16)")
    from neuroflow.nn.fno import FNO1d
    model = FNO1d(
        in_channels=1, out_channels=1, width=args.width,
        modes=args.modes, n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   params = {n_params}")

    # ---- 3. Train FP32 baseline ----
    print("==> Training FP32 baseline")
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
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                val_loss = _rel_l2(model(x_val), y_val).item()
            print(f"   [epoch {epoch:3d}] val={val_loss:.3e}")
    print(f"   elapsed = {time.time() - t0:.1f}s")

    # ---- 4. Calibrate ONCE (for all schemes) ----
    print(f"==> Calibrating with {args.n_calib} samples")
    from neuroflow.quant import quantise_model
    calib_inputs = [x_val[i:i+1] for i in range(args.n_calib)]
    # We need three quantised bundles:
    #   (a) PT W + PT A
    #   (b) PC W + PT A
    #   (c) PC W + FP8 A
    # Calibrate (c) once with PC W + FP8 A (the most
    # general case; PT W and PT A are subsets).
    qm_pc_fp8 = quantise_model(model, calib_inputs,
                                  per_channel_weights=True,
                                  per_token_activations=False,
                                  fp8_activations=True)
    print(f"   PC W qparams: {len(qm_pc_fp8.weight_qparams)}")
    print(f"   PT A qparams: {len(qm_pc_fp8.activation_qparams)}")
    print(f"   FP8 A qparams: {len(qm_pc_fp8.fp8_qparams)}")
    # Bundle (a): PT W (per-tensor) + PT A — built by
    # re-running quantise_model with per_channel_weights=False
    # (the v0.15.0 path).
    qm_pt_pt = quantise_model(model, calib_inputs,
                                per_channel_weights=False,
                                per_token_activations=False,
                                fp8_activations=False)
    # Bundle (b): PC W + PT A — same as (c) but skip FP8.
    qm_pc_pt = quantise_model(model, calib_inputs,
                                per_channel_weights=True,
                                per_token_activations=False,
                                fp8_activations=False)

    # ---- 5. Export combo (c) to NeuroIR v0.22.0 ----
    print("==> Exporting NeuroIR v0.22.0 combo (PC W + FP8 A)")
    from neuroflow.quant import quant_to_ir
    from neuroflow.ir.export import export_all
    sub = out_dir / "ir"
    sub.mkdir(parents=True, exist_ok=True)
    quant_ir_pc_fp8 = quant_to_ir(qm_pc_fp8)
    _, bin_path = export_all(model, sub,
                                basename="fno1d_pc_fp8",
                                quant=quant_ir_pc_fp8)
    print(f"   {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    # ---- 6. Build three fake-quant PyTorch references ----
    print("==> Building PyTorch fake-quant references")
    from neuroflow.quant import build_fake_quant_model
    fq_pt_pt = build_fake_quant_model(model, qm_pt_pt,
                                        use_fp8_activations=False)
    fq_pc_pt = build_fake_quant_model(model, qm_pc_pt,
                                        use_fp8_activations=False)
    fq_pc_fp8 = build_fake_quant_model(model, qm_pc_fp8,
                                         use_fp8_activations=True)
    fq_pt_pt.eval()
    fq_pc_pt.eval()
    fq_pc_fp8.eval()

    with torch.no_grad():
        y_fp32 = model(x_test).numpy()
        y_pt_pt = fq_pt_pt(x_test).numpy()
        y_pc_pt = fq_pc_pt(x_test).numpy()
        y_pc_fp8 = fq_pc_fp8(x_test).numpy()
    y_true = y_test.numpy()

    def rel_l2(a, b):
        return float(np.linalg.norm(
            (a - b).reshape(len(a), -1), axis=1
        ).mean() / max(np.linalg.norm(
            b.reshape(len(b), -1), axis=1
        ).mean(), 1e-8))

    rel_fp32 = rel_l2(y_fp32, y_true)
    rel_pt_pt = rel_l2(y_pt_pt, y_true)
    rel_pc_pt = rel_l2(y_pc_pt, y_true)
    rel_pc_fp8 = rel_l2(y_pc_fp8, y_true)

    print(f"   FP32                    rel L2 = {rel_fp32:.3e}")
    print(f"   PT W + PT A (INT8)      rel L2 = {rel_pt_pt:.3e}")
    print(f"   PC W + PT A (INT8)      rel L2 = {rel_pc_pt:.3e}")
    print(f"   PC W + FP8 A            rel L2 = {rel_pc_fp8:.3e}")
    delta_pt_pt = (rel_pt_pt - rel_fp32) / rel_fp32 * 100
    delta_pc_pt = (rel_pc_pt - rel_fp32) / rel_fp32 * 100
    delta_pc_fp8 = (rel_pc_fp8 - rel_fp32) / rel_fp32 * 100
    print(f"   PT W + PT A:  {delta_pt_pt:+.0f}% worse than FP32")
    print(f"   PC W + PT A:  {delta_pc_pt:+.0f}% worse than FP32")
    print(f"   PC W + FP8 A: {delta_pc_fp8:+.0f}% worse than FP32")
    if rel_pc_fp8 < rel_pc_pt:
        print(f"   [+] FP8 activation helps: "
              f"{(rel_pc_pt - rel_pc_fp8) / rel_pc_pt * 100:+.1f}% "
              f"reduction over PC W + PT A")

    # ---- 7. C++ runtime parity (if available) ----
    diff_cpp = None
    try:
        import neuroflow_cpp
        y_cpp = neuroflow_cpp.infer_arrays(
            str(bin_path), x_test.numpy().astype("float32")
        )
        diff_cpp = float(np.abs(y_pc_fp8 - y_cpp).max())
        print(f"   C++ vs PyTorch PC W + FP8 A max abs diff = {diff_cpp:.2e}")
    except Exception as e:
        print(f"   [skip] C++ runtime not available: {e}")

    # ---- 8. Save metrics CSV ----
    csv_path = out_dir / "metrics_pc_fp8.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"fp32_rel_l2,{rel_fp32}\n")
        f.write(f"pt_pt_rel_l2,{rel_pt_pt}\n")
        f.write(f"pc_pt_rel_l2,{rel_pc_pt}\n")
        f.write(f"pc_fp8_rel_l2,{rel_pc_fp8}\n")
        f.write(f"pt_pt_worse_than_fp32_pct,{delta_pt_pt}\n")
        f.write(f"pc_pt_worse_than_fp32_pct,{delta_pc_pt}\n")
        f.write(f"pc_fp8_worse_than_fp32_pct,{delta_pc_fp8}\n")
        if diff_cpp is not None:
            f.write(f"pc_fp8_cpp_vs_pytorch_max_abs_diff,{diff_cpp}\n")
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
        axes[0].plot(y_pt_pt[0, :, 0], "C3--",
                      label=f"PT W+PT A (INT8, {rel_pt_pt:.2e})", lw=1.2)
        axes[0].plot(y_pc_pt[0, :, 0], "C1--",
                      label=f"PC W+PT A (INT8, {rel_pc_pt:.2e})", lw=1.2)
        axes[0].plot(y_pc_fp8[0, :, 0], "C2:",
                      label=f"PC W+FP8 A ({rel_pc_fp8:.2e})", lw=1.5)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(t+1, x)")
        axes[0].set_title("Predictions (test sample 0)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, alpha=0.3)
        # Noise histograms
        for arr, label, c in [
            (y_pt_pt - y_fp32, "PT W+PT A", "C3"),
            (y_pc_pt - y_fp32, "PC W+PT A", "C1"),
            (y_pc_fp8 - y_fp32, "PC W+FP8 A", "C2"),
        ]:
            axes[1].hist(arr.flatten(), bins=60, color=c, alpha=0.5,
                          label=label)
        axes[1].set_xlabel("Quantisation error vs FP32")
        axes[1].set_ylabel("count")
        axes[1].set_title("Quantisation noise (per-element)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(
            "NeuroFlow — Per-channel W + FP8 A combo on Burgers 1D"
        )
        fig.tight_layout()
        png_path = out_dir / "fno1d_pc_fp8_pred.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
