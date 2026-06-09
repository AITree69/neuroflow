"""Sprint 3.17: QAT (Quantisation-Aware Training) demo.

The headline test: does QAT close the ~3.5e-1
residual that PTQ can't close on FNO1d/Burgers 1D?

Method: take a trained FP32 model, calibrate the
INT8 qparams via PTQ, then train WITH fake-quant
ops in the graph (using STE for the gradient) for
some more epochs.  The model adapts its weights to
the quant noise.

Schemes compared:
  - FP32 baseline
  - PTQ INT8 (per-tensor W + per-tensor A) - the
    v0.18.0 floor (6.5e-1)
  - PTQ FP8 (per-tensor W + FP8 E4M3 A) - Sprint
    3.15 (3.4e-1)
  - QAT INT8 (per-tensor W + per-tensor A) - the
    target of this sprint

The QAT-trained model is exported to the same
NeuroIR v0.23.0 NIRQ format as the PTQ model
(no new C++ code needed).

Honest finding (FNO1d + Burgers 1D):
  Vanilla QAT with FIXED qparams (calibrated once
  on the FP32 model) is UNSTABLE on FNO1d.  The
  per-layer activation qparams were calibrated for
  the FP32 weights; as QAT training shifts the
  weights, the qparams become stale (the activation
  distribution drifts), and the model diverges.
  The QAT module + STE infrastructure is correct
  (verified on a single Linear layer in
  `tests/test_quant.py`), but closing the FNO1d
  residual requires either:
    (a) per-layer LR / gradient clipping,
    (b) periodic re-calibration of qparams,
    (c) learned step size (LSQ).
  These are research-grade QAT tricks and are
  queued for future sprints.

Outputs:
  - out_dir/ir/fno1d_qat_int8.nneuroir
  - out_dir/metrics_qat.csv (4-column comparison)
  - out_dir/fno1d_qat_pred.png
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
    parser.add_argument("--ft-epochs", type=int, default=200,
                        help="FP32 pre-training epochs")
    parser.add_argument("--qat-epochs", type=int, default=200,
                        help="QAT fine-tuning epochs")
    parser.add_argument("--qat-lr", type=float, default=1e-4,
                        help="QAT learning rate (smaller than "
                             "FP32 pre-train)")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default="./artifacts/qat_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Data ----
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
    print(f"==> Building FNO1d (2-layer, width=32, modes=16)")
    from neuroflow.nn.fno import FNO1d
    model = FNO1d(
        in_channels=1, out_channels=1, width=args.width,
        modes=args.modes, n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   params = {n_params}")

    # ---- 3. Pre-train FP32 ----
    print(f"==> Pre-training FP32 for {args.ft_epochs} epochs")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    for epoch in range(args.ft_epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            loss = (model(xb) - yb).pow(2).mean()
            loss.backward()
            opt.step()
        if epoch % max(1, args.ft_epochs // 5) == 0 \
                or epoch == args.ft_epochs - 1:
            with torch.no_grad():
                val_loss = _rel_l2(model(x_val), y_val).item()
            print(f"   [epoch {epoch:3d}] val={val_loss:.3e}")
    print(f"   elapsed = {time.time() - t0:.1f}s")

    # ---- 4. Calibrate (PTQ) ----
    print(f"==> Calibrating with {args.n_calib} samples")
    from neuroflow.quant import quantise_model
    calib_inputs = [x_val[i:i+1] for i in range(args.n_calib)]
    qm = quantise_model(model, calib_inputs,
                         per_channel_weights=False,
                         per_token_activations=False,
                         fp8_activations=False)

    # ---- 5. QAT fine-tune ----
    print(f"==> QAT fine-tuning for {args.qat_epochs} epochs "
          f"(lr={args.qat_lr})")
    from neuroflow.quant.qat import prepare_qat
    qat_model = prepare_qat(model, qm)
    qat_model.train()
    # Smaller LR for QAT (common practice).
    # SGD with momentum is more stable than Adam under
    # quantisation noise (the noisy gradients from the
    # fake-quant STE accumulate less aggressively in
    # SGD's running average).
    qat_opt = torch.optim.SGD(qat_model.parameters(),
                                 lr=args.qat_lr,
                                 momentum=0.9)
    t0 = time.time()
    best_val = float("inf")
    for epoch in range(args.qat_epochs):
        qat_model.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            qat_opt.zero_grad()
            yp = qat_model(xb)
            loss = (yp - yb).pow(2).mean()
            loss.backward()
            qat_opt.step()
        with torch.no_grad():
            qat_model.eval()
            val_loss = _rel_l2(qat_model(x_val), y_val).item()
        best_val = min(best_val, val_loss)
        if epoch % max(1, args.qat_epochs // 5) == 0 \
                or epoch == args.qat_epochs - 1:
            print(f"   [qat epoch {epoch:3d}] val={val_loss:.3e} "
                  f"best={best_val:.3e}")
    print(f"   elapsed = {time.time() - t0:.1f}s")

    # ---- 6. Build PTQ fake-quant for comparison ----
    print("==> Building PTQ fake-quant models for comparison")
    from neuroflow.quant import build_fake_quant_model
    fq_ptq_int8 = build_fake_quant_model(model, qm,
                                           use_fp8_activations=False)
    fq_ptq_int8.eval()

    # Also need a FP8 PTQ for comparison
    qm_fp8 = quantise_model(model, calib_inputs,
                              fp8_activations=True)
    fq_ptq_fp8 = build_fake_quant_model(model, qm_fp8,
                                          use_fp8_activations=True)
    fq_ptq_fp8.eval()

    # QAT model in eval mode (still uses fake-quant via the QATLinear)
    qat_model.eval()

    # ---- 7. Evaluate all schemes ----
    with torch.no_grad():
        y_fp32 = model(x_test).numpy()
        y_ptq_int8 = fq_ptq_int8(x_test).numpy()
        y_ptq_fp8 = fq_ptq_fp8(x_test).numpy()
        y_qat = qat_model(x_test).numpy()
    y_true = y_test.numpy()

    def rel_l2(a, b):
        return float(np.linalg.norm(
            (a - b).reshape(len(a), -1), axis=1
        ).mean() / max(np.linalg.norm(
            b.reshape(len(b), -1), axis=1
        ).mean(), 1e-8))

    rel_fp32 = rel_l2(y_fp32, y_true)
    rel_ptq_int8 = rel_l2(y_ptq_int8, y_true)
    rel_ptq_fp8 = rel_l2(y_ptq_fp8, y_true)
    rel_qat = rel_l2(y_qat, y_true)

    print(f"   FP32 baseline         rel L2 = {rel_fp32:.3e}")
    print(f"   PTQ INT8 (per-tensor) rel L2 = {rel_ptq_int8:.3e}")
    print(f"   PTQ FP8  (E4M3 act)   rel L2 = {rel_ptq_fp8:.3e}")
    print(f"   QAT INT8 (per-tensor) rel L2 = {rel_qat:.3e}")
    if rel_qat < rel_ptq_int8:
        delta = (rel_ptq_int8 - rel_qat) / rel_ptq_int8 * 100
        print(f"   [+] QAT helps: {delta:+.1f}% reduction over PTQ INT8")
    if rel_qat < rel_ptq_fp8:
        delta = (rel_ptq_fp8 - rel_qat) / rel_ptq_fp8 * 100
        print(f"   [+] QAT helps: {delta:+.1f}% reduction over PTQ FP8")

    # ---- 8. Export QAT model to NeuroIR v0.23.0 ----
    print("==> Exporting QAT model to NeuroIR v0.23.0")
    from neuroflow.quant import quant_to_ir
    from neuroflow.ir.export import export_all
    # Use the FP32 model's state_dict (the QAT model
    # has the same weights, just wrapped in QATLinear
    # modules with the same names).
    sub = out_dir / "ir"
    sub.mkdir(parents=True, exist_ok=True)
    # Extract the QAT-trained weights back into the
    # original model structure (the qat_model has the
    # same parameter names as model, but nested
    # under .linear — so we copy them out).
    state_from_qat = {}
    for k, v in qat_model.state_dict().items():
        # Map "0.linear.weight" → "0.weight" so the
        # exported model has the same structure.
        if ".linear." in k:
            new_k = k.replace(".linear.", ".")
            state_from_qat[new_k] = v
    # Verify all the original keys are present
    missing = set(model.state_dict().keys()) - set(
        state_from_qat.keys())
    if missing:
        print(f"   [warn] missing keys: {missing}")
    model.load_state_dict(state_from_qat, strict=False)
    quant_ir = quant_to_ir(qm)
    _, bin_path = export_all(model, sub,
                                basename="fno1d_qat_int8",
                                quant=quant_ir)
    print(f"   {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    # ---- 9. C++ parity check (if available) ----
    diff_cpp = None
    try:
        import neuroflow_cpp
        y_cpp = neuroflow_cpp.infer_arrays(
            str(bin_path), x_test.numpy().astype("float32"))
        diff_cpp = float(np.abs(y_qat - y_cpp).max())
        print(f"   C++ vs PyTorch QAT max abs diff = {diff_cpp:.2e}")
    except Exception as e:
        print(f"   [skip] C++ runtime not available: {e}")

    # ---- 10. Save metrics CSV ----
    csv_path = out_dir / "metrics_qat.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"fp32_rel_l2,{rel_fp32}\n")
        f.write(f"ptq_int8_rel_l2,{rel_ptq_int8}\n")
        f.write(f"ptq_fp8_rel_l2,{rel_ptq_fp8}\n")
        f.write(f"qat_int8_rel_l2,{rel_qat}\n")
        if diff_cpp is not None:
            f.write(f"qat_cpp_vs_pytorch_max_abs_diff,{diff_cpp}\n")
    print(f"==> Wrote {csv_path}")

    # ---- 11. Visualisation ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
        axes[0].plot(y_test[0, :, 0].numpy(), "k-",
                      label="ground truth", lw=2)
        axes[0].plot(y_fp32[0, :, 0], "b--", label="FP32", lw=1.5)
        axes[0].plot(y_ptq_int8[0, :, 0], "C3--",
                      label=f"PTQ INT8 ({rel_ptq_int8:.2e})", lw=1.2)
        axes[0].plot(y_ptq_fp8[0, :, 0], "C1--",
                      label=f"PTQ FP8 ({rel_ptq_fp8:.2e})", lw=1.2)
        axes[0].plot(y_qat[0, :, 0], "C2:",
                      label=f"QAT INT8 ({rel_qat:.2e})", lw=1.5)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(t+1, x)")
        axes[0].set_title("Predictions (test sample 0)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, alpha=0.3)
        for arr, label, c in [
            (y_ptq_int8 - y_fp32, "PTQ INT8", "C3"),
            (y_ptq_fp8 - y_fp32, "PTQ FP8", "C1"),
            (y_qat - y_fp32, "QAT INT8", "C2"),
        ]:
            axes[1].hist(arr.flatten(), bins=60, color=c, alpha=0.5,
                          label=label)
        axes[1].set_xlabel("Quantisation error vs FP32")
        axes[1].set_ylabel("count")
        axes[1].set_title("Quantisation noise (per-element)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(
            "NeuroFlow — QAT INT8 vs PTQ INT8/FP8 on Burgers 1D"
        )
        fig.tight_layout()
        png_path = out_dir / "fno1d_qat_pred.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
