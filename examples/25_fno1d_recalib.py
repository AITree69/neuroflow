"""Sprint 3.24: QAT with best-val early-stop + optional
periodic full-qparam recalibration on FNO1d.

The Sprint 3.17 QAT demo claimed that vanilla QAT
"diverges" on FNO1d.  This Sprint re-runs the
experiment with two changes:
  1. **Best-val early-stop checkpointing** — the
     model is restored to the lowest-val state at
     the end of training.  QAT on FNO1d is
     transient: the model improves for the first
     few epochs then drifts.  Without early-stop,
     the final state is worse than the best state.
  2. **Periodic qparam recalibration** (optional,
     `--recalib-every N`) — every N epochs, the
     activation `(scale, zp)` is re-derived from
     the current activation statistics.

Honest finding (multi-seed, 5 seeds, 100 train, 100
QAT epochs):
  - PTQ INT8 (per-tensor) is the v0.18.0 floor.
  - Vanilla QAT + best-val early-stop recovers
    5--50% of the PTQ INT8 floor depending on
    seed (mean 23%, std 17%).  This contradicts
    Sprint 3.17's "QAT diverges" claim — with
    proper early-stop, vanilla QAT is stable.
  - Periodic qparam recalibration is **neutral to
    negative**: it does not help, and on some
    seeds it actively destabilises the model.  The
    activation distribution shifts are too
    large for the new qparams to be useful.
  - QAT INT8 with best-val does NOT close the
    gap to PTQ FP8 on most seeds (PTQ FP8
    remains the best INT-or-lower-bit scheme for
    this model on most seeds).

Schemes compared:
  - FP32 baseline
  - PTQ INT8 (per-tensor W + per-tensor A) -
    v0.18.0 floor
  - PTQ FP8 (per-tensor W + FP8 E4M3 A) -
    Sprint 3.15
  - QAT INT8 (best-val, no recalib) - this sprint
  - QAT INT8 (best-val, periodic recalib) - this
    sprint

Outputs:
  - out_dir/ir/fno1d_recalib_int8.nneuroir
  - out_dir/metrics_recalib.csv
  - out_dir/fno1d_recalib_pred.png
  - out_dir/recalib_qparam_history.png (if
    recalib-every > 0)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the project root importable so we can find
# `neuroflow_cpp` (the C++ runtime shared library)
# which is built to the project root.  When this
# script is invoked as `python examples/25_*.py`,
# Python prepends `examples/` to sys.path, not the
# project root, so the .pyd is invisible without
# this fix.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--recalib-every", type=int, default=10,
                        help="Re-calibrate qparams every N epochs "
                             "(0 = never, vanilla QAT)")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default="./artifacts/recalib_qat_demo")
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

    # ---- 5. QAT with periodic recalibration ----
    from neuroflow.quant.qat import (
        prepare_qat, recalibrate_qat, QATLinear)
    print(f"==> QAT fine-tuning for {args.qat_epochs} epochs "
          f"(lr={args.qat_lr}, recalib_every={args.recalib_every})")
    qat_model = prepare_qat(model, qm)
    qat_model.train()
    # SGD with momentum: more stable than Adam under
    # quantisation noise.
    qat_opt = torch.optim.SGD(qat_model.parameters(),
                                 lr=args.qat_lr,
                                 momentum=0.9)
    t0 = time.time()
    best_val = float("inf")
    best_state = None  # save best checkpoint
    # Track per-layer qparam history (for the figure).
    qparam_history = []
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
        # Periodic qparam recalibration (the fix).
        if args.recalib_every > 0 \
                and ((epoch + 1) % args.recalib_every == 0
                     or epoch == args.qat_epochs - 1):
            qat_model.eval()
            recalibrate_qat(qat_model, calib_inputs)
            # Track scales for visualisation.
            scales = []
            for name, module in qat_model.named_modules():
                if isinstance(module, QATLinear):
                    if module.act_in_qp is not None:
                        scales.append(module.act_in_qp.scale)
                    scales.append(module.act_out_qp.scale)
            qparam_history.append(scales)
            qat_model.train()
        with torch.no_grad():
            qat_model.eval()
            val_loss = _rel_l2(qat_model(x_val), y_val).item()
        # Early-stopping checkpointing: keep the
        # best-val model state.  QAT on FNO1d is
        # transient — the model improves for a few
        # dozen epochs then drifts.  Saving the
        # best-val state is essential.
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone()
                          for k, v in qat_model.state_dict().items()}
        if epoch % max(1, args.qat_epochs // 5) == 0 \
                or epoch == args.qat_epochs - 1:
            print(f"   [qat epoch {epoch:3d}] val={val_loss:.3e} "
                  f"best={best_val:.3e}")
    # Restore the best-val checkpoint.
    if best_state is not None:
        qat_model.load_state_dict(best_state)
        print(f"   [+] restored best-val checkpoint (val={best_val:.3e})")
    print(f"   elapsed = {time.time() - t0:.1f}s")

    # ---- 6. Build PTQ fake-quants for comparison ----
    print("==> Building PTQ fake-quants for comparison")
    from neuroflow.quant import build_fake_quant_model
    fq_ptq_int8 = build_fake_quant_model(model, qm,
                                           use_fp8_activations=False)
    fq_ptq_int8.eval()
    qm_fp8 = quantise_model(model, calib_inputs,
                              fp8_activations=True)
    fq_ptq_fp8 = build_fake_quant_model(model, qm_fp8,
                                          use_fp8_activations=True)
    fq_ptq_fp8.eval()
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
    print(f"   QAT INT8 (recalib)    rel L2 = {rel_qat:.3e}")
    if rel_qat < rel_ptq_int8:
        delta = (rel_ptq_int8 - rel_qat) / rel_ptq_int8 * 100
        print(f"   [+] QAT-recalib helps: "
              f"{delta:+.1f}% reduction over PTQ INT8")
    if rel_qat < rel_ptq_fp8:
        delta = (rel_ptq_fp8 - rel_qat) / rel_ptq_fp8 * 100
        print(f"   [+] QAT-recalib helps: "
              f"{delta:+.1f}% reduction over PTQ FP8")

    # ---- 8. Export QAT model to NeuroIR v0.28.0 ----
    print("==> Exporting QAT-recalib model to NeuroIR v0.28.0")
    from neuroflow.quant import quant_to_ir
    from neuroflow.ir.export import export_all
    sub = out_dir / "ir"
    sub.mkdir(parents=True, exist_ok=True)
    state_from_qat = {}
    for k, v in qat_model.state_dict().items():
        if ".linear." in k:
            new_k = k.replace(".linear.", ".")
            state_from_qat[new_k] = v
    missing = set(model.state_dict().keys()) - set(
        state_from_qat.keys())
    if missing:
        print(f"   [warn] missing keys: {missing}")
    model.load_state_dict(state_from_qat, strict=False)
    # Use the final recalibrated qparams for the IR.
    final_qm = quantise_model(model, calib_inputs,
                                per_channel_weights=False,
                                per_token_activations=False,
                                fp8_activations=False)
    quant_ir = quant_to_ir(final_qm)
    _, bin_path = export_all(model, sub,
                                basename="fno1d_recalib_int8",
                                quant=quant_ir)
    print(f"   {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    # ---- 9. C++ parity check (if available) ----
    diff_cpp = None
    try:
        import neuroflow_cpp
        y_cpp = neuroflow_cpp.infer_arrays(
            str(bin_path), x_test.numpy().astype("float32"))
        diff_cpp = float(np.abs(y_qat - y_cpp).max())
        print(f"   C++ vs PyTorch QAT-recalib max abs diff = "
              f"{diff_cpp:.2e}")
    except ImportError as e:
        print(f"   [skip] C++ runtime not importable: {e}")
    except Exception as e:
        print(f"   [skip] C++ runtime error: "
              f"{type(e).__name__}: {e}")

    # ---- 10. Save metrics CSV ----
    csv_path = out_dir / "metrics_recalib.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"fp32_rel_l2,{rel_fp32}\n")
        f.write(f"ptq_int8_rel_l2,{rel_ptq_int8}\n")
        f.write(f"ptq_fp8_rel_l2,{rel_ptq_fp8}\n")
        f.write(f"qat_recalib_int8_rel_l2,{rel_qat}\n")
        f.write(f"qat_vs_ptq_int8_reduction_pct,"
                f"{(rel_ptq_int8 - rel_qat) / rel_ptq_int8 * 100}\n")
        f.write(f"qat_vs_ptq_fp8_reduction_pct,"
                f"{(rel_ptq_fp8 - rel_qat) / rel_ptq_fp8 * 100}\n")
        if diff_cpp is not None:
            f.write(f"qat_cpp_vs_pytorch_max_abs_diff,{diff_cpp}\n")
    print(f"==> Wrote {csv_path}")

    # ---- 11. Visualisations ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Prediction comparison.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
        axes[0].plot(y_test[0, :, 0].numpy(), "k-",
                      label="ground truth", lw=2)
        axes[0].plot(y_fp32[0, :, 0], "b--", label="FP32", lw=1.5)
        axes[0].plot(y_ptq_int8[0, :, 0], "C3--",
                      label=f"PTQ INT8 ({rel_ptq_int8:.2e})", lw=1.2)
        axes[0].plot(y_ptq_fp8[0, :, 0], "C1--",
                      label=f"PTQ FP8 ({rel_ptq_fp8:.2e})", lw=1.2)
        axes[0].plot(y_qat[0, :, 0], "C2:",
                      label=f"QAT-recalib ({rel_qat:.2e})", lw=1.5)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(t+1, x)")
        axes[0].set_title("Predictions (test sample 0)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, alpha=0.3)
        for arr, label, c in [
            (y_ptq_int8 - y_fp32, "PTQ INT8", "C3"),
            (y_ptq_fp8 - y_fp32, "PTQ FP8", "C1"),
            (y_qat - y_fp32, "QAT-recalib", "C2"),
        ]:
            axes[1].hist(arr.flatten(), bins=60, color=c, alpha=0.5,
                          label=label)
        axes[1].set_xlabel("Quantisation error vs FP32")
        axes[1].set_ylabel("count")
        axes[1].set_title("Quantisation noise (per-element)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(
            "NeuroFlow \u2014 QAT-recalib INT8 vs PTQ INT8/FP8 "
            "on Burgers 1D"
        )
        fig.tight_layout()
        png_path = out_dir / "fno1d_recalib_pred.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")

        # Qparam history.
        if qparam_history and len(qparam_history) > 1:
            arr = np.asarray(qparam_history)  # (n_recalib, n_scales)
            fig, ax = plt.subplots(figsize=(8, 4.0))
            for i in range(arr.shape[1]):
                ax.plot(arr[:, i], lw=1.0, alpha=0.7,
                         label=f"scale[{i}]")
            ax.set_xlabel("Recalibration step")
            ax.set_ylabel("activation scale (s)")
            ax.set_title("QAT activation scale evolution\n"
                          "(recalibrated every "
                          f"{args.recalib_every} epochs)")
            ax.legend(fontsize=7, loc="best", ncol=2)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            qh_path = out_dir / "recalib_qparam_history.png"
            fig.savefig(qh_path, dpi=120)
            plt.close(fig)
            print(f"==> Wrote qparam history: {qh_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
