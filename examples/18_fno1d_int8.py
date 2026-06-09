"""Stage 2 Sprint 3.9 demo — INT8 (W8A8 fake-quant) post-
training quantisation for an FNO1d trained on Burgers 1D.

Pipeline:
  1. Train an FNO1d on the 1D Burgers dataset.
  2. Calibrate + quantise the trained model with a
     small held-out calibration set (post-training
     quantisation, no fine-tuning).
  3. Export to NeuroIR v0.15.0 with the trailing NIRQ
     block carrying per-tensor scale / zero_point.
  4. Compare three end-to-end paths on the same test
     inputs:
       a. FP32 reference (PyTorch, no quantisation)
       b. INT8 fake-quant reference (PyTorch, the
          `build_fake_quant_model` clone with the same
          fake-quant round-trip the C++ runtime does)
       c. INT8 fake-quant C++ (the v0.15.0 runtime with
          the quant block loaded)
     and report the per-path accuracy + the
     PyTorch-INT8 vs C++-INT8 parity.

Run:
    python examples/18_fno1d_int8.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from neuroflow.ir.export import export_all
from neuroflow.nn.fno import FNO1d, FNO1dConfig
from neuroflow.quant import (
    build_fake_quant_model,
    quant_to_ir,
    quantise_model,
)


def _build_fno1d(n_points: int = 64) -> FNO1d:
    cfg = FNO1dConfig(
        in_channels=1, out_channels=1,
        width=64, modes=min(16, n_points // 2),
        n_layers=4, activation="gelu", pad_factor=1,
        name="fno1d_int8_demo",
    )
    return FNO1d(cfg)


def _rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).reshape(pred.size(0), -1)
    ref = target.reshape(target.size(0), -1)
    return torch.mean(
        torch.linalg.norm(diff, dim=1)
        / torch.linalg.norm(ref, dim=1).clamp_min(1e-8)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="INT8 PTQ for FNO1d on Burgers 1D"
    )
    parser.add_argument("--n-train", type=int, default=80)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--n-calib", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=4)
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--per-channel", action="store_true",
                         help="Use per-output-channel INT8 "
                              "weight quantisation (closes "
                              "the per-tensor weight floor)")
    parser.add_argument("--per-token", action="store_true",
                         help="Use per-spatial-point INT8 "
                              "activation quantisation (closes "
                              "the per-tensor activation floor; "
                              "this is the main accuracy win)")
    parser.add_argument("--percentile", type=float, default=100.0,
                         help="Percentile for activation range "
                              "(e.g. 99.5 to be robust to outliers; "
                              "100.0 = strict min/max, the v0.17 "
                              "behaviour)")
    parser.add_argument("--ema-decay", type=float, default=None,
                         help="EMA decay for the per-sample min/max "
                              "across the calibration batch (e.g. 0.9; "
                              "None = no EMA, use global min/max)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                         default="./artifacts/quant_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Data: 1D Burgers 1-step regression ----
    print("==> Building Burgers 1D 1-step dataset (t_in = t_out = 1)")
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
    print(f"   train = {len(train_set)}, val = {len(val_set)}, "
          f"test = {len(test_set)}, n_points = {args.n_points}")

    # The dataset already returns (n, t_in) and (n, t_out);
    # for t_in = t_out = 1 that is (n, 1) — exactly the
    # (batch, n, in_channels) shape FNO1d expects once we
    # batch them.
    def _to_xy(ds, n: int):
        xs, ys = [], []
        for i in range(n):
            x_i, y_i = ds[i]
            xs.append(x_i)  # (n, 1)
            ys.append(y_i)
        return torch.stack(xs).float(), torch.stack(ys).float()

    x_train, y_train = _to_xy(train_set, len(train_set))
    x_val, y_val = _to_xy(val_set, len(val_set))
    x_test, y_test = _to_xy(test_set, len(test_set))

    # ---- 2. Train the FP32 model ----
    print("==> Training FNO1d (FP32)")
    model = _build_fno1d(n_points=args.n_points)
    print(f"   parameters = {model.num_parameters():,}")
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    optim = Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_count = 0.0, 0
        for xb, yb in train_loader:
            optim.zero_grad()
            pred = model(xb)
            loss = _rel_l2(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * xb.size(0)
            train_count += xb.size(0)
        train_loss /= max(train_count, 1)
        with torch.no_grad():
            val_loss = _rel_l2(model(x_val), y_val).item()
        best_val = min(best_val, val_loss)
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(f"   [epoch {epoch:3d}] train={train_loss:.3e} "
                  f"val={val_loss:.3e} best={best_val:.3e}")
    print(f"   elapsed = {time.time() - t0:.1f}s")

    # ---- 3. Calibrate + quantise ----
    if args.per_token and args.per_channel:
        scheme = "per-channel + per-token"
    elif args.per_token:
        scheme = "per-tensor W + per-token A"
    elif args.per_channel:
        scheme = "per-channel"
    else:
        scheme = "per-tensor"
    if args.percentile < 100.0 or args.ema_decay is not None:
        scheme += f" (calib: p{args.percentile}, ema={args.ema_decay})"
    print(f"==> Calibrating with {args.n_calib} samples ({scheme})")
    calib_inputs = [x_val[i:i+1] for i in range(args.n_calib)]
    qm = quantise_model(model, calib_inputs,
                         per_channel_weights=args.per_channel,
                         per_token_activations=args.per_token,
                         percentile=args.percentile,
                         ema_decay=args.ema_decay)
    print(f"   weight_qparams:    {len(qm.weight_qparams)} tensors")
    print(f"   activation_qparams: {len(qm.activation_qparams)} tensors")
    n_weights_int8 = sum(arr.nbytes for arr in qm.int8_weights.values())
    n_weights_fp32 = sum(t.numel() * 4 for t in model.state_dict().values())
    print(f"   weight storage:  FP32 = {n_weights_fp32 / 1024:.1f} KB, "
          f"INT8 = {n_weights_int8 / 1024:.1f} KB "
          f"({n_weights_int8 / n_weights_fp32 * 100:.1f}%)")

    # ---- 4. Export to NeuroIR v0.18.0 with NIRQ block ----
    print(f"==> Exporting NeuroIR v0.18.0 with NIRQ block ({scheme})")
    sub = out_dir / "ir"
    sub.mkdir(parents=True, exist_ok=True)
    quant_ir = quant_to_ir(qm)
    if args.per_token and args.per_channel:
        basename = "fno1d_int8_pc_pt"
    elif args.per_token:
        basename = "fno1d_int8_pt"
    elif args.per_channel:
        basename = "fno1d_int8_pc"
    else:
        basename = "fno1d_int8"
    _, bin_path = export_all(model, sub, basename=basename,
                                quant=quant_ir)
    print(f"   {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    # ---- 5. PyTorch FP32 vs INT8 (fake-quant) baseline ----
    print("==> Building PyTorch INT8 fake-quant reference")
    fq_model = build_fake_quant_model(model, qm)
    fq_model.eval()
    with torch.no_grad():
        y_fp32 = model(x_test).numpy()
        y_int8_pt = fq_model(x_test).numpy()
    rel_fp32 = float(np.linalg.norm(
        (y_fp32 - y_test.numpy()).reshape(args.n_test, -1), axis=1
    ).mean() / max(np.linalg.norm(
        y_test.numpy().reshape(args.n_test, -1), axis=1
    ).mean(), 1e-8))
    rel_int8_pt = float(np.linalg.norm(
        (y_int8_pt - y_test.numpy()).reshape(args.n_test, -1), axis=1
    ).mean() / max(np.linalg.norm(
        y_test.numpy().reshape(args.n_test, -1), axis=1
    ).mean(), 1e-8))
    # INT8 quantisation error (FP32 model vs INT8 model).
    err_int8 = float(np.abs(y_fp32 - y_int8_pt).max())
    print(f"   FP32 model rel L2 (vs ground truth) = {rel_fp32:.3e}")
    print(f"   INT8 model rel L2 (vs ground truth) = {rel_int8_pt:.3e}")
    print(f"   INT8 quantisation error max abs (FP32 vs INT8) = {err_int8:.3e}")
    try:
        import neuroflow_cpp
        y_int8_cpp = neuroflow_cpp.infer_arrays(
            str(bin_path), x_test.numpy().astype("float32")
        )
        diff_cpp = float(np.abs(y_int8_pt - y_int8_cpp).max())
        print(f"   C++ vs PyTorch INT8 max abs diff = {diff_cpp:.2e}")
    except Exception as e:
        print(f"   [skip] C++ runtime not available: {e}")

    # ---- 7. Save metrics CSV ----
    scheme_slug = scheme.replace(" ", "_").replace("(", "").replace(
        ")", "").replace(",", "").replace(":", "_").replace("=", "_")
    csv_path = out_dir / f"metrics_{scheme_slug}.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"scheme,{scheme}\n")
        f.write(f"fp32_rel_l2,{rel_fp32}\n")
        f.write(f"int8_rel_l2_pytorch,{rel_int8_pt}\n")
        f.write(f"int8_quant_err_max_abs,{err_int8}\n")
        if diff_cpp is not None:
            f.write(f"int8_cpp_vs_pytorch_max_abs_diff,{diff_cpp}\n")
    print(f"==> Wrote {csv_path}")

    # ---- 8. Visualisation: predictions vs ground truth ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        axes[0].plot(y_test[0, :, 0].numpy(), "k-",
                      label="ground truth", lw=2)
        axes[0].plot(y_fp32[0, :, 0], "b--", label="FP32", lw=1.5)
        axes[0].plot(y_int8_pt[0, :, 0], "r--",
                      label="INT8 (fake-quant, PyTorch)", lw=1.5)
        if diff_cpp is not None:
            axes[0].plot(y_int8_cpp[0, :, 0], "g:",
                          label="INT8 (fake-quant, C++)", lw=1.5)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(t+1, x)")
        axes[0].set_title("Predictions")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        err = (y_int8_pt - y_fp32).flatten()
        axes[1].hist(err, bins=60, color="C0", alpha=0.7,
                      label="INT8 vs FP32 (per-element error)")
        axes[1].set_xlabel("INT8 quantisation error")
        axes[1].set_ylabel("count")
        axes[1].set_title(
            f"Quantisation noise ({scheme} weights, "
            f"max abs = {err_int8:.2e})"
        )
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(
            f"NeuroFlow — INT8 ({scheme} W8A8 fake-quant) "
            f"PTQ on Burgers 1D"
        )
        fig.tight_layout()
        png_path = out_dir / f"fno1d_int8_pred_{scheme_slug}.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"==> Wrote figure: {png_path}")
    except Exception as e:
        print(f"   [skip] figure generation failed: {e}")

    print("==> Done.")


if __name__ == "__main__":
    main()
