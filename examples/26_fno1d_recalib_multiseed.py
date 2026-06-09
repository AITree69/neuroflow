"""Sprint 3.24 multi-seed test: vanilla QAT (best-val)
vs QAT (best-val + periodic recalib) vs PTQ INT8
vs PTQ FP8 on FNO1d, 5 seeds."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the project root importable so we can find
# `neuroflow_cpp` (the C++ runtime shared library).
# See examples/25_fno1d_recalib.py for context.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _run_seed(seed: int, qat_epochs: int, recalib_every: int,
              n_train: int = 100, n_val: int = 32, n_test: int = 32,
              n_calib: int = 16, ft_epochs: int = 200,
              qat_lr: float = 1e-4, width: int = 32,
              modes: int = 16, n_layers: int = 2) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    from neuroflow.data.burgers import Burgers1dConfig, Burgers1dDataset
    from neuroflow.nn.fno import FNO1d
    from neuroflow.quant import quantise_model, build_fake_quant_model
    from neuroflow.quant.qat import (
        prepare_qat, recalibrate_qat, QATLinear)
    cfg = Burgers1dConfig(n_points=128, n_tsteps=20, nu=0.01, dt=0.01)
    train_set = Burgers1dDataset(n_samples=n_train, cfg=cfg, t_in=1,
                                  t_out=1, seed=seed)
    val_set = Burgers1dDataset(n_samples=n_val, cfg=cfg, t_in=1,
                                t_out=1, seed=seed + 1)
    test_set = Burgers1dDataset(n_samples=n_test, cfg=cfg, t_in=1,
                                 t_out=1, seed=seed + 2)

    def _to_xy(ds, n):
        return (torch.stack([ds[i][0] for i in range(n)], dim=0),
                torch.stack([ds[i][1] for i in range(n)], dim=0))
    x_train, y_train = _to_xy(train_set, n_train)
    x_val, y_val = _to_xy(val_set, n_val)
    x_test, y_test = _to_xy(test_set, n_test)
    # Pre-train FP32.
    model = FNO1d(in_channels=1, out_channels=1, width=width,
                   modes=modes, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(ft_epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 16):
            idx = perm[i:i+16]
            opt.zero_grad()
            loss = (model(x_train[idx]) - y_train[idx]).pow(2).mean()
            loss.backward()
            opt.step()
    calib_inputs = [x_val[i:i+1] for i in range(n_calib)]
    qm = quantise_model(model, calib_inputs, per_channel_weights=False,
                        per_token_activations=False, fp8_activations=False)
    qm_fp8 = quantise_model(model, calib_inputs, fp8_activations=True)
    fq_ptq_int8 = build_fake_quant_model(model, qm,
                                          use_fp8_activations=False)
    fq_ptq_int8.eval()
    fq_ptq_fp8 = build_fake_quant_model(model, qm_fp8,
                                         use_fp8_activations=True)
    fq_ptq_fp8.eval()
    # QAT.
    qat_model = prepare_qat(model, qm)
    qat_model.train()
    qat_opt = torch.optim.SGD(qat_model.parameters(), lr=qat_lr,
                                momentum=0.9)
    best_val = float("inf")
    best_state = None
    for ep in range(qat_epochs):
        qat_model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 16):
            idx = perm[i:i+16]
            qat_opt.zero_grad()
            yp = qat_model(x_train[idx])
            loss = (yp - y_train[idx]).pow(2).mean()
            loss.backward()
            qat_opt.step()
        if recalib_every > 0 and (ep + 1) % recalib_every == 0:
            qat_model.eval()
            recalibrate_qat(qat_model, calib_inputs)
            qat_model.train()
        with torch.no_grad():
            qat_model.eval()
            v = (torch.linalg.norm(qat_model(x_val) - y_val) /
                 torch.linalg.norm(y_val)).item()
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().clone()
                          for k, t in qat_model.state_dict().items()}
    if best_state is not None:
        qat_model.load_state_dict(best_state)
    qat_model.eval()
    # Evaluate.
    with torch.no_grad():
        y_fp32 = model(x_test).numpy()
        y_ptq_int8 = fq_ptq_int8(x_test).numpy()
        y_ptq_fp8 = fq_ptq_fp8(x_test).numpy()
        y_qat = qat_model(x_test).numpy()
    y_true = y_test.numpy()
    def rl2(a, b):
        return float(np.linalg.norm(
            (a - b).reshape(len(a), -1), axis=1
        ).mean() / max(np.linalg.norm(
            b.reshape(len(b), -1), axis=1
        ).mean(), 1e-8))
    return {
        "seed": seed,
        "fp32": rl2(y_fp32, y_true),
        "ptq_int8": rl2(y_ptq_int8, y_true),
        "ptq_fp8": rl2(y_ptq_fp8, y_true),
        "qat_bestval": rl2(y_qat, y_true),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--qat-epochs", type=int, default=100)
    parser.add_argument("--recalib-every", type=int, default=0)
    parser.add_argument("--ft-epochs", type=int, default=200)
    parser.add_argument("--out-csv", type=str,
                        default="artifacts/multiseed_recalib.csv")
    args = parser.parse_args()
    rows = []
    print(f"Running {args.n_seeds} seeds, qat_epochs={args.qat_epochs}, "
          f"recalib_every={args.recalib_every}")
    for seed in range(args.n_seeds):
        t0 = time.time()
        row = _run_seed(seed, qat_epochs=args.qat_epochs,
                        recalib_every=args.recalib_every,
                        ft_epochs=args.ft_epochs)
        print(f"  seed {seed}: fp32={row['fp32']:.2e}, "
              f"ptq_int8={row['ptq_int8']:.2e}, "
              f"ptq_fp8={row['ptq_fp8']:.2e}, "
              f"qat_bestval={row['qat_bestval']:.2e} "
              f"({time.time()-t0:.1f}s)")
        rows.append(row)
    # Aggregate.
    keys = ["fp32", "ptq_int8", "ptq_fp8", "qat_bestval"]
    print("\n--- Multi-seed summary ---")
    print(f"{'scheme':<15} {'mean':>10} {'std':>10}")
    for k in keys:
        vals = np.array([r[k] for r in rows])
        print(f"{k:<15} {vals.mean():.3e}   {vals.std():.3e}")
    # QAT reduction over PTQ INT8.
    reductions = [(r["ptq_int8"] - r["qat_bestval"]) / r["ptq_int8"] * 100
                  for r in rows]
    print(f"QAT vs PTQ_INT8 reduction: "
          f"mean={np.mean(reductions):+.1f}%, "
          f"std={np.std(reductions):.1f}%")
    # QAT vs PTQ FP8.
    qat_vs_fp8 = [(r["ptq_fp8"] - r["qat_bestval"]) / r["ptq_fp8"] * 100
                  for r in rows]
    print(f"QAT vs PTQ_FP8 reduction:  "
          f"mean={np.mean(qat_vs_fp8):+.1f}%, "
          f"std={np.std(qat_vs_fp8):.1f}%")
    # QAT beats FP8 count.
    n_beats = sum(1 for r in rows if r["qat_bestval"] < r["ptq_fp8"])
    print(f"QAT beats PTQ_FP8 on {n_beats}/{len(rows)} seeds")
    # Save CSV.
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("seed,fp32,ptq_int8,ptq_fp8,qat_bestval\n")
        for r in rows:
            f.write(f"{r['seed']},{r['fp32']:.6e},"
                    f"{r['ptq_int8']:.6e},{r['ptq_fp8']:.6e},"
                    f"{r['qat_bestval']:.6e}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
