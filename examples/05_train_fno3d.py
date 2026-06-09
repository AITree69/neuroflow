"""Stage 2 Sprint 2 demo (1/2): train a small FNO3d on 3D Poisson.

PDE:    -nabla^2 u = f   on (0, 1)^3,   u = 0 on boundary
Source: sum of 1-3 Gaussian bumps, placed randomly in the interior.
Solution: analytical via separable 3D FFT (see neuroflow/data/heat3d.py).

Run:
    python examples/05_train_fno3d.py

Outputs (under ./artifacts/):
    - fno3d_poisson.pt        : PyTorch checkpoint
    - fno3d_poisson.neuroir   : exported IR (v0.3.0 JSON)
    - fno3d_poisson.nneuroir  : exported IR (v0.3.0 binary)
    - fno3d_poisson_train.png : loss curve
    - fno3d_poisson_pred.png  : true vs predicted (first val sample, mid-slice)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from neuroflow.data.heat3d import Heat3dConfig, Heat3dDataset
from neuroflow.ir.export import export_all
from neuroflow.nn.fno3d import FNO3d, FNO3dConfig
from neuroflow.train import Trainer, TrainConfig
from neuroflow.utils.plotting import plot_2d_field_comparison, plot_loss_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FNO3d on 3D Poisson")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--w", type=int, default=16)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--modes-h", type=int, default=4)
    parser.add_argument("--modes-w", type=int, default=4)
    parser.add_argument("--modes-d", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="./artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("==> Building datasets")
    cfg = Heat3dConfig(h=args.h, w=args.w, d=args.d, seed=args.seed)
    train_set = Heat3dDataset(n_samples=args.n_train, cfg=cfg, seed=args.seed)
    val_set = Heat3dDataset(n_samples=args.n_val, cfg=cfg, seed=args.seed + 1)
    print(
        f"   train = {len(train_set)}, val = {len(val_set)}, "
        f"grid = {args.h}x{args.w}x{args.d}, in_ch=1, out_ch=1"
    )

    print("==> Building model")
    model_cfg = FNO3dConfig(
        in_channels=1,
        out_channels=1,
        width=args.width,
        modes_h=args.modes_h,
        modes_w=args.modes_w,
        modes_d=args.modes_d,
        n_layers=args.n_layers,
        name="fno3d_poisson",
    )
    model = FNO3d(model_cfg)
    print(f"   parameters = {model.num_parameters():,}")

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Relative L2 — matches the FNO paper's headline metric.
        diff = (pred - target).reshape(pred.size(0), -1)
        ref = target.reshape(target.size(0), -1)
        num = torch.linalg.norm(diff, dim=1)
        den = torch.linalg.norm(ref, dim=1).clamp_min(1e-8)
        return torch.mean(num / den)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_set=train_set,
        val_set=val_set,
        cfg=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_step=max(1, args.epochs // 4),
            log_every=max(1, args.epochs // 10),
            save_path=str(out_dir / "fno3d_poisson.pt"),
            device="cpu",
            seed=args.seed,
        ),
    )

    print("==> Training")
    t0 = time.time()
    history = trainer.fit()
    print(f"   elapsed = {history.elapsed_sec:.1f}s")

    plot_loss_curve(
        history.epoch,
        history.train_loss,
        history.val_loss,
        save_path=out_dir / "fno3d_poisson_train.png",
    )

    print("==> Visualizing a prediction (mid-D slice)")
    model.eval()
    x, y = val_set[0]
    with torch.no_grad():
        y_pred = model(x.unsqueeze(0)).squeeze(0).squeeze(-1).numpy()
    y_true = y.squeeze(-1).numpy()
    rel_l2 = float(np.linalg.norm(y_true - y_pred) / max(np.linalg.norm(y_true), 1e-8))
    mid_d = y_true.shape[-1] // 2
    plot_2d_field_comparison(
        y_true[..., mid_d],
        y_pred[..., mid_d],
        title=f"FNO3d on 3D Poisson (rel L2 = {rel_l2:.2%}, mid-D slice)",
        save_path=out_dir / "fno3d_poisson_pred.png",
    )
    print(f"   val[0] rel L2 = {rel_l2:.2%}")

    print("==> Exporting NeuroIR v0.3.0 (JSON + binary)")
    json_path, bin_path = export_all(model, out_dir, basename="fno3d_poisson")
    print(f"   JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"   BIN : {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    print("==> Done.")


if __name__ == "__main__":
    main()
