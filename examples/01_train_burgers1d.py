"""Stage 1 demo (1/2): train a small FNO1d on 1D Burgers' equation.

Run:
    python examples/01_train_burgers1d.py

Outputs (under ./artifacts/):
    - fno1d_burgers.pt        : PyTorch checkpoint
    - fno1d_burgers.neuroir   : exported IR (v0 JSON)
    - fno1d_burgers_train.png : loss curve
    - fno1d_burgers_pred.png  : true vs predicted
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from neuroflow.data.burgers import Burgers1dConfig, Burgers1dDataset
from neuroflow.ir.export import export_all
from neuroflow.nn.fno import FNO1d, FNO1dConfig
from neuroflow.train import Trainer, TrainConfig
from neuroflow.utils.plotting import plot_burgers_prediction, plot_loss_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FNO1d on Burgers 1D")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--n-tsteps", type=int, default=200)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=20)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="./artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("==> Building datasets")
    cfg = Burgers1dConfig(n_points=args.n_points, n_tsteps=args.n_tsteps, seed=args.seed)
    train_set = Burgers1dDataset(
        n_samples=args.n_train, cfg=cfg, t_in=args.t_in, t_out=args.t_out, seed=args.seed
    )
    val_set = Burgers1dDataset(
        n_samples=args.n_val, cfg=cfg, t_in=args.t_in, t_out=args.t_out, seed=args.seed + 1
    )
    print(
        f"   train samples = {len(train_set)}, val samples = {len(val_set)}, "
        f"t_in = {args.t_in}, t_out = {args.t_out}, n = {args.n_points}"
    )

    print("==> Building model")
    model_cfg = FNO1dConfig(
        in_channels=args.t_in,
        out_channels=args.t_out,
        width=args.width,
        modes=args.modes,
        n_layers=args.n_layers,
        name="fno1d_burgers",
    )
    model = FNO1d(model_cfg)
    print(f"   parameters = {model.num_parameters():,}")

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)

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
            save_path=str(out_dir / "fno1d_burgers.pt"),
            device="cpu",
            seed=args.seed,
        ),
    )

    print("==> Training")
    t0 = time.time()
    history = trainer.fit()
    print(f"   elapsed = {history.elapsed_sec:.1f}s")

    # Plots
    plot_loss_curve(
        history.epoch,
        history.train_loss,
        history.val_loss,
        save_path=out_dir / "fno1d_burgers_train.png",
    )

    # Visualize prediction on first val sample
    print("==> Visualizing a prediction")
    model.eval()
    x, y = val_set[0]
    with torch.no_grad():
        y_pred = model(x.unsqueeze(0)).squeeze(0).numpy()
    # x/y are (n, t_in/t_out); plot_burgers_prediction expects (T, n).
    y_np = y.numpy().T
    x_grid = np.linspace(0.0, 2 * np.pi, args.n_points, endpoint=False)
    plot_burgers_prediction(
        x_grid,
        y_np,
        y_pred.T,
        title=f"FNO1d on Burgers 1D (L2 rel = "
        f"{np.linalg.norm(y_np - y_pred.T) / np.linalg.norm(y_np):.2%})",
        save_path=out_dir / "fno1d_burgers_pred.png",
    )

    # Export IR (both JSON .neuroir and binary .nneuroir)
    print("==> Exporting NeuroIR v0 (JSON + binary)")
    json_path, bin_path = export_all(model, out_dir, basename="fno1d_burgers")
    print(f"   JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"   BIN : {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    print("==> Done.")


if __name__ == "__main__":
    main()
