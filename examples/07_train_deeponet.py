"""Stage 2 Sprint 2 demo (3/4): train a DeepONet on the 1D integral operator.

Operator:   G(u)(x) = integral from 0 to x of u(s) ds
Data:       neuroflow.data.integral_op.IntegralOp1dDataset
            (trapezoidal-rule ground truth; u is a sum-of-sines random
            function sampled at `n_sensor` points).

Run:
    python examples/07_train_deeponet.py

Outputs (under ./artifacts/):
    - deeponet_integral.pt        : PyTorch checkpoint
    - deeponet_integral.neuroir   : exported IR (v0.4.0 JSON)
    - deeponet_integral.nneuroir  : exported IR (v0.4.0 binary)
    - deeponet_integral_train.png : loss curve
    - deeponet_integral_pred.png  : true vs predicted integral (first val sample)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from neuroflow.data.integral_op import IntegralOp1dConfig, IntegralOp1dDataset
from neuroflow.ir.export import export_all
from neuroflow.nn.deeponet import DeepONet, DeepONetConfig
from neuroflow.utils.plotting import plot_loss_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepONet on 1D integral operator")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--n-sensor", type=int, default=100)
    parser.add_argument("--n-query", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-layers-branch", type=int, default=3)
    parser.add_argument("--n-layers-trunk", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="./artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("==> Building datasets")
    cfg = IntegralOp1dConfig(
        n_sensor=args.n_sensor, n_query=args.n_query, seed=args.seed,
    )
    train_set = IntegralOp1dDataset(n_samples=args.n_train, cfg=cfg, seed=args.seed)
    val_set = IntegralOp1dDataset(n_samples=args.n_val, cfg=cfg, seed=args.seed + 1)
    print(
        f"   train = {len(train_set)}, val = {len(val_set)}, "
        f"n_sensor = {args.n_sensor}, n_query = {args.n_query}"
    )

    print("==> Building model")
    model_cfg = DeepONetConfig(
        in_branch=cfg.in_branch,
        in_trunk=cfg.in_trunk,
        latent_dim=args.latent_dim,
        out_channels=cfg.out_channels,
        hidden_branch=args.hidden,
        hidden_trunk=args.hidden,
        n_layers_branch=args.n_layers_branch,
        n_layers_trunk=args.n_layers_trunk,
        name="deeponet_integral",
    )
    model = DeepONet(model_cfg)
    print(f"   parameters = {model.num_parameters():,}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    optim = Adam(model.parameters(), lr=args.lr)
    sched = StepLR(optim, step_size=max(1, args.epochs // 4), gamma=0.5)

    def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).reshape(pred.size(0), -1)
        ref = target.reshape(target.size(0), -1)
        return torch.mean(
            torch.linalg.norm(diff, dim=1)
            / torch.linalg.norm(ref, dim=1).clamp_min(1e-8)
        )

    history = {"epoch": [], "train": [], "val": []}
    print("==> Training")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for u, y, target in train_loader:
            optim.zero_grad()
            pred = model(u, y)
            loss = rel_l2(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * u.size(0)
            train_count += u.size(0)
        train_loss /= max(train_count, 1)
        sched.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for u, y, target in val_loader:
                pred = model(u, y)
                loss = rel_l2(pred, target)
                val_loss += loss.item() * u.size(0)
                val_count += u.size(0)
        val_loss /= max(val_count, 1)
        history["epoch"].append(epoch)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(
                f"[epoch {epoch:3d}] train={train_loss:.4e} val={val_loss:.4e} "
                f"lr={optim.param_groups[0]['lr']:.2e}"
            )
    print(f"   elapsed = {time.time() - t0:.1f}s")

    plot_loss_curve(
        history["epoch"],
        history["train"],
        history["val"],
        save_path=out_dir / "deeponet_integral_train.png",
    )

    print("==> Visualizing a prediction")
    model.eval()
    u, y, target = val_set[0]
    with torch.no_grad():
        pred = model(u.unsqueeze(0), y.unsqueeze(0)).squeeze(0).squeeze(-1).numpy()
    target_np = target.squeeze(-1).numpy()
    pred_loss = float(
        np.linalg.norm(pred - target_np) / max(np.linalg.norm(target_np), 1e-8)
    )
    # Plot the integral of u (true) vs DeepONet output (pred) along x.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x_axis = np.arange(len(target_np))
    ax.plot(x_axis, target_np, label="true G(u)(x)", marker="o", markersize=3)
    ax.plot(x_axis, pred, label="DeepONet pred", marker="s", markersize=3)
    ax.set_xlabel("query index")
    ax.set_ylabel("G(u)(x)")
    ax.set_title(f"DeepONet 1D integral operator (val[0] rel L2 = {pred_loss:.2%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "deeponet_integral_pred.png", dpi=120)
    plt.close(fig)
    print(f"   val[0] rel L2 = {pred_loss:.2%}")

    # Save the PyTorch checkpoint for reproducibility.
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(model_cfg),
            "history": history,
        },
        out_dir / "deeponet_integral.pt",
    )

    print("==> Exporting NeuroIR v0.4.0 (JSON + binary)")
    json_path, bin_path = export_all(model, out_dir, basename="deeponet_integral")
    print(f"   JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"   BIN : {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    print("==> Done.")


if __name__ == "__main__":
    main()
