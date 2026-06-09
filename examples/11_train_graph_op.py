"""Stage 2 Sprint 3.2 demo (1/2): train a GraphOp (GCN-style) operator on a
1D per-node regression task.

Mirrors examples/09_train_tokenmixer.py but with a graph operator. The
ground truth is the same per-point nonlinear function

    y(s_i) = sin(u(s_i)) + 0.5 * cos(2 * u(s_i))

where u(s) is a random sum of sinusoids. The model receives a per-node
feature (u(s_i), s_i) and must output y(s_i). The line graph (1D
neighbour connectivity) gives the operator some spatial context, but
the target is still per-node, so the loss curve is mostly a sanity
check on the end-to-end pipeline rather than a benchmark against FNO /
TokenMixer.

Run:
    python examples/11_train_graph_op.py

Outputs (under ./artifacts/):
    - graphop_demo.pt            : PyTorch checkpoint
    - graphop_demo.neuroir       : exported IR (v0.6.0 JSON)
    - graphop_demo.nneuroir      : exported IR (v0.6.0 binary)
    - graphop_demo_train.png     : loss curve
    - graphop_demo_pred.png      : true vs predicted y (first val sample)
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

from neuroflow.data.token_mixer_demo import TokenMixerDemo1dConfig, TokenMixerDemo1dDataset
from neuroflow.ir.export import export_all
from neuroflow.nn.graph_op import GraphOp, GraphOpConfig
from neuroflow.utils.plotting import plot_loss_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GraphOp on 1D regression")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--n-nodes", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
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
    data_cfg = TokenMixerDemo1dConfig(n_points=args.n_nodes, seed=args.seed)
    train_set = TokenMixerDemo1dDataset(n_samples=args.n_train, cfg=data_cfg, seed=args.seed)
    val_set = TokenMixerDemo1dDataset(n_samples=args.n_val, cfg=data_cfg, seed=args.seed + 1)
    print(f"   train = {len(train_set)}, val = {len(val_set)}, n_nodes = {args.n_nodes}")

    print("==> Building model")
    model_cfg = GraphOpConfig(
        in_dim=2,
        out_dim=1,
        n_nodes=args.n_nodes,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        activation="gelu",
        name="graphop_demo",
    )
    model = GraphOp(model_cfg)
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
        for x, target in train_loader:
            optim.zero_grad()
            pred = model(x)
            loss = rel_l2(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * x.size(0)
            train_count += x.size(0)
        train_loss /= max(train_count, 1)
        sched.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for x, target in val_loader:
                pred = model(x)
                loss = rel_l2(pred, target)
                val_loss += loss.item() * x.size(0)
                val_count += x.size(0)
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
        save_path=out_dir / "graphop_demo_train.png",
    )

    print("==> Visualizing a prediction")
    model.eval()
    x, target = val_set[0]
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).squeeze(0).squeeze(-1).numpy()
    target_np = target.squeeze(-1).numpy()
    pred_loss = float(
        np.linalg.norm(pred - target_np) / max(np.linalg.norm(target_np), 1e-8)
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x_axis = np.arange(len(target_np))
    ax.plot(x_axis, target_np, label="true y(s_i)", marker="o", markersize=3)
    ax.plot(x_axis, pred, label="GraphOp pred", marker="s", markersize=3)
    ax.set_xlabel("node index")
    ax.set_ylabel("y(s_i)")
    ax.set_title(f"GraphOp 1D demo (val[0] rel L2 = {pred_loss:.2%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "graphop_demo_pred.png", dpi=120)
    plt.close(fig)
    print(f"   val[0] rel L2 = {pred_loss:.2%}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(model_cfg),
            "history": history,
        },
        out_dir / "graphop_demo.pt",
    )

    print("==> Exporting NeuroIR v0.6.0 (JSON + binary)")
    json_path, bin_path = export_all(model, out_dir, basename="graphop_demo")
    print(f"   JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"   BIN : {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")

    print("==> Done.")


if __name__ == "__main__":
    main()
