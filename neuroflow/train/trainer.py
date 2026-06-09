"""Minimal trainer for neural operators.

Stage 1: Adam + L2 loss + step LR schedule. No fancy distributed machinery.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    lr_step: int = 5
    lr_gamma: float = 0.5
    weight_decay: float = 0.0
    log_every: int = 50
    save_path: str | None = None
    device: str = "cpu"
    grad_clip: float | None = 1.0
    seed: int = 0


@dataclass
class TrainHistory:
    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    elapsed_sec: float = 0.0


class Trainer:
    """Minimal trainer. Single-GPU / CPU, no DDP."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_set: Dataset,
        val_set: Dataset | None = None,
        cfg: TrainConfig | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_set = train_set
        self.val_set = val_set
        self.cfg = cfg or TrainConfig()
        self.history = TrainHistory()
        torch.manual_seed(self.cfg.seed)

    def _make_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_loader = DataLoader(
            self.train_set, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader: DataLoader | None = None
        if self.val_set is not None:
            val_loader = DataLoader(
                self.val_set, batch_size=self.cfg.batch_size, shuffle=False
            )
        return train_loader, val_loader

    def _train_epoch(self, loader: DataLoader, optim: torch.optim.Optimizer) -> float:
        self.model.train()
        total, count = 0.0, 0
        for x, y in loader:
            x = x.to(self.cfg.device)
            y = y.to(self.cfg.device)
            optim.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            optim.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(count, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for x, y in loader:
            x = x.to(self.cfg.device)
            y = y.to(self.cfg.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(count, 1)

    def fit(self) -> TrainHistory:
        train_loader, val_loader = self._make_loaders()
        optim = Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        sched = StepLR(optim, step_size=self.cfg.lr_step, gamma=self.cfg.lr_gamma)

        start = time.time()
        for epoch in range(self.cfg.epochs):
            train_loss = self._train_epoch(train_loader, optim)
            val_loss = self._eval_epoch(val_loader) if val_loader else float("nan")
            sched.step()
            lr = optim.param_groups[0]["lr"]
            self.history.epoch.append(epoch)
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.lr.append(lr)
            if epoch % max(self.cfg.log_every, 1) == 0 or epoch == self.cfg.epochs - 1:
                print(
                    f"[epoch {epoch:3d}] train={train_loss:.4e} "
                    f"val={val_loss:.4e} lr={lr:.2e}"
                )
        self.history.elapsed_sec = time.time() - start

        if self.cfg.save_path is not None:
            self.save(self.cfg.save_path)
        return self.history

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": asdict(self.cfg),
                "history": asdict(self.history),
            },
            p,
        )
        meta = p.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {
                    "config": asdict(self.cfg),
                    "history": asdict(self.history),
                    "n_params": sum(x.numel() for x in self.model.parameters()),
                },
                indent=2,
            )
        )
