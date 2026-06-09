"""1D integral operator dataset for DeepONet (Stage 2 Sprint 2).

Operator:   G(u)(x) = integral from 0 to x of u(s) ds
Domain:     x in [0, 1]
Source:     u(s) = sum_k A_k * sin(k * pi * s + phi_k)
            A_k ~ N(0, 1 / (k ** decay)),  phi_k ~ Uniform(0, 2*pi)
Solution:   computed numerically via the trapezoidal rule on a uniform
            grid of `n_sensor` points.

This is the canonical DeepONet benchmark from Lu Lu et al. 2021.
We follow the standard "stacked sensor" trick: the branch input
carries BOTH the sensor location s_i AND the sensor value u(s_i) as
a 2-column feature (in_branch=2). Without the location feature, the
branch net is permutation-invariant but has no information about
where each sensor sits, which makes the integral operator
inherently hard to learn (the output at x depends on the
*integrated* shape of u on [0, x], which the net cannot recover
without knowing the order of the samples).

Each sample:
    u: (n_sensor, in_branch=2)     stacked [s_i, u(s_i)] features
    y: (n_query, in_trunk=1)       query points x in [0, 1]
    out: (n_query, out_channels)    G(u)(y) computed by trapezoidal rule
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class IntegralOp1dConfig:
    n_sensor: int = 100
    n_query: int = 50
    in_branch: int = 2  # [s, u(s)]
    in_trunk: int = 1   # [x]
    out_channels: int = 1
    n_modes_ic: int = 5
    decay_rate: float = 2.0
    seed: int = 0


def _sample_function(
    cfg: IntegralOp1dConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample u on a uniform grid of n_sensor points in (0, 1)."""
    s_grid = (np.arange(cfg.n_sensor) + 0.5) / cfg.n_sensor
    u = np.zeros(cfg.n_sensor, dtype=np.float64)
    for k in range(1, cfg.n_modes_ic + 1):
        amp = rng.normal(0.0, 1.0 / (k ** cfg.decay_rate))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u += amp * np.sin(k * np.pi * s_grid + phase)
    return s_grid.astype(np.float32), u.astype(np.float32)


def _integrate_cumulative(u: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    """Compute G(u)(x) = integral_0^x u(s) ds for each x in query_x.

    u is sampled uniformly on (0, 1) at n_sensor cell centres
    s_j = (j + 0.5) / n_sensor. The trapezoidal rule is used as the
    discrete approximation.
    """
    n_sensor = u.shape[0]
    s_grid = (np.arange(n_sensor) + 0.5) / n_sensor
    ds = 1.0 / n_sensor
    trap = 0.5 * (u[:-1] + u[1:]) * ds
    cum_int_at_cell = np.concatenate([[0.0], np.cumsum(trap)])
    G = np.interp(query_x, s_grid, cum_int_at_cell).astype(np.float32)
    return G


def generate_integral_op_sample(
    cfg: IntegralOp1dConfig, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    s, u = _sample_function(cfg, rng)
    query_x = (np.arange(cfg.n_query) + 0.5) / cfg.n_query
    query_x = query_x.astype(np.float32)
    G = _integrate_cumulative(u, query_x)
    # Stack [s, u] along the feature axis.
    branch_in = np.stack([s, u], axis=-1).astype(np.float32)
    return (
        branch_in.reshape(cfg.n_sensor, cfg.in_branch).astype(np.float32),
        query_x.reshape(cfg.n_query, cfg.in_trunk).astype(np.float32),
        G.reshape(cfg.n_query, cfg.out_channels).astype(np.float32),
    )


class IntegralOp1dDataset(Dataset):
    """In-memory 1D integral operator dataset.

    Each item is (u, y, out):
        u: (n_sensor, in_branch=2)     stacked [s_i, u(s_i)]
        y: (n_query, in_trunk=1)       query points
        out: (n_query, out_channels)   G(u)(y)
    """

    def __init__(
        self,
        n_samples: int = 500,
        cfg: IntegralOp1dConfig | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg or IntegralOp1dConfig(seed=seed)
        self.n_samples = n_samples
        rng = np.random.default_rng(seed)
        us = np.empty((n_samples, self.cfg.n_sensor, self.cfg.in_branch), dtype=np.float32)
        ys = np.empty((n_samples, self.cfg.n_query, self.cfg.in_trunk), dtype=np.float32)
        outs = np.empty((n_samples, self.cfg.n_query, self.cfg.out_channels), dtype=np.float32)
        for i in range(n_samples):
            u, y, o = generate_integral_op_sample(self.cfg, rng=rng)
            us[i] = u
            ys[i] = y
            outs[i] = o
        self._u = us
        self._y = ys
        self._out = outs

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._u[idx]),
            torch.from_numpy(self._y[idx]),
            torch.from_numpy(self._out[idx]),
        )
