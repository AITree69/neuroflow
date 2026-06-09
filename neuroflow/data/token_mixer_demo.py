"""Demo dataset for TokenMixer (Transolver-style) — per-point regression.

Task:
    Take a 1D function u sampled at n_points grid points and predict a
    per-point target

        y(s_i) = sin(u(s_i)) + 0.5 * cos(2 * u(s_i))     (1D demo).

    We feed the model the per-point feature (u(s_i), s_i) so it has both
    the input function value and the spatial location. The model must
    learn the per-point nonlinear map (no cross-point dependency is
    required for the ground truth, so the TokenMixer's attention over
    patches is an overkill that still works — the loss curve is mostly
    driven by the per-point MLP pathway).

    This is a deliberately easy, fully observable regression: there is no
    spatial correlation in the target, so the model is testing the
    end-to-end IR + C++ parity pipeline rather than the algorithm's
    representational power. Use a more interesting operator (e.g. 2D
    Darcy, 3D heat) when comparing against FNO baselines.

The dataset returns a 3-tuple ``(x, y)`` where both are ``(n_points, in_dim)``
and ``(n_points, out_dim)`` respectively.  No batch dimension is included
(Dataset / DataLoader will stack them).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TokenMixerDemo1dConfig:
    n_points: int = 64
    n_freq: int = 3
    freq_low: float = 0.5
    freq_high: float = 3.0
    noise_std: float = 0.0
    seed: int = 0


def _make_u(s_grid: np.ndarray, n_freq: int, low: float, high: float,
            rng: np.random.Generator) -> np.ndarray:
    """Random sum of sinusoids evaluated on `s_grid` (in [0, 1])."""
    freqs = rng.uniform(low, high, size=n_freq)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_freq)
    amps = rng.uniform(0.3, 1.0, size=n_freq)
    u = np.zeros_like(s_grid)
    for f, p, a in zip(freqs, phases, amps):
        u = u + a * np.sin(2.0 * np.pi * f * s_grid + p)
    return u


def _target(u: np.ndarray) -> np.ndarray:
    return np.sin(u) + 0.5 * np.cos(2.0 * u)


class TokenMixerDemo1dDataset(Dataset):
    """Synthetic per-point regression dataset for TokenMixer.

    Returns:
        x:  (n_points, 2) float32 — stacked [u(s_i), s_i]
        y:  (n_points, 1) float32 — target value
    """

    def __init__(self, n_samples: int, cfg: TokenMixerDemo1dConfig | None = None,
                 seed: int = 0) -> None:
        if cfg is None:
            cfg = TokenMixerDemo1dConfig(seed=seed)
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        s = (np.arange(cfg.n_points) + 0.5) / cfg.n_points
        self.s_grid = s.astype(np.float32)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for _ in range(n_samples):
            u = _make_u(s, cfg.n_freq, cfg.freq_low, cfg.freq_high, rng)
            if cfg.noise_std > 0:
                u = u + rng.normal(0.0, cfg.noise_std, size=u.shape)
            t = _target(u)
            xs.append(np.stack([u, s], axis=-1).astype(np.float32))
            ys.append(t.astype(np.float32).reshape(-1, 1))
        self.x = torch.from_numpy(np.stack(xs, axis=0))    # (N, n_points, 2)
        self.y = torch.from_numpy(np.stack(ys, axis=0))    # (N, n_points, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
