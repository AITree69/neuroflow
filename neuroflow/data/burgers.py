"""1D Burgers' equation dataset generator.

PDE:    du/dt + u du/dx = nu * d2u/dx2
Scheme: 2nd-order central differences in space, RK4 in time, periodic boundary.

This is the canonical benchmark from Li et al. (FNO, ICLR 2021).

Each sample is a trajectory of `n_tsteps` time steps over a uniform grid of `n_points`.
Initial condition: u(x,0) ~ sum of A_k * sin(k*x + phi_k), A_k ~ N(0, decay).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Burgers1dConfig:
    n_points: int = 256
    n_tsteps: int = 100
    nu: float = 0.01
    dt: float = 0.001
    domain: tuple[float, float] = (0.0, 2 * np.pi)
    n_modes_ic: int = 5
    decay_rate: float = 2.0
    seed: int = 0


def generate_burgers1d_trajectory(
    cfg: Burgers1dConfig,
    u0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a single 1D Burgers trajectory.

    Returns:
        u: array of shape (n_tsteps, n_points) (excluding the initial condition).
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    if u0 is None:
        x = np.linspace(cfg.domain[0], cfg.domain[1], cfg.n_points, endpoint=False)
        u0 = np.zeros(cfg.n_points)
        for k in range(1, cfg.n_modes_ic + 1):
            amp = rng.normal(0.0, 1.0 / (k ** cfg.decay_rate))
            phase = rng.uniform(0.0, 2 * np.pi)
            u0 += amp * np.sin(k * x + phase)

    dx = (cfg.domain[1] - cfg.domain[0]) / cfg.n_points
    dt = cfg.dt
    nu = cfg.nu
    n = cfg.n_points
    traj = np.zeros((cfg.n_tsteps, n), dtype=np.float32)
    u = u0.astype(np.float32).copy()

    def rhs(u: np.ndarray) -> np.ndarray:
        # Periodic finite differences
        u_left = np.roll(u, 1)
        u_right = np.roll(u, -1)
        u_xx = (u_right - 2.0 * u + u_left) / (dx * dx)
        u_dx = (u_right - u_left) / (2.0 * dx)
        return -u * u_dx + nu * u_xx

    for t in range(cfg.n_tsteps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[t] = u

    return traj


class Burgers1dDataset(Dataset):
    """In-memory Burgers 1D dataset.

    Each item is (input_window, target_window):
        - input_window: (T_in, n_points) of past states
        - target_window: (T_out, n_points) of future states
    Both are stored as float32 tensors.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        cfg: Burgers1dConfig | None = None,
        t_in: int = 10,
        t_out: int = 10,
        seed: int = 0,
        trajectory_stride: int = 10,
    ) -> None:
        super().__init__()
        self.cfg = cfg or Burgers1dConfig(seed=seed)
        self.t_in = t_in
        self.t_out = t_out
        self.trajectory_stride = trajectory_stride

        rng = np.random.default_rng(seed)
        self._trajectories: list[np.ndarray] = []
        self._indices: list[tuple[int, int]] = []

        for sample_idx in range(n_samples):
            traj = generate_burgers1d_trajectory(self.cfg, rng=rng)
            self._trajectories.append(traj)
            # Sliding windows: t_in history → t_out future
            # We use trajectory[0] as the implicit initial condition (returned as part of history)
            # trajectory is already (n_tsteps, n_points); we need t_in + t_out frames.
            available = self.cfg.n_tsteps - t_in - t_out + 1
            for start in range(0, available, trajectory_stride):
                self._indices.append((sample_idx, start))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        traj_idx, start = self._indices[idx]
        traj = self._trajectories[traj_idx]
        # x: (t_in, n) -> (n, t_in) so that the batched shape becomes
        # (batch, n, in_channels) — what FNO1d expects.
        x = traj[start : start + self.t_in].T  # (n, t_in)
        y = traj[start + self.t_in : start + self.t_in + self.t_out].T  # (n, t_out)
        return torch.from_numpy(x), torch.from_numpy(y)

    def iter_trajectories(self) -> Iterator[np.ndarray]:
        return iter(self._trajectories)
