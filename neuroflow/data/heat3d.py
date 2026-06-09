"""3D Poisson equation dataset generator (Stage 2 Sprint 2).

PDE:    -nabla^2 u = f   on Omega = (0, 1)^3
        u = 0              on boundary (Dirichlet, homogeneous)

Source f is a sum of 1..k Gaussian bumps placed at random interior points.
The analytical solution is recovered by a separable 3D FFT on the
discrete Laplacian eigenvalues (the "spectral Poisson solver" extended
to 3D):

    u_hat[i, j, k] = f_hat[i, j, k] / (lambda_i + lambda_j + lambda_k)
    u = ifftn(u_hat).real

where lambda_r = 2 * (1 - cos(r * pi / (N + 1))) * (N + 1)^2 for an
N-interior-point grid on (0, 1).

Each sample:
    x: (H, W, D, 1)  source field f
    y: (H, W, D, 1)  solution field u
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Heat3dConfig:
    """Configuration for the 3D Poisson dataset generator."""

    h: int = 16
    w: int = 16
    d: int = 16
    n_bumps_range: tuple[int, int] = (1, 3)
    bump_amplitude_range: tuple[float, float] = (0.5, 2.0)
    bump_width_range: tuple[float, float] = (0.08, 0.20)  # sigma in (0, 1)
    seed: int = 0


def _laplacian_eigenvalues(n: int) -> np.ndarray:
    """Same as the 2D helper; see neuroflow/data/heat2d.py."""
    k = np.arange(1, n + 1, dtype=np.float64)
    return 2.0 * (1.0 - np.cos(k * np.pi / (n + 1))) * (n + 1) ** 2


def _solve_poisson_3d(
    f: np.ndarray, lam_h: np.ndarray, lam_w: np.ndarray, lam_d: np.ndarray
) -> np.ndarray:
    """Solve -nabla^2 u = f with u=0 on boundary, via separable 3D FFT.

    Args:
        f: (H, W, D) source field, uniform cell-centre grid on (0, 1).
        lam_h, lam_w, lam_d: 1D Laplacian eigenvalues of length H, W, D.

    Returns:
        u: (H, W, D) solution.
    """
    f_hat = np.fft.fftn(f)
    lam = lam_h[:, None, None] + lam_w[None, :, None] + lam_d[None, None, :]
    with np.errstate(divide="ignore", invalid="warn"):
        u_hat = np.where(lam > 0, f_hat / lam, 0.0)
    u = np.fft.ifftn(u_hat).real
    return u.astype(np.float32)


def _sample_source_field(
    cfg: Heat3dConfig, rng: np.random.Generator
) -> np.ndarray:
    """Random Gaussian-sum source field of shape (H, W, D) on (0, 1)^3."""
    n_bumps = int(rng.integers(cfg.n_bumps_range[0], cfg.n_bumps_range[1] + 1))
    h_grid = (np.arange(cfg.h) + 0.5) / (cfg.h + 1)
    w_grid = (np.arange(cfg.w) + 0.5) / (cfg.w + 1)
    d_grid = (np.arange(cfg.d) + 0.5) / (cfg.d + 1)
    H, W, D = np.meshgrid(h_grid, w_grid, d_grid, indexing="ij")
    f = np.zeros((cfg.h, cfg.w, cfg.d), dtype=np.float64)
    for _ in range(n_bumps):
        A = rng.uniform(*cfg.bump_amplitude_range)
        sigma = rng.uniform(*cfg.bump_width_range)
        x0 = rng.uniform(sigma, 1.0 - sigma)
        y0 = rng.uniform(sigma, 1.0 - sigma)
        z0 = rng.uniform(sigma, 1.0 - sigma)
        f += A * np.exp(
            -(((H - x0) ** 2 + (W - y0) ** 2 + (D - z0) ** 2) / (2.0 * sigma * sigma))
        )
    return f


def generate_heat3d_sample(
    cfg: Heat3dConfig, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single (x, y) sample.

    Returns:
        x: (H, W, D, 1) float32 source field.
        y: (H, W, D, 1) float32 solution field.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    f = _sample_source_field(cfg, rng)
    lam_h = _laplacian_eigenvalues(cfg.h)
    lam_w = _laplacian_eigenvalues(cfg.w)
    lam_d = _laplacian_eigenvalues(cfg.d)
    u = _solve_poisson_3d(f, lam_h, lam_w, lam_d)
    return f[..., None].astype(np.float32), u[..., None].astype(np.float32)


class Heat3dDataset(Dataset):
    """In-memory 3D Poisson dataset.

    Each item is (x, y):
        x: (H, W, D, 1) source field.
        y: (H, W, D, 1) solution field.
    """

    def __init__(
        self,
        n_samples: int = 200,
        cfg: Heat3dConfig | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg or Heat3dConfig(seed=seed)
        self.n_samples = n_samples
        rng = np.random.default_rng(seed)
        xs = np.empty(
            (n_samples, self.cfg.h, self.cfg.w, self.cfg.d, 1), dtype=np.float32
        )
        ys = np.empty_like(xs)
        for i in range(n_samples):
            x, y = generate_heat3d_sample(self.cfg, rng=rng)
            xs[i] = x
            ys[i] = y
        self._x = xs
        self._y = ys

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self._x[idx]), torch.from_numpy(self._y[idx])
