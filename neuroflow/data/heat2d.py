"""2D Poisson equation dataset generator (Stage 2 Sprint 1 toy).

PDE:    -nabla^2 u = f   on Omega = (0, 1)^2
        u = 0              on boundary (Dirichlet, homogeneous)

Source f is a sum of 1..k Gaussian bumps placed at random interior points.
The analytical solution is recovered by a 2D FFT on the discrete Laplacian
eigenvalues (the standard "spectral Poisson solver"):

    u_hat[m, n] = f_hat[m, n] / (lambda_m + lambda_n)
    u = ifft2(u_hat).real

where lambda_j = 2 * (1 - cos(j * pi / N)) for a unit-interval grid of N+1
points (Neumann-Neumann for the interior, Dirichlet boundary; equivalent
to a sine expansion). The 1/N^2 scale cancels the ifft normalization.

Each sample:
    x: (H, W, 1)  source field f
    y: (H, W, 1)  solution field u

This is intentionally simpler than the FNO paper's Darcy benchmark
(where a(x) is heterogeneous and the solution needs FEM), but it
exercises the exact FNO2d forward pass and is a faithful first Sprint 1
demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Heat2dConfig:
    """Configuration for the 2D Poisson dataset generator."""

    h: int = 32
    w: int = 32
    n_bumps_range: tuple[int, int] = (1, 4)  # uniform integer in [a, b]
    bump_amplitude_range: tuple[float, float] = (0.5, 2.0)
    bump_width_range: tuple[float, float] = (0.05, 0.15)  # sigma in (0, 1)
    seed: int = 0


def _laplacian_eigenvalues(n: int) -> np.ndarray:
    """Eigenvalues of the 1D second-difference operator with homogeneous
    Dirichlet BC on a uniform grid of `n` interior points (scaled to (0, 1)).

    For a Dirichlet problem on (0, 1) with N+1 points (including the
    boundary), the eigenvalues are 2 * (1 - cos(k * pi / (N + 1))) / dx^2.
    We work with `n` interior points and a cell-centre grid of spacing
    1 / (n + 1) so the eigenvalues are 2 * (1 - cos(k * pi / (n + 1))) *
    (n + 1)^2. We return them already multiplied by (n + 1)^2.
    """
    k = np.arange(1, n + 1, dtype=np.float64)
    return 2.0 * (1.0 - np.cos(k * np.pi / (n + 1))) * (n + 1) ** 2


def _solve_poisson_2d(
    f: np.ndarray, lam_h: np.ndarray, lam_w: np.ndarray
) -> np.ndarray:
    """Solve -nabla^2 u = f with u=0 on boundary, via separable FFT.

    Args:
        f: (H, W) source field, uniform grid on (0, 1) with cell centres
           at (i+0.5) / (H+1), (j+0.5) / (W+1).
        lam_h, lam_w: 1D Laplacian eigenvalues of length H, W respectively.

    Returns:
        u: (H, W) solution.
    """
    f_hat = np.fft.fft2(f)  # not 2*rfft — we need the full complex spectrum
    lam = lam_h[:, None] + lam_w[None, :]
    # Avoid division by zero: with H, W >= 1, lam[0, 0] = 0 + 0 = 0 only if
    # both eigenvalues have a zero at index 0, which they do. But the
    # constant mode of f_hat is also zero (mean of f on a zero-boundary
    # interior is unconstrained by the Laplacian; we set it to 0 by
    # convention). Use 0/0 -> 0.
    with np.errstate(divide="ignore", invalid="warn"):
        u_hat = np.where(lam > 0, f_hat / lam, 0.0)
    u = np.fft.ifft2(u_hat).real
    return u.astype(np.float32)


def _sample_source_field(
    cfg: Heat2dConfig, rng: np.random.Generator
) -> np.ndarray:
    """Random Gaussian-sum source field of shape (H, W) on (0, 1)^2.

    Each sample is a sum of `n_bumps` Gaussians:
        f(x, y) = sum_k  A_k * exp(-((x-x_k)^2 + (y-y_k)^2) / (2 sigma_k^2))
    """
    n_bumps = int(rng.integers(cfg.n_bumps_range[0], cfg.n_bumps_range[1] + 1))
    h_grid = (np.arange(cfg.h) + 0.5) / (cfg.h + 1)  # (H,)
    w_grid = (np.arange(cfg.w) + 0.5) / (cfg.w + 1)  # (W,)
    H, W = np.meshgrid(h_grid, w_grid, indexing="ij")
    f = np.zeros((cfg.h, cfg.w), dtype=np.float64)
    for _ in range(n_bumps):
        A = rng.uniform(*cfg.bump_amplitude_range)
        sigma = rng.uniform(*cfg.bump_width_range)
        x0 = rng.uniform(sigma, 1.0 - sigma)
        y0 = rng.uniform(sigma, 1.0 - sigma)
        f += A * np.exp(-(((H - x0) ** 2 + (W - y0) ** 2) / (2.0 * sigma * sigma)))
    return f


def generate_heat2d_sample(
    cfg: Heat2dConfig, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single (x, y) sample.

    Args:
        cfg: dataset config.
        rng: numpy random generator (defaults to a fresh one seeded from cfg.seed).

    Returns:
        x: (H, W, 1) float32 source field.
        y: (H, W, 1) float32 solution field.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    f = _sample_source_field(cfg, rng)
    lam_h = _laplacian_eigenvalues(cfg.h)
    lam_w = _laplacian_eigenvalues(cfg.w)
    u = _solve_poisson_2d(f, lam_h, lam_w)
    return f[..., None].astype(np.float32), u[..., None].astype(np.float32)


class Heat2dDataset(Dataset):
    """In-memory 2D Poisson dataset.

    Each item is (x, y):
        x: (H, W, 1) source field.
        y: (H, W, 1) solution field.
    """

    def __init__(
        self,
        n_samples: int = 500,
        cfg: Heat2dConfig | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg or Heat2dConfig(seed=seed)
        self.n_samples = n_samples
        rng = np.random.default_rng(seed)
        xs = np.empty((n_samples, self.cfg.h, self.cfg.w, 1), dtype=np.float32)
        ys = np.empty_like(xs)
        for i in range(n_samples):
            x, y = generate_heat2d_sample(self.cfg, rng=rng)
            xs[i] = x
            ys[i] = y
        self._x = xs
        self._y = ys

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self._x[idx]), torch.from_numpy(self._y[idx])
