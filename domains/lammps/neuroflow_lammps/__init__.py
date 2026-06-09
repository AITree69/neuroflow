"""NeuroFlow × LAMMPS — Python SDK.

The SDK is intentionally small.  It exposes three things:

  1. `MorseSurrogate` — a thin wrapper around a NeuroFlow
     model (FNO1d in this Sprint) that exposes two methods:
     `energy(r)` and `force(r)`.  The energy is the model's
     forward pass; the force is computed via a 5-point
     central finite difference of the energy, evaluated by
     five forward passes.

  2. `SurrogateMDDriver` — a minimal MD driver loop that
     queries the surrogate instead of a classical pair
     potential, demonstrating the integration contract
     that a future LAMMPS `fix nflow` plugin will implement.

  3. `build_morse_dataset` — a small dataset builder that
     synthesises a 1D Morse potential across a parameter
     sweep, ready to be consumed by the example 17 driver.

The public API is documented in `docs()` at the bottom of
this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Morse potential — analytical reference
# ---------------------------------------------------------------------------


def morse_potential(r: np.ndarray, D_e: float, a: float, r_e: float) -> np.ndarray:
    """1D Morse potential $V(r) = D_e (1 - e^{-a (r - r_e)})^2 - D_e$."""
    return D_e * (1.0 - np.exp(-a * (r - r_e))) ** 2 - D_e


def morse_force(r: np.ndarray, D_e: float, a: float, r_e: float) -> np.ndarray:
    """Analytical force $F(r) = -\\mathrm{d}V/\\mathrm{d}r$."""
    return -2.0 * D_e * a * (1.0 - np.exp(-a * (r - r_e))) * np.exp(-a * (r - r_e))


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


@dataclass
class MorseConfig:
    n_points: int = 128
    n_samples: int = 400
    r_min: float = 0.5
    r_max: float = 4.0
    D_e_range: tuple[float, float] = (0.5, 2.0)
    a_range: tuple[float, float] = (1.0, 3.0)
    r_e_range: tuple[float, float] = (0.8, 1.5)
    seed: int = 0


def build_morse_dataset(cfg: MorseConfig | None = None):
    """Build a (x, y) dataset for the FNO1d surrogate.

    The dataset is parameter-conditioned: each sample's
    per-point input is the 4-vector `[r, D_e, a, r_e]`,
    with `(D_e, a, r_e)` broadcast at every r-grid point.
    This is the standard "global-parameter conditioning"
    pattern for an FNO surrogate of a parameterised
    potential: the FNO learns the family
    `V(r | D_e, a, r_e)` instead of memorising a single
    curve.

    Returns:
        x: (n_samples, n_points, 4) float32 — input, last
           three channels are the (D_e, a, r_e) used for
           this sample, broadcast at every point.
        y: (n_samples, n_points, 1) float32 — target V(r).
        r_grid: (n_points,) float32 — the r axis.
        params: (n_samples, 3) float32 — the (D_e, a, r_e)
           used for each sample (for diagnostics).
    """
    if cfg is None:
        cfg = MorseConfig()
    rng = np.random.default_rng(cfg.seed)
    r_grid = np.linspace(cfg.r_min, cfg.r_max, cfg.n_points,
                          dtype=np.float32)
    xs = np.empty((cfg.n_samples, cfg.n_points, 4), dtype=np.float32)
    ys = np.empty((cfg.n_samples, cfg.n_points, 1), dtype=np.float32)
    params = np.empty((cfg.n_samples, 3), dtype=np.float32)
    for i in range(cfg.n_samples):
        D_e = float(rng.uniform(*cfg.D_e_range))
        a = float(rng.uniform(*cfg.a_range))
        r_e = float(rng.uniform(*cfg.r_e_range))
        v = morse_potential(r_grid, D_e, a, r_e).astype(np.float32)
        # The first channel is the per-point r coordinate;
        # the remaining three channels are the global
        # Morse parameters broadcast at every r-grid
        # point.  This is what lets a single FNO1d
        # approximate the *family* of Morse curves.
        xs[i, :, 0] = r_grid
        xs[i, :, 1] = D_e
        xs[i, :, 2] = a
        xs[i, :, 3] = r_e
        ys[i, :, 0] = v
        params[i] = (D_e, a, r_e)
    return xs, ys, r_grid, params


# ---------------------------------------------------------------------------
# Surrogate wrapper
# ---------------------------------------------------------------------------


class MorseSurrogate:
    """A NeuroFlow-backed surrogate for the 1D Morse potential.

    The energy is the model's forward pass; the force is a
    5-point central finite difference (four extra forward
    passes per query, evaluated in PyTorch).  The class is
    deliberately small — the goal is to demonstrate the
    integration contract, not to compete with autograd-based
    force evaluation.
    """

    def __init__(self, model: torch.nn.Module, r_min: float, r_max: float,
                 fd_step: float = 1e-3) -> None:
        self.model = model
        self.r_min = r_min
        self.r_max = r_max
        self.fd_step = fd_step
        self.model.eval()

    @torch.no_grad()
    def energy(self, r: torch.Tensor) -> torch.Tensor:
        # The model expects (batch, n_points, in_dim).
        if r.ndim == 1:
            r = r.unsqueeze(-1)  # (n_points, 1)
        if r.ndim == 2:
            r = r.unsqueeze(0)   # (1, n_points, 1)
        out = self.model(r)
        return out.squeeze(-1)  # (batch, n_points)

    @torch.no_grad()
    def force(self, r: torch.Tensor) -> torch.Tensor:
        # 5-point central FD: F(r) = (V(r - 2h) - 8 V(r - h)
        # + 8 V(r + h) - V(r + 2h)) / (12 h).
        h = self.fd_step
        r_minus2h = r - 2.0 * h
        r_minus_h = r - h
        r_plus_h = r + h
        r_plus2h = r + 2.0 * h
        v_minus2h = self.energy(r_minus2h)
        v_minus_h = self.energy(r_minus_h)
        v_plus_h = self.energy(r_plus_h)
        v_plus2h = self.energy(r_plus2h)
        return (-v_minus2h + 8.0 * v_minus_h - 8.0 * v_plus_h + v_plus2h) / (12.0 * h)

    def energy_force(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.energy(r), self.force(r)


# ---------------------------------------------------------------------------
# Minimal MD driver loop
# ---------------------------------------------------------------------------


def surrogate_md_loop(surrogate: MorseSurrogate,
                      r_grid: np.ndarray,
                      n_steps: int = 32,
                      dt: float = 0.01,
                      r_init: float = 1.0) -> np.ndarray:
    """A toy MD loop where the force is the surrogate's
    numerical force.  Velocity Verlet integration on a
    single particle in 1D; the only "interaction" is with the
    effective 1D potential represented by the surrogate.

    The output is a (n_steps,) array of positions; the loop
    illustrates the integration contract that a future
    LAMMPS `fix nflow` would implement (call the surrogate
    every step, return the force to the integrator).
    """
    r = float(r_init)
    v = 0.0
    a = 0.0
    traj = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        f = float(surrogate.force(torch.tensor([[r]], dtype=torch.float32)
                                    ).squeeze().item())
        # velocity Verlet (single particle, mass = 1)
        r_new = r + v * dt + 0.5 * a * dt * dt
        a_new = -f
        v_new = v + 0.5 * (a + a_new) * dt
        r, v, a = r_new, v_new, a_new
        traj[t] = r
    return traj


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


def docs() -> str:
    return (
        "neuroflow_lammps SDK — public API:\n"
        "  - build_morse_dataset(cfg)        -> (x, y, r_grid, params)\n"
        "  - morse_potential(r, D_e, a, r_e) -> V(r)\n"
        "  - morse_force(r, D_e, a, r_e)     -> F(r)\n"
        "  - MorseSurrogate(model, ...).energy(r)\n"
        "  - MorseSurrogate(model, ...).force(r)\n"
        "  - surrogate_md_loop(surrogate, r_grid, n_steps, dt)\n"
    )
