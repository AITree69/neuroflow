"""Plotting helpers (matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:  # pragma: no cover
    _HAS_PLT = False


def plot_loss_curve(
    epochs: Sequence[int],
    train: Sequence[float],
    val: Sequence[float] | None = None,
    save_path: str | Path | None = None,
) -> None:
    if not _HAS_PLT:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(epochs, train, label="train", marker="o", markersize=3)
    if val is not None and not all(np.isnan(val)):
        ax.semilogy(epochs, val, label="val", marker="s", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_burgers_prediction(
    x_grid: np.ndarray,
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    title: str = "Burgers 1D — true vs predicted",
    save_path: str | Path | None = None,
) -> None:
    """Plot the final-state comparison.

    Args:
        x_grid: (n,) spatial grid.
        true_traj: (T, n) ground truth.
        pred_traj: (T, n) prediction (must match true_traj shape).
    """
    if not _HAS_PLT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_steps = true_traj.shape[0]
    n_points = true_traj.shape[1]
    # pcolormesh needs C of shape (len(Y), len(X)) when shading='auto'
    # We plot time on x-axis, x on y-axis; transpose the trajectory to (n_points, n_steps).
    t_grid = np.arange(n_steps)
    cmappable = axes[0].pcolormesh(
        t_grid, x_grid, true_traj.T, cmap="viridis", shading="auto"
    )
    fig.colorbar(cmappable, ax=axes[0])
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("x")
    axes[0].set_title("ground truth")
    axes[0].set_ylim(x_grid[0], x_grid[-1])

    cmappable2 = axes[1].pcolormesh(
        t_grid, x_grid, pred_traj.T, cmap="viridis", shading="auto"
    )
    fig.colorbar(cmappable2, ax=axes[1])
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("x")
    axes[1].set_title("FNO prediction")
    axes[1].set_ylim(x_grid[0], x_grid[-1])
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_2d_field_comparison(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    title: str = "2D field — true vs predicted",
    save_path: str | Path | None = None,
) -> None:
    """Plot a 2D field comparison (true / pred / abs error).

    Args:
        true_field: (H, W) ground truth.
        pred_field: (H, W) prediction.
        title: plot title.
        save_path: output path; if None, does not save.
    """
    if not _HAS_PLT:
        return
    if true_field.shape != pred_field.shape:
        raise ValueError(
            f"shape mismatch: true {true_field.shape} vs pred {pred_field.shape}"
        )
    err = np.abs(true_field - pred_field)

    vmax = max(np.abs(true_field).max(), np.abs(pred_field).max(), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, data, label, v in (
        (axes[0], true_field, "ground truth", vmax),
        (axes[1], pred_field, "FNO prediction", vmax),
        (axes[2], err, "abs error", err.max()),
    ):
        im = ax.imshow(data, cmap="viridis", origin="lower", vmin=0.0, vmax=v)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label)
        ax.set_xlabel("w")
        ax.set_ylabel("h")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
