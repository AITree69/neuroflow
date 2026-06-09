"""Sprint 3.8 — pytest cases for the NeuroFlow × LAMMPS domain SDK.

These tests cover:
  * `morse_potential` / `morse_force` — analytical reference
  * `build_morse_dataset` — shape + parameter coverage
  * `MorseSurrogate` — energy + FD-force, sanity checks
  * `surrogate_md_loop` — toy MD driver returns a sane trajectory
  * C++ runtime parity (skipped if the binary is unavailable)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from domains.lammps.neuroflow_lammps import (
    MorseConfig,
    MorseSurrogate,
    build_morse_dataset,
    morse_force,
    morse_potential,
    surrogate_md_loop,
)
from neuroflow.nn.fno import FNO1d, FNO1dConfig


def _make_dummy_fno1d(n_points: int = 64) -> FNO1d:
    cfg = FNO1dConfig(
        in_channels=1,
        out_channels=1,
        width=16,
        modes=min(8, n_points // 2),
        n_layers=2,
        activation="gelu",
        pad_factor=1,
        name="dummy_fno1d_morse",
    )
    return FNO1d(cfg)


def test_morse_potential_minimum() -> None:
    D_e, a, r_e = 1.5, 2.0, 1.1
    # V(r_e) = -D_e exactly; evaluating at the analytical
    # minimum avoids linspace quantisation error.
    v_at_min = morse_potential(np.array([r_e], dtype=np.float32),
                                 D_e, a, r_e)
    assert float(v_at_min[0]) == pytest.approx(-D_e, abs=1e-5)
    # The global minimum on a fine grid is in the well, not
    # at the endpoints, and reaches within float32 round-off
    # of -D_e.
    r = np.linspace(0.5, 4.0, 4000)
    v = morse_potential(r, D_e, a, r_e)
    idx_min = int(np.argmin(v))
    assert 0 < idx_min < len(r) - 1
    assert float(v[idx_min]) == pytest.approx(-D_e, abs=1e-3)


def test_morse_force_at_minimum_zero() -> None:
    D_e, a, r_e = 1.5, 2.0, 1.1
    # Force evaluated at the analytical minimum r_e is
    # exactly zero (the Morse potential is differentiable
    # there and the analytical derivative is closed form).
    f_at_min = morse_force(np.array([r_e], dtype=np.float32),
                            D_e, a, r_e)
    assert float(f_at_min[0]) == pytest.approx(0.0, abs=1e-5)
    # Force has a sign change at r_e (attractive for r>r_e,
    # repulsive for r<r_e).
    r = np.linspace(0.6, 2.5, 200)
    f = morse_force(r, D_e, a, r_e)
    assert f[0] > 0     # r<r_e: force pushes away (towards larger r)
    assert f[-1] < 0    # r>r_e: force pulls back (towards smaller r)


def test_build_morse_dataset_shapes() -> None:
    cfg = MorseConfig(n_points=64, n_samples=10, seed=0)
    x, y, r_grid, params = build_morse_dataset(cfg)
    # Parameter-conditioned dataset: x has 4 channels
    # (r, D_e, a, r_e) per point.
    assert x.shape == (10, 64, 4)
    assert y.shape == (10, 64, 1)
    assert r_grid.shape == (64,)
    assert params.shape == (10, 3)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_build_morse_dataset_conditioning_broadcast() -> None:
    cfg = MorseConfig(n_points=32, n_samples=4, seed=0,
                       D_e_range=(0.7, 0.7), a_range=(1.2, 1.2),
                       r_e_range=(0.9, 0.9))
    x, y, r_grid, params = build_morse_dataset(cfg)
    # The last three input channels are the per-sample
    # (D_e, a, r_e) broadcast at every r-grid point.
    for i in range(4):
        D_e, a, r_e = params[i]
        assert (x[i, :, 1] == D_e).all()
        assert (x[i, :, 2] == a).all()
        assert (x[i, :, 3] == r_e).all()
        assert (x[i, :, 0] == r_grid).all()
        # Energy matches analytical Morse potential.
        v_ana = morse_potential(r_grid, D_e, a, r_e).astype(np.float32)
        assert np.allclose(y[i, :, 0], v_ana, atol=1e-5)


def test_surrogate_energy_shape() -> None:
    torch.manual_seed(0)
    n_points = 64
    model = _make_dummy_fno1d(n_points=n_points)
    surrogate = MorseSurrogate(model, r_min=0.5, r_max=4.0)
    r = torch.linspace(0.5, 4.0, n_points).unsqueeze(-1).unsqueeze(0)
    e = surrogate.energy(r)
    assert e.shape == (1, n_points)
    assert torch.isfinite(e).all().item()


def test_surrogate_force_shape_and_finite() -> None:
    torch.manual_seed(0)
    n_points = 64
    model = _make_dummy_fno1d(n_points=n_points)
    surrogate = MorseSurrogate(model, r_min=0.5, r_max=4.0, fd_step=1e-2)
    r = torch.linspace(0.5, 4.0, n_points).unsqueeze(-1).unsqueeze(0)
    f = surrogate.force(r)
    assert f.shape == (1, n_points)
    assert torch.isfinite(f).all().item()


def test_surrogate_md_loop_runs() -> None:
    torch.manual_seed(0)
    n_points = 64
    model = _make_dummy_fno1d(n_points=n_points)
    surrogate = MorseSurrogate(model, r_min=0.5, r_max=4.0, fd_step=1e-2)
    r_grid = np.linspace(0.5, 4.0, n_points)
    traj = surrogate_md_loop(surrogate, r_grid, n_steps=8, dt=1e-2, r_init=1.0)
    assert traj.shape == (8,)
    assert np.isfinite(traj).all()


def test_cpp_runtime_parity_skipped_if_unavailable() -> None:
    """If the C++ runtime is built and the demo IR exists, the
    energy parity on a held-out test sample should be < 1e-3.
    Otherwise this test is skipped (so the test suite still
    passes on a fresh clone without a C++ build).
    """
    try:
        import neuroflow_cpp  # noqa: F401
    except ImportError:
        pytest.skip("neuroflow_cpp not built; skipping C++ parity test")

    ir_path = (_PROJECT_ROOT / "artifacts" / "lammps_demo" / "ir"
                / "morse_surrogate.nfir")
    if not ir_path.exists():
        pytest.skip(
            "Morse surrogate IR not generated yet; "
            "run examples/17_morse_surrogate.py first."
        )

    cfg = MorseConfig(n_points=64, n_samples=1, seed=1234,
                       D_e_range=(1.0, 1.0), a_range=(1.5, 1.5),
                       r_e_range=(1.0, 1.0))
    x, _, _, _ = build_morse_dataset(cfg)
    import neuroflow_cpp
    y_cpp = neuroflow_cpp.infer_arrays(str(ir_path), x.astype("float32"))
    # Build a quick reference with the same dummy model.
    torch.manual_seed(1234)
    model = _make_dummy_fno1d(n_points=64)
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x)).numpy()
    diff = float(np.abs(y_torch - y_cpp).max())
    # The dummy model + C++ parity target for this short test.
    assert diff < 1e-3
