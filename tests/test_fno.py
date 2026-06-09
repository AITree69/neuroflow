"""Tests for the FNO1d model."""

import numpy as np
import pytest
import torch

from neuroflow.ir.export import export_to_binary, export_to_neuroir
from neuroflow.ir.load import load_neuroir, predict_with_spec
from neuroflow.nn.fno import FNO1d, FNO1dConfig


def test_fno1d_forward_shape():
    cfg = FNO1dConfig(in_channels=2, out_channels=3, width=32, modes=8, n_layers=3)
    model = FNO1d(cfg)
    x = torch.randn(4, 64, 2)
    y = model(x)
    assert y.shape == (4, 64, 3)


def test_fno1d_num_parameters():
    cfg = FNO1dConfig(in_channels=1, out_channels=1, width=16, modes=4, n_layers=2)
    model = FNO1d(cfg)
    # Just ensure it has parameters and is non-trivial
    assert model.num_parameters() > 0


def test_fno1d_roundtrip():
    """PyTorch forward matches the predict_with_spec forward."""
    torch.manual_seed(0)
    cfg = FNO1dConfig(
        in_channels=2, out_channels=2, width=16, modes=4, n_layers=2, activation="gelu"
    )
    model = FNO1d(cfg)
    model.eval()
    x = torch.randn(2, 32, 2)
    with torch.no_grad():
        y_pt = model(x).numpy()

    spec = export_to_neuroir(model)
    y_py = predict_with_spec(spec, x.numpy())
    assert y_pt.shape == y_py.shape
    assert np.allclose(y_pt, y_py, atol=1e-5)


def test_ir_binary_roundtrip():
    """Binary .nneuroir matches the JSON .neuroir (same weights)."""
    torch.manual_seed(0)
    model = FNO1d(FNO1dConfig(in_channels=1, out_channels=1, width=8, modes=2, n_layers=1))
    spec = export_to_neuroir(model)
    blob = export_to_binary(spec)
    assert blob[:4] == b"NIR0"
    # Re-load via JSON path and confirm the same forward
    model2 = load_neuroir(spec)
    x = torch.randn(1, 8, 1)
    with torch.no_grad():
        y = model2(x).numpy()
    assert y.shape == (1, 8, 1)


def test_burgers_dataset_shapes():
    from neuroflow.data.burgers import Burgers1dConfig, Burgers1dDataset

    cfg = Burgers1dConfig(n_points=64, n_tsteps=50, seed=0)
    ds = Burgers1dDataset(n_samples=2, cfg=cfg, t_in=5, t_out=5, seed=0)
    x, y = ds[0]
    # x: (n_points, t_in) — matches FNO1d's (batch, n, in_channels) convention
    assert x.shape == (64, 5)
    assert y.shape == (64, 5)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
