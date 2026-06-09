"""Tests for the FNO3d model."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from neuroflow.ir.export import export_to_binary, export_to_neuroir
from neuroflow.ir.load import load_neuroir, predict_with_spec
from neuroflow.nn.fno3d import FNO3d, FNO3dConfig, SpectralConv3d


def test_fno3d_forward_shape():
    cfg = FNO3dConfig(
        in_channels=2, out_channels=3, width=16,
        modes_h=4, modes_w=4, modes_d=4, n_layers=2,
    )
    model = FNO3d(cfg)
    x = torch.randn(2, 8, 8, 8, 2)
    y = model(x)
    assert y.shape == (2, 8, 8, 8, 3)


def test_fno3d_num_parameters():
    cfg = FNO3dConfig(
        in_channels=1, out_channels=1, width=8,
        modes_h=2, modes_w=2, modes_d=2, n_layers=1,
    )
    model = FNO3d(cfg)
    assert model.num_parameters() > 0


def test_spectral_conv3d_shapes():
    conv = SpectralConv3d(in_channels=4, out_channels=4, modes_h=2, modes_w=2, modes_d=2)
    x = torch.randn(1, 4, 8, 8, 8)
    y = conv(x)
    assert y.shape == x.shape
    # FNO paper convention: (in, out, modes_h, modes_w, modes_d).
    assert conv.weights_real.shape == (4, 4, 2, 2, 2)
    assert conv.weights_imag.shape == (4, 4, 2, 2, 2)


def test_fno3d_torch_matches_numpy_ir():
    """PyTorch forward matches the predict_with_spec forward within float32 noise."""
    torch.manual_seed(0)
    cfg = FNO3dConfig(
        in_channels=1, out_channels=1, width=8,
        modes_h=2, modes_w=2, modes_d=2, n_layers=1, pad_factor=1,
    )
    model = FNO3d(cfg).eval()
    x_np = np.random.default_rng(0).standard_normal((1, 8, 8, 8, 1)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        y_pt = model(x_pt).numpy()
    spec = export_to_neuroir(model)
    y_py = predict_with_spec(spec, x_np)
    max_abs = np.abs(y_pt - y_py).max()
    # Float32, no GEMM, should be near-exact.
    assert max_abs < 1e-4, f"max abs diff = {max_abs}"


def test_fno3d_ir_roundtrip_via_load():
    """load_neuroir(spec) reconstructs the model exactly (same weights)."""
    torch.manual_seed(0)
    cfg = FNO3dConfig(
        in_channels=1, out_channels=1, width=8,
        modes_h=2, modes_w=2, modes_d=2, n_layers=1,
    )
    model = FNO3d(cfg).eval()
    x = torch.randn(1, 8, 8, 8, 1)
    with torch.no_grad():
        y0 = model(x).numpy()

    spec = export_to_neuroir(model)
    assert spec.op == "FNO3d"
    assert spec.config["modes_h"] == 2
    assert spec.config["modes_w"] == 2
    assert spec.config["modes_d"] == 2

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fno3d.neuroir"
        spec.save(path)
        loaded = load_neuroir(str(path))
    with torch.no_grad():
        y1 = loaded(x).numpy()

    np.testing.assert_allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_fno3d_binary_roundtrip():
    """FNO3d writes version=3 + op_code=0x03 + 8 int32 config block."""
    torch.manual_seed(0)
    cfg = FNO3dConfig(
        in_channels=1, out_channels=1, width=8,
        modes_h=2, modes_w=3, modes_d=4, n_layers=2,
    )
    model = FNO3d(cfg)
    spec = export_to_neuroir(model)
    blob = export_to_binary(spec)
    assert blob[:4] == b"NIR0"
    # version (u16) = 3, op_code (u8) = 0x03, reserved (u8) = 0
    version, op_code, _ = struct.unpack("<HBB", blob[4:8])
    assert version == 3, f"expected binary version 3, got {version}"
    assert op_code == 0x03, f"expected op_code 0x03 (FNO3d), got {op_code:#x}"
    # Config block: 8 int32 = [in_ch, out_ch, width, modes_h, modes_w, modes_d, n_layers, pad_factor]
    cfg_block = struct.unpack("<8i", blob[8:40])
    in_ch, out_ch, width, modes_h, modes_w, modes_d, n_layers, pad_factor = cfg_block
    assert (in_ch, out_ch, width, modes_h, modes_w, modes_d, n_layers, pad_factor) == (
        1, 1, 8, 2, 3, 4, 2, 1,
    )


def test_legacy_fno2d_unchanged_in_v0_3_writer():
    """The v0.3.0 writer preserves the v0.2.0 on-disk layout for FNO1d / FNO2d."""
    from neuroflow.nn.fno2d import FNO2d, FNO2dConfig

    torch.manual_seed(0)
    cfg = FNO2dConfig(
        in_channels=1, out_channels=1, width=8,
        modes_h=2, modes_w=2, n_layers=1,
    )
    model = FNO2d(cfg)
    spec = export_to_neuroir(model)
    blob = export_to_binary(spec)
    # FNO2d should still be version=2 + 7 int32 (backward compatible).
    version, op_code, _ = struct.unpack("<HBB", blob[4:8])
    assert version == 2
    assert op_code == 0x02
    cfg_block = struct.unpack("<7i", blob[8:36])
    assert len(cfg_block) == 7
    in_ch, out_ch, width, modes_h, modes_w, n_layers, pad_factor = cfg_block
    assert (in_ch, out_ch, width, modes_h, modes_w, n_layers, pad_factor) == (1, 1, 8, 2, 2, 1, 1)
