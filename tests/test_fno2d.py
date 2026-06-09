"""Tests for the FNO2d model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuroflow.ir.export import export_to_binary, export_to_neuroir
from neuroflow.ir.load import load_neuroir, predict_with_spec
from neuroflow.nn.fno2d import FNO2d, FNO2dConfig, SpectralConv2d


def test_fno2d_forward_shape():
    cfg = FNO2dConfig(in_channels=2, out_channels=3, width=32, modes_h=8, modes_w=8, n_layers=3)
    model = FNO2d(cfg)
    x = torch.randn(4, 64, 32, 2)  # (batch, h, w, in_ch)
    y = model(x)
    assert y.shape == (4, 64, 32, 3)


def test_fno2d_num_parameters():
    cfg = FNO2dConfig(in_channels=1, out_channels=1, width=16, modes_h=4, modes_w=4, n_layers=2)
    model = FNO2d(cfg)
    assert model.num_parameters() > 0


def test_spectral_conv2d_shapes():
    conv = SpectralConv2d(in_channels=8, out_channels=8, modes_h=4, modes_w=4)
    x = torch.randn(2, 8, 32, 32)  # (b, c, h, w)
    y = conv(x)
    assert y.shape == x.shape
    # Weight shape mirrors FNO paper convention: (in, out, modes_h, modes_w).
    assert conv.weights_real.shape == (8, 8, 4, 4)
    assert conv.weights_imag.shape == (8, 8, 4, 4)


def test_fno2d_torch_matches_numpy_ir():
    """PyTorch forward matches the predict_with_spec forward within float32 noise."""
    torch.manual_seed(0)
    cfg = FNO2dConfig(
        in_channels=2,
        out_channels=1,
        width=16,
        modes_h=4,
        modes_w=4,
        n_layers=2,
        pad_factor=1,
    )
    model = FNO2d(cfg).eval()
    x_np = np.random.default_rng(0).standard_normal((2, 16, 16, 2)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    with torch.no_grad():
        y_pt = model(x_pt).numpy()

    spec = export_to_neuroir(model)
    y_py = predict_with_spec(spec, x_np)
    max_abs = np.abs(y_pt - y_py).max()
    # Float32, no GEMM, should be near-exact.
    assert max_abs < 1e-4, f"max abs diff = {max_abs}"


def test_fno2d_ir_roundtrip_via_load():
    """load_neuroir(spec) reconstructs the model exactly (same weights)."""
    torch.manual_seed(0)
    cfg = FNO2dConfig(in_channels=1, out_channels=1, width=8, modes_h=2, modes_w=2, n_layers=1)
    model = FNO2d(cfg).eval()
    x = torch.randn(1, 8, 8, 1)
    with torch.no_grad():
        y0 = model(x).numpy()

    spec = export_to_neuroir(model)
    assert spec.op == "FNO2d"
    assert spec.config["modes_h"] == 2
    assert spec.config["modes_w"] == 2

    # Write to a tmp path and re-read.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fno2d.neuroir"
        spec.save(path)
        loaded = load_neuroir(str(path))
    with torch.no_grad():
        y1 = loaded(x).numpy()

    np.testing.assert_allclose(y0, y1, atol=1e-5, rtol=1e-5)


def test_fno2d_binary_roundtrip():
    """export_to_binary produces a NIR0 file with the correct op_code and config."""
    torch.manual_seed(0)
    cfg = FNO2dConfig(in_channels=1, out_channels=1, width=8, modes_h=3, modes_w=4, n_layers=2)
    model = FNO2d(cfg)
    spec = export_to_neuroir(model)
    blob = export_to_binary(spec)
    assert blob[:4] == b"NIR0"
    # version=2 (u16) then op_code=0x02 (u8) then reserved=0 (u8)
    import struct
    version, op_code, _ = struct.unpack("<HBB", blob[4:8])
    assert version == 2
    assert op_code == 0x02  # FNO2d
    # Config block: 7 int32, slots 3 and 4 are modes_h, modes_w
    cfg_block = struct.unpack("<7i", blob[8:36])
    in_ch, out_ch, width, modes_h, modes_w, n_layers, pad_factor = cfg_block
    assert (in_ch, out_ch, width, modes_h, modes_w, n_layers, pad_factor) == (1, 1, 8, 3, 4, 2, 1)
