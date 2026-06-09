"""Tests for the DeepONet model + IR roundtrip."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from neuroflow.ir.export import export_to_binary, export_to_neuroir
from neuroflow.ir.load import load_neuroir, predict_with_spec
from neuroflow.nn.deeponet import BranchNet, DeepONet, DeepONetConfig, TrunkNet


def test_deeponet_forward_shape():
    cfg = DeepONetConfig(
        in_branch=2, in_trunk=2, latent_dim=8, out_channels=3,
        hidden_branch=12, hidden_trunk=12,
        n_layers_branch=2, n_layers_trunk=2,
    )
    model = DeepONet(cfg)
    u = torch.randn(4, 16, 2)
    y = torch.randn(4, 10, 2)
    out = model(u, y)
    assert out.shape == (4, 10, 3)


def test_deeponet_num_parameters():
    cfg = DeepONetConfig(
        in_branch=1, in_trunk=1, latent_dim=4, out_channels=1,
        hidden_branch=8, hidden_trunk=8,
        n_layers_branch=2, n_layers_trunk=2,
    )
    model = DeepONet(cfg)
    assert model.num_parameters() > 0


def test_branch_and_trunk_shapes():
    cfg = DeepONetConfig(
        in_branch=1, in_trunk=1, latent_dim=4, out_channels=2,
        hidden_branch=8, hidden_trunk=8,
        n_layers_branch=2, n_layers_trunk=2,
    )
    model = DeepONet(cfg).eval()
    u = torch.randn(2, 7, 1)
    y = torch.randn(2, 5, 1)
    # Branch: (b, n_sensor, in_branch) -> (b, out_ch, latent_dim)
    b = model.branch(u)
    assert b.shape == (2, 2, 4)
    # Trunk: (b, n_query, in_trunk) -> (b, n_query, latent_dim)
    t = model.trunk(y)
    assert t.shape == (2, 5, 4)


def test_deeponet_torch_matches_numpy_ir():
    """PyTorch forward matches the predict_with_spec forward within float32 noise."""
    torch.manual_seed(0)
    cfg = DeepONetConfig(
        in_branch=1, in_trunk=1, latent_dim=4, out_channels=1,
        hidden_branch=8, hidden_trunk=8,
        n_layers_branch=2, n_layers_trunk=2,
    )
    model = DeepONet(cfg).eval()
    u_np = np.random.default_rng(0).standard_normal((1, 8, 1)).astype(np.float32)
    y_np = np.linspace(0.0, 1.0, 6, endpoint=False, dtype=np.float32).reshape(1, 6, 1)
    u_pt = torch.from_numpy(u_np)
    y_pt = torch.from_numpy(y_np)
    with torch.no_grad():
        out_pt = model(u_pt, y_pt).numpy()
    spec = export_to_neuroir(model)
    out_py = predict_with_spec(spec, u_np, y_np)
    max_abs = np.abs(out_pt - out_py).max()
    # Float32, no GEMM, should be near-exact.
    assert max_abs < 1e-4, f"max abs diff = {max_abs}"


def test_deeponet_ir_roundtrip_via_load():
    """load_neuroir(spec) reconstructs the model exactly (same weights)."""
    torch.manual_seed(0)
    cfg = DeepONetConfig(
        in_branch=1, in_trunk=1, latent_dim=4, out_channels=1,
        hidden_branch=8, hidden_trunk=8,
        n_layers_branch=2, n_layers_trunk=2,
    )
    model = DeepONet(cfg).eval()
    u = torch.randn(1, 8, 1)
    y = torch.randn(1, 6, 1)
    with torch.no_grad():
        out0 = model(u, y).numpy()

    spec = export_to_neuroir(model)
    assert spec.op == "DeepONet"
    assert spec.config["latent_dim"] == 4
    assert spec.config["n_layers_branch"] == 2
    assert spec.config["n_layers_trunk"] == 2

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deeponet.neuroir"
        spec.save(path)
        loaded = load_neuroir(str(path))
    with torch.no_grad():
        out1 = loaded(u, y).numpy()

    np.testing.assert_allclose(out0, out1, atol=1e-5, rtol=1e-5)


def test_deeponet_binary_roundtrip():
    """DeepONet writes version=3 + op_code=0x04 + 7 int32 config block."""
    torch.manual_seed(0)
    cfg = DeepONetConfig(
        in_branch=2, in_trunk=1, latent_dim=6, out_channels=3,
        hidden_branch=10, hidden_trunk=10,
        n_layers_branch=2, n_layers_trunk=3,
    )
    model = DeepONet(cfg)
    spec = export_to_neuroir(model)
    blob = export_to_binary(spec)
    assert blob[:4] == b"NIR0"
    # version (u16) = 3, op_code (u8) = 0x04, reserved (u8) = 0
    version, op_code, _ = struct.unpack("<HBB", blob[4:8])
    assert version == 3
    assert op_code == 0x04
    # Config block: 7 int32 = [in_branch, in_trunk, latent_dim,
    # out_channels, n_layers_branch, n_layers_trunk, _]
    cfg_block = struct.unpack("<7i", blob[8:36])
    in_branch, in_trunk, latent_dim, out_channels, nlb, nlt, _ = cfg_block
    assert (in_branch, in_trunk, latent_dim, out_channels, nlb, nlt) == (2, 1, 6, 3, 2, 3)
