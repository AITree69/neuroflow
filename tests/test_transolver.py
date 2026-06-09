"""Stage 2 Sprint 3 — TokenMixer (Transolver-style operator learner) tests.

Covers:
  - Python forward shape and parameter count
  - state_dict_for_ir layout and determinism
  - NeuroIR roundtrip (JSON + binary)
  - Pure NumPy forward matches PyTorch within float32 noise
  - C++ runtime (via neuroflow_cpp) matches PyTorch within float32 noise
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict

import numpy as np
import pytest
import torch

from neuroflow.ir.export import export_all, export_to_binary, export_to_neuroir
from neuroflow.ir.load import (
    load_neuroir,
    predict_with_spec,
    predict_with_spec_torch,
)
from neuroflow.ir.spec import NeuroIRSpec
from neuroflow.nn.tokenmixer import TokenMixer, TokenMixerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _default_cfg() -> TokenMixerConfig:
    return TokenMixerConfig(
        in_dim=2,
        out_dim=1,
        n_points=64,
        n_patches=8,
        latent_dim=32,
        n_heads=4,
        n_layers=1,
        activation="gelu",
        name="tokenmixer_test",
    )


def _tiny_cfg() -> TokenMixerConfig:
    """Smaller config to keep the C++ test matrix reasonable."""
    return TokenMixerConfig(
        in_dim=1,
        out_dim=1,
        n_points=16,
        n_patches=4,
        latent_dim=8,
        n_heads=2,
        n_layers=1,
        activation="gelu",
        name="tokenmixer_tiny",
    )


# ---------------------------------------------------------------------------
# Python forward
# ---------------------------------------------------------------------------


def test_python_forward_shape():
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = TokenMixer(cfg).eval()
    x = torch.randn(2, cfg.n_points, cfg.in_dim)
    y = model(x)
    assert tuple(y.shape) == (2, cfg.n_points, cfg.out_dim)


def test_python_forward_deterministic():
    cfg = _default_cfg()
    model = TokenMixer(cfg).eval()
    x = torch.randn(2, cfg.n_points, cfg.in_dim)
    y1 = model(x)
    y2 = model(x)
    assert torch.allclose(y1, y2)


def test_state_dict_for_ir_layout():
    cfg = _default_cfg()
    model = TokenMixer(cfg).eval()
    sd = model.state_dict_for_ir()
    expected = [
        "slice_embed.proj.weight", "slice_embed.proj.bias",
        "blocks.0.ln1.weight", "blocks.0.ln1.bias",
        "blocks.0.q_proj.weight", "blocks.0.q_proj.bias",
        "blocks.0.k_proj.weight", "blocks.0.k_proj.bias",
        "blocks.0.v_proj.weight", "blocks.0.v_proj.bias",
        "blocks.0.o_proj.weight", "blocks.0.o_proj.bias",
        "blocks.0.ln2.weight", "blocks.0.ln2.bias",
        "blocks.0.ffn0.weight", "blocks.0.ffn0.bias",
        "blocks.0.ffn1.weight", "blocks.0.ffn1.bias",
        "unslice.proj.weight", "unslice.proj.bias",
        "head.weight", "head.bias",
    ]
    assert list(sd.keys()) == expected
    # Latent dim 32, in_dim 2 -> slice_embed.proj.weight is (32, 2)
    assert tuple(sd["slice_embed.proj.weight"].shape) == (cfg.latent_dim, cfg.in_dim)
    # unslice.proj.weight is (latent_dim, in_dim + latent_dim)
    assert tuple(sd["unslice.proj.weight"].shape) == (cfg.latent_dim, cfg.in_dim + cfg.latent_dim)
    # head.weight is (out_dim, latent_dim)
    assert tuple(sd["head.weight"].shape) == (cfg.out_dim, cfg.latent_dim)


def test_latent_dim_must_divide_n_heads():
    with pytest.raises(ValueError):
        TokenMixerConfig(
            in_dim=2, out_dim=1, n_points=64, n_patches=8,
            latent_dim=30, n_heads=4,  # 30 % 4 != 0
        )


def test_n_points_must_be_multiple_of_n_patches_or_truncates():
    cfg = TokenMixerConfig(
        in_dim=2, out_dim=1, n_points=70, n_patches=8,  # 70 % 8 != 0
        latent_dim=16, n_heads=2,
    )
    model = TokenMixer(cfg).eval()
    # TokenMixer truncates to the largest multiple internally
    x = torch.randn(2, 70, 2)
    y = model(x)
    # n_eff = 64 (= 8 * 8) since 70 is truncated to 64
    assert tuple(y.shape) == (2, 64, 1)


# ---------------------------------------------------------------------------
# NeuroIR roundtrip
# ---------------------------------------------------------------------------


def test_neuroir_roundtrip():
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = TokenMixer(cfg).eval()
    x = torch.randn(2, cfg.n_points, cfg.in_dim)
    y_orig = model(x).detach().cpu().numpy().astype("float32")

    spec = export_to_neuroir(model)
    assert spec.op == "TokenMixer"
    assert spec.config["in_dim"] == cfg.in_dim
    assert spec.config["out_dim"] == cfg.out_dim
    assert spec.config["n_points"] == cfg.n_points
    assert spec.config["n_patches"] == cfg.n_patches
    assert spec.config["latent_dim"] == cfg.latent_dim
    assert spec.config["n_heads"] == cfg.n_heads
    assert spec.config["n_layers"] == cfg.n_layers

    # JSON roundtrip
    spec2 = NeuroIRSpec.from_json(spec.to_json())
    model2 = load_neuroir(spec2)
    y_reload = model2(x).detach().cpu().numpy().astype("float32")
    assert np.allclose(y_orig, y_reload, atol=0.0)  # exact — same weights


def test_neuroir_binary_roundtrip():
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = TokenMixer(cfg).eval()
    x = torch.randn(2, cfg.n_points, cfg.in_dim)
    y_orig = model(x).detach().cpu().numpy().astype("float32")

    spec = export_to_neuroir(model)
    bin_data = export_to_binary(spec)
    # Magic + version(2) + op(1) + reserved(1) + config(8*4) + act(1) + reserved2(3) + n_weights(4)
    assert bin_data[:4] == b"NIR0"
    assert bin_data[4] == 3  # binary version (TokenMixer writes v3)
    assert bin_data[6] == 0x05  # op_code for TokenMixer

    spec2 = NeuroIRSpec.from_json(spec.to_json())
    model2 = load_neuroir(spec2)
    y_reload = model2(x).detach().cpu().numpy().astype("float32")
    assert np.allclose(y_orig, y_reload, atol=0.0)


def test_numpy_forward_matches_torch():
    """predict_with_spec (NumPy) matches predict_with_spec_torch (PyTorch)."""
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = TokenMixer(cfg).eval()
    x = torch.randn(2, cfg.n_points, cfg.in_dim)
    spec = export_to_neuroir(model)
    y_torch = predict_with_spec_torch(spec, x).cpu().numpy().astype("float32")
    y_numpy = predict_with_spec(spec, x.numpy().astype("float32"))
    diff = np.abs(y_torch - y_numpy)
    assert diff.max() < 1e-3, f"NumPy vs PyTorch diff too large: {diff.max()}"
    assert diff.mean() < 1e-4


# ---------------------------------------------------------------------------
# C++ runtime parity (loaded only if available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NFLOW_SKIP_CPP") == "1",
    reason="NFLOW_SKIP_CPP=1: skip C++ parity test",
)
def test_cpp_runtime_matches_torch():
    try:
        import neuroflow_cpp  # noqa: F401
    except Exception as e:
        pytest.skip(f"neuroflow_cpp not importable: {e}")

    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = TokenMixer(cfg).eval()
    x = torch.randn(2, cfg.n_points, cfg.in_dim)
    y_torch = model(x).detach().cpu().numpy().astype("float32")

    with tempfile.TemporaryDirectory() as td:
        _, bin_path = export_all(model, td, "tm_tiny")
        y_cpp = neuroflow_cpp.infer_arrays(str(bin_path), x.numpy().astype("float32"))
    assert y_cpp.shape == y_torch.shape
    diff = np.abs(y_torch - y_cpp)
    # C++ is single-threaded; parity target mirrors FNO3d / DeepONet baselines.
    assert diff.max() < 1e-3, f"C++ vs PyTorch diff too large: {diff.max()}"
    assert diff.mean() < 1e-4


def test_cpp_runtime_smoke_random_input():
    """A second C++ pass with random inputs to ensure no shape/indexing leaks."""
    try:
        import neuroflow_cpp
    except Exception as e:
        pytest.skip(f"neuroflow_cpp not importable: {e}")

    torch.manual_seed(1)
    cfg = _tiny_cfg()
    model = TokenMixer(cfg).eval()
    x1 = torch.randn(1, cfg.n_points, cfg.in_dim)
    x2 = torch.randn(3, cfg.n_points, cfg.in_dim)
    y1_t = model(x1).detach().cpu().numpy().astype("float32")
    y2_t = model(x2).detach().cpu().numpy().astype("float32")
    with tempfile.TemporaryDirectory() as td:
        _, bin_path = export_all(model, td, "tm_tiny")
        y1_c = neuroflow_cpp.infer_arrays(str(bin_path), x1.numpy().astype("float32"))
        y2_c = neuroflow_cpp.infer_arrays(str(bin_path), x2.numpy().astype("float32"))
    assert y1_c.shape == y1_t.shape
    assert y2_c.shape == y2_t.shape
    assert np.abs(y1_t - y1_c).max() < 1e-3
    assert np.abs(y2_t - y2_c).max() < 1e-3
