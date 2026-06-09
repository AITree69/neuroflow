"""Stage 2 Sprint 3.6 — TokenMixer2D + GraphOp2D C++ parity tests.

The Python end-to-end checks live in test_2d_ops.py.  This file
focuses on the cross-language parity for the 2D operator
families introduced in Sprint 3.5 (Python) and made C++-native
in Sprint 3.6.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from neuroflow.ir.export import export_all
from neuroflow.nn.graph_op2d import GraphOp2D, GraphOp2DConfig
from neuroflow.nn.tokenmixer2d import TokenMixer2D, TokenMixer2DConfig


def _tiny_tm_cfg() -> TokenMixer2DConfig:
    return TokenMixer2DConfig(
        in_dim=1, out_dim=1, h=8, w=8,
        n_patches=4, latent_dim=8, n_heads=2, n_layers=1,
        activation="gelu", name="tokenmixer2d_parity",
    )


def _tiny_gcn_cfg() -> GraphOp2DConfig:
    return GraphOp2DConfig(
        in_dim=1, out_dim=1, h=8, w=8,
        hidden_dim=8, n_layers=1,
        activation="gelu", name="graphop2d_parity",
    )


def test_tokenmixer2d_python_forward_shape():
    cfg = _tiny_tm_cfg()
    model = TokenMixer2D(cfg).eval()
    x = torch.randn(2, cfg.h, cfg.w, cfg.in_dim)
    y = model(x)
    assert tuple(y.shape) == (2, cfg.h, cfg.w, cfg.out_dim)


def test_tokenmixer2d_neuroir_roundtrip():
    torch.manual_seed(0)
    cfg = _tiny_tm_cfg()
    model = TokenMixer2D(cfg).eval()
    x = torch.randn(2, cfg.h, cfg.w, cfg.in_dim)
    y_orig = model(x).detach().cpu().numpy().astype("float32")

    from neuroflow.ir.load import load_neuroir
    from neuroflow.ir.spec import NeuroIRSpec

    spec = NeuroIRSpec.from_json(model_to_json(model))
    m2 = load_neuroir(spec)
    y_reload = m2(x).detach().cpu().numpy().astype("float32")
    assert np.allclose(y_orig, y_reload, atol=0.0)


def test_graphop2d_python_forward_shape():
    cfg = _tiny_gcn_cfg()
    model = GraphOp2D(cfg).eval()
    x = torch.randn(2, cfg.h, cfg.w, cfg.in_dim)
    y = model(x)
    assert tuple(y.shape) == (2, cfg.h, cfg.w, cfg.out_dim)


def test_graphop2d_neuroir_roundtrip():
    torch.manual_seed(0)
    cfg = _tiny_gcn_cfg()
    model = GraphOp2D(cfg).eval()
    x = torch.randn(2, cfg.h, cfg.w, cfg.in_dim)
    y_orig = model(x).detach().cpu().numpy().astype("float32")

    from neuroflow.ir.load import load_neuroir
    from neuroflow.ir.spec import NeuroIRSpec

    spec = NeuroIRSpec.from_json(model_to_json(model))
    m2 = load_neuroir(spec)
    y_reload = m2(x).detach().cpu().numpy().astype("float32")
    assert np.allclose(y_orig, y_reload, atol=0.0)


# ---------------------------------------------------------------------------
# C++ parity (skipped if the C++ extension is unavailable).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NFLOW_SKIP_CPP") == "1",
    reason="NFLOW_SKIP_CPP=1: skip C++ parity test",
)
def test_tokenmixer2d_cpp_matches_torch():
    try:
        import neuroflow_cpp  # noqa: F401
    except Exception as e:
        pytest.skip(f"neuroflow_cpp not importable: {e}")

    torch.manual_seed(0)
    cfg = _tiny_tm_cfg()
    model = TokenMixer2D(cfg).eval()
    x = torch.randn(2, cfg.h, cfg.w, cfg.in_dim)
    y_torch = model(x).detach().cpu().numpy().astype("float32")

    with tempfile.TemporaryDirectory() as td:
        _, bin_path = export_all(model, td, "tm2d_parity")
        y_cpp = neuroflow_cpp.infer_arrays(str(bin_path),
                                            x.numpy().astype("float32"))
    assert y_cpp.shape == y_torch.shape
    diff = np.abs(y_torch - y_cpp)
    assert diff.max() < 1e-3, f"TokenMixer2D C++ vs PyTorch diff: {diff.max()}"


@pytest.mark.skipif(
    os.environ.get("NFLOW_SKIP_CPP") == "1",
    reason="NFLOW_SKIP_CPP=1: skip C++ parity test",
)
def test_graphop2d_cpp_matches_torch():
    try:
        import neuroflow_cpp  # noqa: F401
    except Exception as e:
        pytest.skip(f"neuroflow_cpp not importable: {e}")

    torch.manual_seed(0)
    cfg = _tiny_gcn_cfg()
    model = GraphOp2D(cfg).eval()
    x = torch.randn(2, cfg.h, cfg.w, cfg.in_dim)
    y_torch = model(x).detach().cpu().numpy().astype("float32")

    with tempfile.TemporaryDirectory() as td:
        _, bin_path = export_all(model, td, "gcn2d_parity")
        y_cpp = neuroflow_cpp.infer_arrays(str(bin_path),
                                            x.numpy().astype("float32"))
    assert y_cpp.shape == y_torch.shape
    diff = np.abs(y_torch - y_cpp)
    assert diff.max() < 1e-3, f"GraphOp2D C++ vs PyTorch diff: {diff.max()}"


def model_to_json(model):
    """Convenience: dump a model's NeuroIRSpec to JSON text."""
    from neuroflow.ir.export import export_to_neuroir
    return export_to_neuroir(model).to_json()
