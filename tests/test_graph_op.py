"""Stage 2 Sprint 3.2 — GraphOp (GCN-style operator learner) tests.

Covers:
  - Python forward shape and parameter count
  - state_dict_for_ir layout, including the graph topology entries
  - line_graph + degree-inverse helper correctness
  - NeuroIR roundtrip (JSON + binary)
  - Pure NumPy forward matches PyTorch within float32 noise
  - C++ runtime (via neuroflow_cpp) matches PyTorch within float32 noise
"""

from __future__ import annotations

import os
import tempfile

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
from neuroflow.nn.graph_op import (
    GraphOp,
    GraphOpConfig,
    compute_degree_inv,
    line_graph,
)


def _default_cfg() -> GraphOpConfig:
    return GraphOpConfig(
        in_dim=2,
        out_dim=1,
        n_nodes=64,
        hidden_dim=32,
        n_layers=1,
        activation="gelu",
        name="graphop_test",
    )


def _tiny_cfg() -> GraphOpConfig:
    return GraphOpConfig(
        in_dim=1,
        out_dim=1,
        n_nodes=16,
        hidden_dim=8,
        n_layers=1,
        activation="gelu",
        name="graphop_tiny",
    )


def test_python_forward_shape():
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = GraphOp(cfg).eval()
    x = torch.randn(2, cfg.n_nodes, cfg.in_dim)
    y = model(x)
    assert tuple(y.shape) == (2, cfg.n_nodes, cfg.out_dim)


def test_python_forward_deterministic():
    cfg = _default_cfg()
    model = GraphOp(cfg).eval()
    x = torch.randn(2, cfg.n_nodes, cfg.in_dim)
    y1 = model(x)
    y2 = model(x)
    assert torch.allclose(y1, y2)


def test_line_graph_endpoints():
    offsets, indices = line_graph(5)
    # Each node has 2 or 3 neighbours (self + optional left + optional right).
    # Node 0: [0, 1]            -> length 2
    # Node 1: [1, 0, 2]         -> length 3
    # Node 2: [2, 1, 3]         -> length 3
    # Node 3: [3, 2, 4]         -> length 3
    # Node 4: [4, 3]            -> length 2
    assert offsets.tolist() == [0, 2, 5, 8, 11, 13]
    assert indices.tolist() == [0, 1, 1, 0, 2, 2, 1, 3, 3, 2, 4, 4, 3]
    deg = compute_degree_inv(offsets, indices, 5)
    assert torch.allclose(deg, torch.tensor([0.5, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5]))


def test_state_dict_for_ir_layout():
    cfg = _default_cfg()
    model = GraphOp(cfg).eval()
    sd = model.state_dict_for_ir()
    expected = [
        "lift.weight", "lift.bias",
        "blocks.0.lin_self.weight", "blocks.0.lin_self.bias",
        "blocks.0.lin_neigh.weight", "blocks.0.lin_neigh.bias",
        "head.weight", "head.bias",
        "graph.adj_offsets", "graph.adj_indices", "graph.deg_inv",
    ]
    assert list(sd.keys()) == expected
    assert tuple(sd["lift.weight"].shape) == (cfg.hidden_dim, cfg.in_dim)
    assert tuple(sd["blocks.0.lin_self.weight"].shape) == (cfg.hidden_dim, cfg.hidden_dim)
    assert tuple(sd["head.weight"].shape) == (cfg.out_dim, cfg.hidden_dim)
    assert tuple(sd["graph.adj_offsets"].shape) == (cfg.n_nodes + 1,)
    assert tuple(sd["graph.deg_inv"].shape) == (cfg.n_nodes,)


def test_neuroir_roundtrip():
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = GraphOp(cfg).eval()
    x = torch.randn(2, cfg.n_nodes, cfg.in_dim)
    y_orig = model(x).detach().cpu().numpy().astype("float32")

    spec = export_to_neuroir(model)
    assert spec.op == "GraphOp"
    assert spec.config["in_dim"] == cfg.in_dim
    assert spec.config["out_dim"] == cfg.out_dim
    assert spec.config["n_nodes"] == cfg.n_nodes
    assert spec.config["hidden_dim"] == cfg.hidden_dim
    assert spec.config["n_layers"] == cfg.n_layers

    spec2 = NeuroIRSpec.from_json(spec.to_json())
    model2 = load_neuroir(spec2)
    y_reload = model2(x).detach().cpu().numpy().astype("float32")
    assert np.allclose(y_orig, y_reload, atol=0.0)


def test_neuroir_binary_roundtrip():
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = GraphOp(cfg).eval()
    x = torch.randn(2, cfg.n_nodes, cfg.in_dim)
    y_orig = model(x).detach().cpu().numpy().astype("float32")

    spec = export_to_neuroir(model)
    bin_data = export_to_binary(spec)
    assert bin_data[:4] == b"NIR0"
    assert bin_data[4] == 3
    assert bin_data[6] == 0x06  # op_code for GraphOp

    spec2 = NeuroIRSpec.from_json(spec.to_json())
    model2 = load_neuroir(spec2)
    y_reload = model2(x).detach().cpu().numpy().astype("float32")
    assert np.allclose(y_orig, y_reload, atol=0.0)


def test_numpy_forward_matches_torch():
    torch.manual_seed(0)
    cfg = _default_cfg()
    model = GraphOp(cfg).eval()
    x = torch.randn(2, cfg.n_nodes, cfg.in_dim)
    spec = export_to_neuroir(model)
    y_torch = predict_with_spec_torch(spec, x).cpu().numpy().astype("float32")
    y_numpy = predict_with_spec(spec, x.numpy().astype("float32"))
    diff = np.abs(y_torch - y_numpy)
    assert diff.max() < 1e-3
    assert diff.mean() < 1e-4


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
    model = GraphOp(cfg).eval()
    x = torch.randn(2, cfg.n_nodes, cfg.in_dim)
    y_torch = model(x).detach().cpu().numpy().astype("float32")

    with tempfile.TemporaryDirectory() as td:
        _, bin_path = export_all(model, td, "gcn_tiny")
        y_cpp = neuroflow_cpp.infer_arrays(str(bin_path), x.numpy().astype("float32"))
    assert y_cpp.shape == y_torch.shape
    diff = np.abs(y_torch - y_cpp)
    assert diff.max() < 1e-3
    assert diff.mean() < 1e-4


def test_cpp_runtime_smoke_random_input():
    try:
        import neuroflow_cpp
    except Exception as e:
        pytest.skip(f"neuroflow_cpp not importable: {e}")

    torch.manual_seed(1)
    cfg = _tiny_cfg()
    model = GraphOp(cfg).eval()
    x1 = torch.randn(1, cfg.n_nodes, cfg.in_dim)
    x2 = torch.randn(3, cfg.n_nodes, cfg.in_dim)
    y1_t = model(x1).detach().cpu().numpy().astype("float32")
    y2_t = model(x2).detach().cpu().numpy().astype("float32")
    with tempfile.TemporaryDirectory() as td:
        _, bin_path = export_all(model, td, "gcn_tiny")
        y1_c = neuroflow_cpp.infer_arrays(str(bin_path), x1.numpy().astype("float32"))
        y2_c = neuroflow_cpp.infer_arrays(str(bin_path), x2.numpy().astype("float32"))
    assert y1_c.shape == y1_t.shape
    assert y2_c.shape == y2_t.shape
    assert np.abs(y1_t - y1_c).max() < 1e-3
    assert np.abs(y2_t - y2_c).max() < 1e-3
