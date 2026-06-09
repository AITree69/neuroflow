"""Graph operator learner (Stage 2 Sprint 3.2, GCN-style simplification).

Reference: Kipf & Welling, "Semi-Supervised Classification with Graph
Convolutional Networks", ICLR 2017. We use a *single hidden* GCN with
self-loops, applied to a fixed graph whose adjacency is supplied at
construction time:

    H' = act( W_self @ H + W_neigh @ (D^-1 A H) )
    Y  = head(H')

Where A is the binary adjacency of the supplied graph (with self-loops)
and D is the diagonal degree matrix. The graph topology is *not* a
weight — it is encoded as a list of per-node neighbor offsets and
indices and stored alongside the trained weights in the NeuroIR
export.

Stage 2 simplification:
    - The graph is a regular 1D line (each node is connected to its
      immediate neighbours, with self-loops). Higher-order /
      multi-hop message passing and edge features are a Stage 3
      extension.
    - One hidden layer + one head (n_layers=1). Multi-block is a
      Stage 3 extension.
    - No graph-level readout; per-node outputs only.

Forward signature: (batch, n_nodes, in_dim) -> (batch, n_nodes, out_dim).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphOpConfig:
    """Configuration for the GraphOp (GCN-style) operator.

    in_dim:    per-node input feature dim
    out_dim:   per-node output feature dim
    n_nodes:   number of nodes in the graph (fixed)
    hidden_dim: width of the GCN hidden / message channel
    n_layers:  number of GCN message-passing layers (Stage 2 limit: 1)
    activation: "gelu" or "relu"
    name:      operator name (for IR config and plots)
    """

    in_dim: int = 1
    out_dim: int = 1
    n_nodes: int = 64
    hidden_dim: int = 32
    n_layers: int = 1
    activation: str = "gelu"
    name: str = "graphop"


def _act(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"unknown activation: {name!r}")


def line_graph(n_nodes: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """Build a 1D line graph with self-loops.

    Returns:
        adj_offsets: (n_nodes + 1,) int64 — CSR row pointers.
        adj_indices: (n_edges,) int64 — CSR column indices.
    The neighbor list of node ``i`` is adj_indices[adj_offsets[i]:adj_offsets[i+1]].
    Self-loops are added (i is always its own neighbour), so each node has
    2 or 3 neighbours (left, self, right) depending on position.
    """
    offsets = [0]
    indices: list[int] = []
    for i in range(n_nodes):
        neighbours = [i]
        if i > 0:
            neighbours.append(i - 1)
        if i < n_nodes - 1:
            neighbours.append(i + 1)
        indices.extend(neighbours)
        offsets.append(len(indices))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(indices, dtype=torch.int64),
    )


def compute_degree_inv(adj_offsets: torch.Tensor, adj_indices: torch.Tensor,
                       n_nodes: int) -> torch.Tensor:
    """Return 1 / deg(i) for the (self-loop-included) graph."""
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    for i in range(n_nodes):
        deg[i] = float(adj_offsets[i + 1] - adj_offsets[i])
    return 1.0 / deg.clamp_min(1.0)


def aggregate_neighbours(
    h: torch.Tensor,
    adj_offsets: torch.Tensor,
    adj_indices: torch.Tensor,
    deg_inv: torch.Tensor,
) -> torch.Tensor:
    """For each node i, sum h[j] * deg_inv[i] over j in adj[i].

    h: (b, n_nodes, hidden). Returns (b, n_nodes, hidden).
    """
    bsz, n_nodes, hidden = h.shape
    out = torch.zeros_like(h)
    for i in range(n_nodes):
        nbs = adj_indices[adj_offsets[i]:adj_offsets[i + 1]]
        # Sum over neighbors: (b, k, hidden) -> (b, hidden) for k=|nbs|
        agg = h[:, nbs, :].sum(dim=1)  # (b, hidden)
        out[:, i, :] = agg * deg_inv[i]
    return out


class GCNBlock(nn.Module):
    """A single GCN message-passing layer: W_self + W_neigh, residual + act."""

    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh = nn.Linear(hidden_dim, hidden_dim)
        self.act = _act(activation)

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin_self(h) + self.lin_neigh(agg)) + h


class GraphOp(nn.Module):
    """Graph operator learner (Stage 2 simplified GCN).

    Architecture:
        lift:     Linear(in_dim -> hidden_dim)
        block:    GCNBlock x n_layers
        head:     Linear(hidden_dim -> out_dim)
    The graph topology (neighbor offsets, neighbour indices, degree
    inverse) is stored in `self.graph` and exported to the IR.
    """

    def __init__(self, config: GraphOpConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = GraphOpConfig(**kwargs)
        self.config = config

        self.lift = nn.Linear(config.in_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([
            GCNBlock(config.hidden_dim, config.activation)
            for _ in range(config.n_layers)
        ])
        self.head = nn.Linear(config.hidden_dim, config.out_dim)

        # Topology buffers (registered so .to(device) moves them too).
        adj_offsets, adj_indices = line_graph(config.n_nodes)
        deg_inv = compute_degree_inv(adj_offsets, adj_indices, config.n_nodes)
        self.register_buffer("adj_offsets", adj_offsets)
        self.register_buffer("adj_indices", adj_indices)
        self.register_buffer("deg_inv", deg_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n_nodes, in_dim)
        h = self.lift(x)  # (b, n_nodes, hidden_dim)
        for block in self.blocks:
            agg = aggregate_neighbours(h, self.adj_offsets, self.adj_indices,
                                        self.deg_inv)
            h = block(h, agg)
        return self.head(h)

    def state_dict_for_ir(self) -> "OrderedDict[str, torch.Tensor]":
        """Collect weights in deterministic order for IR export.

        Naming follows the NeuroIR convention so the C++ loader can
        recognise each block:
            lift.{weight,bias}
            blocks.{i}.lin_self.{weight,bias}
            blocks.{i}.lin_neigh.{weight,bias}
            head.{weight,bias}
            graph.adj_offsets   (int64, n_nodes + 1)
            graph.adj_indices   (int64, variable)
            graph.deg_inv       (float32, n_nodes)
        """
        sd: OrderedDict[str, torch.Tensor] = OrderedDict()
        sd["lift.weight"] = self.lift.weight.detach().cpu()
        sd["lift.bias"] = self.lift.bias.detach().cpu()
        for i, block in enumerate(self.blocks):
            sd[f"blocks.{i}.lin_self.weight"] = block.lin_self.weight.detach().cpu()
            sd[f"blocks.{i}.lin_self.bias"] = block.lin_self.bias.detach().cpu()
            sd[f"blocks.{i}.lin_neigh.weight"] = block.lin_neigh.weight.detach().cpu()
            sd[f"blocks.{i}.lin_neigh.bias"] = block.lin_neigh.bias.detach().cpu()
        sd["head.weight"] = self.head.weight.detach().cpu()
        sd["head.bias"] = self.head.bias.detach().cpu()
        # Graph topology — float32 for both the int arrays and deg_inv
        # (the IR `Tensor` only supports float32 weights; we mark the
        # meaning with a name prefix and cast in C++).
        sd["graph.adj_offsets"] = self.adj_offsets.detach().cpu().to(torch.float32)
        sd["graph.adj_indices"] = self.adj_indices.detach().cpu().to(torch.float32)
        sd["graph.deg_inv"] = self.deg_inv.detach().cpu()
        return sd

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
