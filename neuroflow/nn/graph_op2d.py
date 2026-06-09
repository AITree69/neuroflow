"""2D GCN-style operator (Stage 2 Sprint 3.5).

Mirrors `neuroflow.nn.graph_op.GraphOp` but for a 2D regular grid
graph (8-connectivity with self-loops).  Architecture:

  1. lift:  (b, h, w, in_dim) -> flatten -> (b, n_nodes, hidden_dim) -> Linear
  2. block: per-node degree-normalised 8-neighbour aggregation:
        h' = act(W_self h + W_neigh (D^-1 A h)) + h
  3. head:  (b, n_nodes, hidden_dim) -> (b, h, w, out_dim)

Forward: (batch, h, w, in_dim) -> (batch, h, w, out_dim).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphOp2DConfig:
    in_dim: int = 1
    out_dim: int = 1
    h: int = 16
    w: int = 16
    hidden_dim: int = 32
    n_layers: int = 1
    activation: str = "gelu"
    name: str = "graphop2d"


def _act(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"unknown activation: {name!r}")


def grid8_graph(h: int, w: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """Build a 2D grid graph with 8-connectivity (and self-loops).

    Returns:
        adj_offsets: (h*w + 1,) int64 — CSR row pointers.
        adj_indices: (variable,) int64 — CSR column indices.
    Each node (i, j) is connected to itself plus its 8 neighbours
    (i + di, j + dj) for di, dj in {-1, 0, 1} that fall inside the
    h x w grid.
    """
    offsets = [0]
    indices: list[int] = []
    for i in range(h):
        for j in range(w):
            neighbours: list[int] = [i * w + j]  # self-loop
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbours.append(ni * w + nj)
            indices.extend(neighbours)
            offsets.append(len(indices))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(indices, dtype=torch.int64),
    )


def compute_degree_inv(adj_offsets: torch.Tensor, adj_indices: torch.Tensor,
                       n_nodes: int) -> torch.Tensor:
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
        out[:, i, :] = h[:, nbs, :].sum(dim=1) * deg_inv[i]
    return out


class GCNBlock2D(nn.Module):
    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh = nn.Linear(hidden_dim, hidden_dim)
        self.act = _act(activation)

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin_self(h) + self.lin_neigh(agg)) + h


class GraphOp2D(nn.Module):
    def __init__(self, config: GraphOp2DConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = GraphOp2DConfig(**kwargs)
        self.config = config

        self.lift = nn.Linear(config.in_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([
            GCNBlock2D(config.hidden_dim, config.activation)
            for _ in range(config.n_layers)
        ])
        self.head = nn.Linear(config.hidden_dim, config.out_dim)

        adj_offsets, adj_indices = grid8_graph(config.h, config.w)
        deg_inv = compute_degree_inv(adj_offsets, adj_indices, config.h * config.w)
        self.register_buffer("adj_offsets", adj_offsets)
        self.register_buffer("adj_indices", adj_indices)
        self.register_buffer("deg_inv", deg_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, h, w, in_d = x.shape
        n_nodes = h * w
        hidden = self.config.hidden_dim

        # Flatten to (b, n_nodes, in_d) and lift.
        h_feat = self.lift(x.view(bsz, n_nodes, in_d))
        for block in self.blocks:
            agg = aggregate_neighbours(h_feat, self.adj_offsets,
                                        self.adj_indices, self.deg_inv)
            h_feat = block(h_feat, agg)

        y = self.head(h_feat)  # (b, n_nodes, out_d)
        return y.view(bsz, h, w, self.config.out_dim)

    def state_dict_for_ir(self) -> "OrderedDict[str, torch.Tensor]":
        sd: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        sd["lift.weight"] = self.lift.weight.detach().cpu()
        sd["lift.bias"] = self.lift.bias.detach().cpu()
        for i, block in enumerate(self.blocks):
            sd[f"blocks.{i}.lin_self.weight"] = block.lin_self.weight.detach().cpu()
            sd[f"blocks.{i}.lin_self.bias"] = block.lin_self.bias.detach().cpu()
            sd[f"blocks.{i}.lin_neigh.weight"] = block.lin_neigh.weight.detach().cpu()
            sd[f"blocks.{i}.lin_neigh.bias"] = block.lin_neigh.bias.detach().cpu()
        sd["head.weight"] = self.head.weight.detach().cpu()
        sd["head.bias"] = self.head.bias.detach().cpu()
        sd["graph.adj_offsets"] = self.adj_offsets.detach().cpu().to(torch.float32)
        sd["graph.adj_indices"] = self.adj_indices.detach().cpu().to(torch.float32)
        sd["graph.deg_inv"] = self.deg_inv.detach().cpu()
        return sd

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
