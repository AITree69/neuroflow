"""DeepONet — branch/trunk 双塔算子学习 (Stage 2 Sprint 2).

Reference:
    Lu Lu et al., "DeepONet: Learning nonlinear operators for identifying
    differential equations based on the universal approximation theorem
    of operators", 2021.

Stage 2 scope:
    - Branch net: takes a discretised function u (sampled at `n_sensor`
      points) and produces a (b, out_ch, latent_dim) representation
      (after mean-pooling over sensors).
    - Trunk net: takes a query location y (a vector of dimension
      `in_trunk`) and produces a (b, n_query, latent_dim) representation.
    - Forward: y[b, i, c] = sum_k branch[b, c, k] * trunk[b, i, k] + bias[c]
      via `einsum("bck,bik->bci", branch, trunk)`.
    - Weights stored in a single `branch.layers.{i}.weight/bias` /
      `trunk.layers.{i}.weight/bias` / `bias.weight` layout, exported
      to NeuroIR v0.4.0 with `op_code = 0x04`.

This implementation supports arbitrary `out_channels`. The classic
DeepONet (Lu 2021) targets a scalar output (out_ch=1); multi-out is
a natural extension where each output channel gets its own branch
coefficients (shared trunk weights).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeepONetConfig:
    """Configuration for DeepONet. Used at construction and IR export."""

    in_branch: int = 100      # sensor point feature dim (e.g. 1 for scalar u)
    in_trunk: int = 1         # query point feature dim (e.g. 1 for scalar x)
    latent_dim: int = 32
    out_channels: int = 1
    hidden_branch: int = 64
    hidden_trunk: int = 64
    n_layers_branch: int = 3
    n_layers_trunk: int = 3
    activation: str = "gelu"
    name: str = "deeponet"


class _MLP(nn.Module):
    """Small MLP used by both branch and trunk. Activation is supplied."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        n_layers: int,
        activation: str,
        name: str,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"{name}: n_layers must be >= 1")
        layers: list[nn.Linear] = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Linear(hidden, out_dim))
        self.layers = nn.ModuleList(layers)
        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"{name}: unknown activation {activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


class BranchNet(nn.Module):
    """Branch network: (b, n_sensor, in_branch) -> (b, out_ch, latent_dim).

    The last linear expands to `out_ch * latent_dim` so that each
    output channel has its own latent representation. The intermediate
    `n_sensor` axis is reduced by mean-pooling to get a permutation-
    invariant representation of the input function u.
    """

    def __init__(
        self,
        in_branch: int,
        hidden: int,
        latent_dim: int,
        out_channels: int,
        n_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.mlp = _MLP(
            in_dim=in_branch,
            hidden=hidden,
            out_dim=out_channels * latent_dim,
            n_layers=n_layers,
            activation=activation,
            name="branch",
        )
        self.out_channels = out_channels
        self.latent_dim = latent_dim

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (b, n_sensor, in_branch)
        b = self.mlp(u)  # (b, n_sensor, out_ch * latent_dim)
        bsz, n_sensor, _ = b.shape
        b = b.view(bsz, n_sensor, self.out_channels, self.latent_dim)
        # Mean over the sensor axis (n_sensor).
        b = b.mean(dim=1)  # (b, out_ch, latent_dim)
        return b


class TrunkNet(nn.Module):
    """Trunk network: (b, n_query, in_trunk) -> (b, n_query, latent_dim)."""

    def __init__(
        self,
        in_trunk: int,
        hidden: int,
        latent_dim: int,
        n_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.mlp = _MLP(
            in_dim=in_trunk,
            hidden=hidden,
            out_dim=latent_dim,
            n_layers=n_layers,
            activation=activation,
            name="trunk",
        )
        self.latent_dim = latent_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (b, n_query, in_trunk)
        return self.mlp(y)


class DeepONet(nn.Module):
    """DeepONet operator learner.

    Forward:
        u: (b, n_sensor, in_branch)   discretised input function
        y: (b, n_query, in_trunk)     query locations
    Returns:
        out: (b, n_query, out_channels) operator output

    The architecture mirrors Lu Lu et al. 2021 with a multi-channel
    extension: each output channel gets its own branch coefficients
    (the branch MLP's last layer expands to out_ch * latent_dim);
    the trunk weights are shared across output channels.
    """

    def __init__(self, config: DeepONetConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = DeepONetConfig(**kwargs)
        self.config = config

        self.branch = BranchNet(
            in_branch=config.in_branch,
            hidden=config.hidden_branch,
            latent_dim=config.latent_dim,
            out_channels=config.out_channels,
            n_layers=config.n_layers_branch,
            activation=config.activation,
        )
        self.trunk = TrunkNet(
            in_trunk=config.in_trunk,
            hidden=config.hidden_trunk,
            latent_dim=config.latent_dim,
            n_layers=config.n_layers_trunk,
            activation=config.activation,
        )
        # Per-output-channel bias added after the dot product.
        self.bias = nn.Parameter(torch.zeros(config.out_channels, dtype=torch.float32))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # u: (b, n_sensor, in_branch); y: (b, n_query, in_trunk)
        b = self.branch(u)  # (b, out_ch, latent_dim)
        t = self.trunk(y)   # (b, n_query, latent_dim)
        # Einsum: contract over the latent axis.
        out = torch.einsum("bck,bik->bci", b, t)  # (b, out_ch, n_query)
        out = out.permute(0, 2, 1).contiguous()    # (b, n_query, out_ch)
        out = out + self.bias
        return out

    def state_dict_for_ir(self) -> "OrderedDict[str, torch.Tensor]":
        """Collect weights in deterministic order for IR export.

        Naming follows the NeuroIR convention so the C++ loader can
        recognise each block:

            branch.layers.{i}.weight   (hidden, in_dim) per layer
            branch.layers.{i}.bias     (hidden,)
            trunk.layers.{i}.weight    (hidden, in_dim) per layer
            trunk.layers.{i}.bias      (hidden,)
            bias.weight                (out_ch,)
        """
        sd: OrderedDict[str, torch.Tensor] = OrderedDict()
        for i, layer in enumerate(self.branch.mlp.layers):
            sd[f"branch.layers.{i}.weight"] = layer.weight.detach().cpu()
            sd[f"branch.layers.{i}.bias"] = layer.bias.detach().cpu()
        for i, layer in enumerate(self.trunk.mlp.layers):
            sd[f"trunk.layers.{i}.weight"] = layer.weight.detach().cpu()
            sd[f"trunk.layers.{i}.bias"] = layer.bias.detach().cpu()
        sd["bias.weight"] = self.bias.detach().cpu()
        return sd

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
