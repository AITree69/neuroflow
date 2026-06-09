"""2D Transolver-style "TokenMixer" operator (Stage 2 Sprint 3.5).

Reference: Wu et al., "Transolver: A Efficient Transformer Operator
for Physical PDEs", ICLR 2024.  We extend the Stage 2 1D TokenMixer
(`neuroflow.nn.tokenmixer`) to a 2D regular grid: the input field
is flattened along the spatial dimensions, the slice/attn/unslice
pipeline runs along the resulting 1D sequence of length
$n_{\\mathrm{points}} = h \\cdot w$, and the output is reshaped back
to a per-point representation on the grid.

Architecture (mirrors `TokenMixer`):
  1. SliceEmbed:  (b, h, w, in_dim) -> flatten -> (b, n_p, in_dim)
                  -> group into n_patches mean-pool -> (b, n_patches, latent_dim)
  2. TransformerBlock x n_layers (Stage 2 limit: 1):
        Pre-LN -> Q,K,V Linear -> multi-head self-attn over (n_patches, head_dim)
        -> O Linear -> residual
        Pre-LN -> FFN -> residual
  3. UnsliceDecode: broadcast patch back to per-point, concat with
        original per-point features, Linear -> latent_dim.
  4. Head: Linear(latent_dim -> out_dim).
  5. Reshape (b, n_p, out_dim) -> (b, h, w, out_dim).

Stage 2 limitation: $n_{\\mathrm{layers}}=1$ only.

Forward signature: (batch, h, w, in_dim) -> (batch, h, w, out_dim).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn

from neuroflow.nn.tokenmixer import (
    TransformerBlock as _TransformerBlock1D,
)


@dataclass
class TokenMixer2DConfig:
    in_dim: int = 1
    out_dim: int = 1
    h: int = 16
    w: int = 16
    n_patches: int = 16
    latent_dim: int = 32
    n_heads: int = 4
    n_layers: int = 1
    activation: str = "gelu"
    name: str = "tokenmixer2d"

    def __post_init__(self) -> None:
        if self.latent_dim % self.n_heads != 0:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be a multiple of "
                f"n_heads ({self.n_heads})"
            )
        if (self.h * self.w) % self.n_patches != 0:
            raise ValueError(
                f"h*w ({self.h * self.w}) must be a multiple of "
                f"n_patches ({self.n_patches})"
            )


class TokenMixer2D(nn.Module):
    """2D Transolver-style operator learner (Stage 2 simplified).

    SliceEmbed and UnsliceDecode operate along the flattened
    spatial dimension; the transformer block is the same as the
    1D version.  This keeps the IR + C++ runtime re-usable.
    """

    def __init__(self, config: TokenMixer2DConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = TokenMixer2DConfig(**kwargs)
        self.config = config

        n_points = config.h * config.w

        # SliceEmbed on a flattened (b, n_points, in_dim) -> (b, n_patches, latent_dim).
        self.slice_embed = nn.Linear(config.in_dim, config.latent_dim)
        # TransformerBlock over the latent tokens.
        self.blocks = nn.ModuleList([
            _TransformerBlock1D(config.latent_dim, config.n_heads, config.activation)
            for _ in range(config.n_layers)
        ])
        # UnsliceDecode: concat(in_dim, latent) -> latent.
        self.unslice = nn.Linear(config.in_dim + config.latent_dim, config.latent_dim)
        # Head.
        self.head = nn.Linear(config.latent_dim, config.out_dim)

        # Sanity (catch misuse early).
        if n_points % config.n_patches != 0:
            # already caught by __post_init__, but kept as a safety net.
            raise ValueError(
                f"h*w ({n_points}) must be a multiple of n_patches ({config.n_patches})"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, h, w, in_dim)
        bsz, h, w, in_d = x.shape
        n_points = h * w
        n_patches = self.config.n_patches
        latent = self.config.latent_dim
        pp = n_points // n_patches

        # Flatten spatial: (b, n_points, in_dim)
        x_flat = x.view(bsz, n_points, in_d)
        # Mean-pool into patches: (b, n_patches, in_dim)
        x_patches = x_flat.view(bsz, n_patches, pp, in_d).mean(dim=2)
        # Lift to latent tokens.
        tokens = self.slice_embed(x_patches)  # (b, n_patches, latent)

        # Transformer blocks (in the 1D latent space — same as 1D TokenMixer).
        for block in self.blocks:
            tokens = block(tokens)

        # Broadcast tokens back to per-point: (b, n_points, latent)
        tokens_rep = tokens.repeat_interleave(pp, dim=1)
        # Concat with original per-point features: (b, n_points, in_d + latent)
        h_in = torch.cat([x_flat, tokens_rep], dim=-1)
        features = self.unslice(h_in)  # (b, n_points, latent)

        # Head: (b, n_points, out_d)
        y_flat = self.head(features)
        # Reshape back to 2D grid.
        y = y_flat.view(bsz, h, w, self.config.out_dim)
        return y

    def state_dict_for_ir(self) -> "OrderedDict[str, torch.Tensor]":
        sd: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        sd["slice_embed.proj.weight"] = self.slice_embed.weight.detach().cpu()
        sd["slice_embed.proj.bias"] = self.slice_embed.bias.detach().cpu()
        for i, block in enumerate(self.blocks):
            sd[f"blocks.{i}.ln1.weight"] = block.ln1.weight.detach().cpu()
            sd[f"blocks.{i}.ln1.bias"] = block.ln1.bias.detach().cpu()
            sd[f"blocks.{i}.q_proj.weight"] = block.q_proj.weight.detach().cpu()
            sd[f"blocks.{i}.q_proj.bias"] = block.q_proj.bias.detach().cpu()
            sd[f"blocks.{i}.k_proj.weight"] = block.k_proj.weight.detach().cpu()
            sd[f"blocks.{i}.k_proj.bias"] = block.k_proj.bias.detach().cpu()
            sd[f"blocks.{i}.v_proj.weight"] = block.v_proj.weight.detach().cpu()
            sd[f"blocks.{i}.v_proj.bias"] = block.v_proj.bias.detach().cpu()
            sd[f"blocks.{i}.o_proj.weight"] = block.o_proj.weight.detach().cpu()
            sd[f"blocks.{i}.o_proj.bias"] = block.o_proj.bias.detach().cpu()
            sd[f"blocks.{i}.ln2.weight"] = block.ln2.weight.detach().cpu()
            sd[f"blocks.{i}.ln2.bias"] = block.ln2.bias.detach().cpu()
            sd[f"blocks.{i}.ffn0.weight"] = block.ffn0.weight.detach().cpu()
            sd[f"blocks.{i}.ffn0.bias"] = block.ffn0.bias.detach().cpu()
            sd[f"blocks.{i}.ffn1.weight"] = block.ffn1.weight.detach().cpu()
            sd[f"blocks.{i}.ffn1.bias"] = block.ffn1.bias.detach().cpu()
        sd["unslice.proj.weight"] = self.unslice.weight.detach().cpu()
        sd["unslice.proj.bias"] = self.unslice.bias.detach().cpu()
        sd["head.weight"] = self.head.weight.detach().cpu()
        sd["head.bias"] = self.head.bias.detach().cpu()
        return sd

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
