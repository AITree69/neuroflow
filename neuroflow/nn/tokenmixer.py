"""Transolver-style "TokenMixer" operator (Stage 2 Sprint 3, simplified).

Reference:
    Wu et al., "Transolver: A Efficient Transformer Operator for Physical
    PDEs", ICLR 2024.

Core idea: project the spatial field into a small set of latent tokens,
run self-attention in the latent space (O(n_patches^2) instead of
O(n_points^2)), then broadcast back to the per-point representation
and combine with the original features.

Stage 2 simplification (this implementation):
    - One transformer block (n_layers=1). A real Transolver stacks
      several; that is a straightforward extension of this class.
    - Single "physics-aware" slice: mean-pool `n_points` into
      `n_patches` per the natural ordering. The original paper
      uses learned slice weights driven by the input field; we
      skip that to keep the IR layout tractable for v0.7.0.
    - Pre-LN multi-head self-attention + 2-layer pointwise FFN inside
      each block.
    - The unslice step concatenates the per-point input features
      with the broadcast patch embedding and projects back to
      `latent_dim`.
    - A final head maps `latent_dim -> out_dim`.

Forward signature: (batch, n_points, in_dim) -> (batch, n_points, out_dim).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TokenMixerConfig:
    """Configuration for the TokenMixer (Transolver) operator.

    in_dim:    per-point input feature dim
    out_dim:   per-point output feature dim
    n_points:  number of spatial points per sample (fixed)
    n_patches: number of latent tokens (must divide n_points for the
               mean-pool slice; we pad / truncate if not exact)
    latent_dim: token / per-point feature dim
    n_heads:   multi-head attention heads (must divide latent_dim)
    n_layers:  number of transformer blocks
    activation: "gelu" or "relu"
    name:      operator name (for IR config and plots)
    """

    in_dim: int = 1
    out_dim: int = 1
    n_points: int = 64
    n_patches: int = 8
    latent_dim: int = 32
    n_heads: int = 4
    n_layers: int = 1
    activation: str = "gelu"
    name: str = "tokenmixer"

    def __post_init__(self) -> None:
        if self.latent_dim % self.n_heads != 0:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be a multiple of "
                f"n_heads ({self.n_heads})"
            )


def _act(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"unknown activation: {name!r}")


class SliceEmbed(nn.Module):
    """(b, n_points, in_dim) -> (b, n_patches, latent_dim) via mean pool + Linear."""

    def __init__(self, in_dim: int, n_patches: int, n_points: int, latent_dim: int) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.n_points = n_points
        self.proj = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n, in_dim = x.shape
        # We want (b, n_patches, points_per_patch, in_dim). The natural
        # choice is points_per_patch = n // n_patches. If n is not a
        # multiple, we truncate to the largest multiple.
        pp = n // self.n_patches
        if pp * self.n_patches != n:
            # Truncate to a clean multiple so mean-pool is exact.
            x = x[:, : pp * self.n_patches, :]
            n_eff = pp * self.n_patches
        else:
            n_eff = n
        # (b, n_patches, pp, in_dim)
        x = x.view(bsz, self.n_patches, pp, in_dim)
        x = x.mean(dim=2)  # (b, n_patches, in_dim)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-LN multi-head self-attention + pointwise FFN."""

    def __init__(self, latent_dim: int, n_heads: int, activation: str) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.ln1 = nn.LayerNorm(latent_dim)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.o_proj = nn.Linear(latent_dim, latent_dim)

        self.ln2 = nn.LayerNorm(latent_dim)
        self.ffn0 = nn.Linear(latent_dim, 2 * latent_dim)
        self.ffn1 = nn.Linear(2 * latent_dim, latent_dim)

        self.act = _act(activation)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (b, n_patches, latent_dim)
        bsz, np_, _ = tokens.shape
        h = self.ln1(tokens)
        q = self.q_proj(h).view(bsz, np_, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, np_, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, np_, self.n_heads, self.head_dim).transpose(1, 2)
        # (b, h, np, hd) @ (b, h, hd, np) -> (b, h, np, np)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        # (b, h, np, np) @ (b, h, np, hd) -> (b, h, np, hd) -> (b, np, h*hd)
        out = (attn @ v).transpose(1, 2).reshape(bsz, np_, self.n_heads * self.head_dim)
        tokens = tokens + self.o_proj(out)

        h = self.ln2(tokens)
        tokens = tokens + self.ffn1(self.act(self.ffn0(h)))
        return tokens


class UnsliceDecode(nn.Module):
    """(b, n_patches, latent_dim) + (b, n_points, in_dim) -> (b, n_points, latent_dim).

    Each point inherits the embedding of its patch (broadcast), then we
    concatenate with the point's original in_dim features and project.
    """

    def __init__(self, in_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim + latent_dim, latent_dim)

    def forward(self, tokens: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        # tokens: (b, n_patches, latent_dim); x_orig: (b, n_points, in_dim)
        bsz, np_lat, _ = tokens.shape
        _, n, in_dim = x_orig.shape
        # Repeat each token across its patch (assumes points_per_patch = n / np_lat).
        pp = n // np_lat
        if pp * np_lat != n:
            x_orig = x_orig[:, : pp * np_lat, :]
            n_eff = pp * np_lat
        else:
            n_eff = n
        # (b, n_eff, latent_dim) by repeating each token pp times.
        tokens_rep = tokens.repeat_interleave(pp, dim=1)
        # Concat along the feature axis and project.
        h = torch.cat([x_orig, tokens_rep], dim=-1)
        return self.proj(h)


class TokenMixer(nn.Module):
    """Transolver-style operator learner (Stage 2 simplified).

    SliceEmbed: per-point input -> latent tokens
    Blocks:     Pre-LN multi-head self-attention + FFN
    UnsliceDecode: broadcast latent back, concat with per-point features
    Head:       latent_dim -> out_dim
    """

    def __init__(self, config: TokenMixerConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = TokenMixerConfig(**kwargs)
        self.config = config

        self.slice_embed = SliceEmbed(
            config.in_dim, config.n_patches, config.n_points, config.latent_dim
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.latent_dim, config.n_heads, config.activation
            )
            for _ in range(config.n_layers)
        ])
        self.unslice = UnsliceDecode(config.in_dim, config.latent_dim)
        self.head = nn.Linear(config.latent_dim, config.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_points, in_dim)
        tokens = self.slice_embed(x)  # (b, n_patches, latent_dim)
        for block in self.blocks:
            tokens = block(tokens)
        features = self.unslice(tokens, x)  # (b, n_points, latent_dim)
        out = self.head(features)  # (b, n_points, out_dim)
        return out

    def state_dict_for_ir(self) -> "OrderedDict[str, torch.Tensor]":
        """Collect weights in deterministic order for IR export.

        Naming follows the NeuroIR convention so the C++ loader can
        recognise each block:
            slice_embed.proj.{weight,bias}
            blocks.{i}.ln1.{weight,bias}
            blocks.{i}.q_proj.{weight,bias}
            blocks.{i}.k_proj.{weight,bias}
            blocks.{i}.v_proj.{weight,bias}
            blocks.{i}.o_proj.{weight,bias}
            blocks.{i}.ln2.{weight,bias}
            blocks.{i}.ffn0.{weight,bias}
            blocks.{i}.ffn1.{weight,bias}
            unslice.proj.{weight,bias}
            head.{weight,bias}
        """
        sd: OrderedDict[str, torch.Tensor] = OrderedDict()
        sd["slice_embed.proj.weight"] = self.slice_embed.proj.weight.detach().cpu()
        sd["slice_embed.proj.bias"] = self.slice_embed.proj.bias.detach().cpu()
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
        sd["unslice.proj.weight"] = self.unslice.proj.weight.detach().cpu()
        sd["unslice.proj.bias"] = self.unslice.proj.bias.detach().cpu()
        sd["head.weight"] = self.head.weight.detach().cpu()
        sd["head.bias"] = self.head.bias.detach().cpu()
        return sd

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
