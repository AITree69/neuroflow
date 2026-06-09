"""Fourier Neural Operator (FNO) implementation.

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.
    Zongyi Li's reference implementation: https://github.com/neural-operator/neural-operator

Stage 1 scope:
    - FNO1d with spectral convolutions on 1D signals
    - Configurable width, modes, depth
    - Weight shapes captured for IR export
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FNO1dConfig:
    """Configuration for FNO1d. Used both at construction and IR export."""

    in_channels: int = 1
    out_channels: int = 1
    width: int = 64
    modes: int = 16
    n_layers: int = 4
    activation: str = "gelu"
    pad_factor: int = 1
    name: str = "fno1d"


class SpectralConv1d(nn.Module):
    """1D spectral convolution: FFT → truncate to first `modes` frequencies
    → pointwise linear → IFFT.

    Maintains two trainable weights of shape (in_channels, out_channels, modes),
    real and imaginary parts.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.float32)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, n)
        batch, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, channels, n//2+1)

        # Truncate to modes frequencies
        x_ft_trunc = x_ft[..., : self.modes]  # (batch, channels, modes)

        # Complex multiply: (b, in, m) * (in, out, m) -> (b, out, m)
        weights = torch.complex(self.weights_real, self.weights_imag)
        out_ft = torch.einsum("bim,iom->bom", x_ft_trunc, weights)

        # Inverse FFT
        out_ft_padded = F.pad(
            out_ft, (0, (n // 2 + 1) - self.modes)
        )  # back to (b, out, n//2+1)
        out = torch.fft.irfft(out_ft_padded, n=n, dim=-1)
        return out

    def export_weights(self) -> dict[str, torch.Tensor]:
        """Return weights as a dict for IR export."""
        return {
            "weights_real": self.weights_real.detach().cpu(),
            "weights_imag": self.weights_imag.detach().cpu(),
        }


class FNO1d(nn.Module):
    """1D Fourier Neural Operator.

    Architecture:
        Lifting linear → (n_layers blocks of [SpectralConv1d + skip linear + GELU]) →
        Projection linear (Q) → GELU → Projection linear (output).

    Forward signature: (batch, n, in_channels) -> (batch, n, out_channels).
    """

    def __init__(self, config: FNO1dConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = FNO1dConfig(**kwargs)
        self.config = config

        c, w, m, L = config.in_channels, config.width, config.modes, config.n_layers
        self.width = w
        self.modes = m
        self.pad_factor = config.pad_factor

        # Lifting
        self.lift = nn.Linear(c, w)

        # Spectral convs + skip connections
        self.specs = nn.ModuleList([SpectralConv1d(w, w, m) for _ in range(L)])
        self.locs = nn.ModuleList([nn.Linear(w, w) for _ in range(L)])

        # Activations
        if config.activation == "gelu":
            self.act = F.gelu
        elif config.activation == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"unknown activation: {config.activation}")

        # Projection: Q (w -> w) then output (w -> out_channels)
        self.proj_q = nn.Linear(w, w)
        self.proj_out = nn.Linear(w, config.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n, in_channels)
        x = self.lift(x)  # (batch, n, width)
        x = x.permute(0, 2, 1)  # (batch, width, n)

        # Optionally pad to make n a multiple of pad_factor * 2 for cleaner FFT
        if self.pad_factor > 1:
            pad_len = (self.pad_factor - (x.shape[-1] % self.pad_factor)) % self.pad_factor
            x = F.pad(x, (0, pad_len))

        for spec_conv, loc_lin in zip(self.specs, self.locs, strict=True):
            x1 = spec_conv(x)
            x2 = loc_lin(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x1 + x2
            x = self.act(x)

        if self.pad_factor > 1 and pad_len > 0:
            x = x[..., :-pad_len]

        # Project back to (batch, n, out_channels)
        x = x.permute(0, 2, 1)
        x = self.proj_q(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x

    def state_dict_for_ir(self) -> "OrderedDict[str, torch.Tensor]":
        """Collect weights in deterministic order for IR export."""
        sd: OrderedDict[str, torch.Tensor] = OrderedDict()
        sd["lift.weight"] = self.lift.weight.detach().cpu()
        sd["lift.bias"] = self.lift.bias.detach().cpu()
        for i, (spec, loc) in enumerate(zip(self.specs, self.locs, strict=True)):
            sd[f"specs.{i}.weights_real"] = spec.weights_real.detach().cpu()
            sd[f"specs.{i}.weights_imag"] = spec.weights_imag.detach().cpu()
            sd[f"locs.{i}.weight"] = loc.weight.detach().cpu()
            sd[f"locs.{i}.bias"] = loc.bias.detach().cpu()
        sd["proj_q.weight"] = self.proj_q.weight.detach().cpu()
        sd["proj_q.bias"] = self.proj_q.bias.detach().cpu()
        sd["proj_out.weight"] = self.proj_out.weight.detach().cpu()
        sd["proj_out.bias"] = self.proj_out.bias.detach().cpu()
        return sd

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
