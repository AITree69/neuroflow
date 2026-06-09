"""Fourier Neural Operator — 2D variant (Stage 2 Sprint 1).

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential
    Equations", ICLR 2021.

Stage 2 scope:
    - FNO2d with spectral convolutions on 2D signals (Darcy / 2D heat / etc.)
    - Same architecture skeleton as FNO1d: lifting -> L x [spectral + local
      + GELU] -> Q + projection head.
    - Weight shapes captured for NeuroIR v1 export:
        - spectral: (in, out, modes_h, modes_w)
        - local:    (out, in)        (PyTorch nn.Linear layout)
    - Forward signature: (batch, h, w, in_channels) -> (batch, h, w, out_channels).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FNO2dConfig:
    """Configuration for FNO2d. Used both at construction and IR export."""

    in_channels: int = 1
    out_channels: int = 1
    width: int = 64
    modes_h: int = 16
    modes_w: int = 16
    n_layers: int = 4
    activation: str = "gelu"
    pad_factor: int = 1
    name: str = "fno2d"


class SpectralConv2d(nn.Module):
    """2D spectral convolution: 2D real FFT -> truncate to a (modes_h, modes_w)
    rectangle of low frequencies -> pointwise complex linear -> 2D inverse FFT.

    Maintains two trainable weights of shape (in_channels, out_channels, modes_h, modes_w)
    (real and imaginary parts). The ordering matches the 1D variant
    (SpectralConv1d) and FNO paper convention; in/out are NOT PyTorch
    nn.Linear ordering.
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes_h: int, modes_w: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes_h, modes_w, dtype=torch.float32)
        )
        self.weights_imag = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes_h, modes_w, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, h, w)
        batch, _, h, w = x.shape
        # rfftn along the last two dims gives shape (batch, channels, h, w//2+1).
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        # Truncate to (modes_h, modes_w) rectangle of low frequencies.
        x_ft_trunc = x_ft[..., : self.modes_h, : self.modes_w]
        # Complex multiply: (b, in, mh, mw) * (in, out, mh, mw) -> (b, out, mh, mw)
        weights = torch.complex(self.weights_real, self.weights_imag)
        out_ft = torch.einsum("bimn,iomn->bomn", x_ft_trunc, weights)
        # Pad the w//2+1 axis back from `modes_w` to the full size.
        # The h axis (rfft on h gives full h bins) is already at size h;
        # we truncated to modes_h so we need to pad modes_h -> h.
        out_ft_padded = F.pad(
            out_ft, (0, (w // 2 + 1) - self.modes_w, 0, h - self.modes_h)
        )  # back to (b, out, h, w//2+1)
        out = torch.fft.irfftn(out_ft_padded, s=(h, w), dim=(-2, -1))
        return out

    def export_weights(self) -> dict[str, torch.Tensor]:
        """Return weights as a dict for IR export."""
        return {
            "weights_real": self.weights_real.detach().cpu(),
            "weights_imag": self.weights_imag.detach().cpu(),
        }


class FNO2d(nn.Module):
    """2D Fourier Neural Operator.

    Architecture:
        Lifting linear (c -> w) ->
        L blocks of [SpectralConv2d + skip linear + GELU] ->
        Projection Q (w -> w) + GELU + Projection out (w -> out_channels).

    Forward signature: (batch, h, w, in_channels) -> (batch, h, w, out_channels).
    """

    def __init__(self, config: FNO2dConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = FNO2dConfig(**kwargs)
        self.config = config

        c, w, mh, mw, L = (
            config.in_channels,
            config.width,
            config.modes_h,
            config.modes_w,
            config.n_layers,
        )
        self.width = w
        self.modes_h = mh
        self.modes_w = mw
        self.pad_factor = config.pad_factor

        # Lifting
        self.lift = nn.Linear(c, w)

        # Spectral convs + skip connections
        self.specs = nn.ModuleList(
            [SpectralConv2d(w, w, mh, mw) for _ in range(L)]
        )
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
        # x: (batch, h, w, in_channels)
        x = self.lift(x)  # (batch, h, w, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, h, w)

        # Optionally pad h and w to multiples of pad_factor
        if self.pad_factor > 1:
            pf = self.pad_factor
            pad_h = (pf - (x.shape[-2] % pf)) % pf
            pad_w = (pf - (x.shape[-1] % pf)) % pf
            # F.pad order for the last two dims is (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            pad_h = 0
            pad_w = 0

        for spec_conv, loc_lin in zip(self.specs, self.locs, strict=True):
            x1 = spec_conv(x)  # (b, w, h, w)
            # Local linear along the channel dim only: permute (b, w, h, w) ->
            # (b, h, w, w), apply Linear(w->w) on last dim, permute back.
            x2 = loc_lin(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x1 + x2
            x = self.act(x)

        if self.pad_factor > 1 and (pad_h > 0 or pad_w > 0):
            x = x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]

        # Project back to (batch, h, w, out_channels)
        x = x.permute(0, 2, 3, 1)
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
