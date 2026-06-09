"""Fourier Neural Operator — 3D variant (Stage 2 Sprint 2).

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential
    Equations", ICLR 2021.

Stage 2 scope:
    - FNO3d with spectral convolutions on 3D signals
    - Same architecture skeleton as FNO1d / FNO2d: lifting -> L blocks of
      [spectral + local + GELU] -> Q + projection head.
    - Weight shapes captured for NeuroIR v0.3.0 export:
        - spectral: (in, out, modes_h, modes_w, modes_d)
        - local:    (out, in)        (PyTorch nn.Linear layout)
    - Forward signature: (batch, h, w, d, in_channels) ->
                          (batch, h, w, d, out_channels).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FNO3dConfig:
    """Configuration for FNO3d. Used both at construction and IR export."""

    in_channels: int = 1
    out_channels: int = 1
    width: int = 64
    modes_h: int = 8
    modes_w: int = 8
    modes_d: int = 8
    n_layers: int = 4
    activation: str = "gelu"
    pad_factor: int = 1
    name: str = "fno3d"


class SpectralConv3d(nn.Module):
    """3D spectral convolution: 3D real FFT -> truncate to a
    (modes_h, modes_w, modes_d) box of low frequencies -> pointwise complex
    linear -> 3D inverse FFT.

    Maintains two trainable weights of shape
    (in_channels, out_channels, modes_h, modes_w, modes_d) (real and imag).
    Ordering matches SpectralConv1d / SpectralConv2d (in, out, modes...).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_h: int,
        modes_w: int,
        modes_d: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.modes_d = modes_d

        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_h,
                modes_w,
                modes_d,
                dtype=torch.float32,
            )
        )
        self.weights_imag = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_h,
                modes_w,
                modes_d,
                dtype=torch.float32,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, h, w, d)
        batch, _, h, w, d = x.shape
        # rfftn over the last three dims gives shape (batch, channels, h, w, d//2+1).
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        # Truncate to (modes_h, modes_w, modes_d) box. The d axis is the
        # rfft axis (size d//2+1); the h, w axes are full size.
        x_ft_trunc = x_ft[..., : self.modes_h, : self.modes_w, : self.modes_d]
        # Complex multiply: (b, in, mh, mw, md) * (in, out, mh, mw, md)
        #                 -> (b, out, mh, mw, md)
        weights = torch.complex(self.weights_real, self.weights_imag)
        out_ft = torch.einsum("bimnp,iomnp->bomnp", x_ft_trunc, weights)
        # Pad back to (b, out, h, w, d//2+1).
        out_ft_padded = F.pad(
            out_ft,
            (
                0,
                (d // 2 + 1) - self.modes_d,  # last axis (d rfft)
                0,
                w - self.modes_w,  # middle axis (w)
                0,
                h - self.modes_h,  # first padded axis (h)
            ),
        )
        out = torch.fft.irfftn(out_ft_padded, s=(h, w, d), dim=(-3, -2, -1))
        return out

    def export_weights(self) -> dict[str, torch.Tensor]:
        """Return weights as a dict for IR export."""
        return {
            "weights_real": self.weights_real.detach().cpu(),
            "weights_imag": self.weights_imag.detach().cpu(),
        }


class FNO3d(nn.Module):
    """3D Fourier Neural Operator.

    Architecture:
        Lifting linear (c -> w) ->
        L blocks of [SpectralConv3d + skip linear + GELU] ->
        Projection Q (w -> w) + GELU + Projection out (w -> out_channels).

    Forward signature: (batch, h, w, d, in_channels) ->
                       (batch, h, w, d, out_channels).
    """

    def __init__(self, config: FNO3dConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = FNO3dConfig(**kwargs)
        self.config = config

        c, w, mh, mw, md, L = (
            config.in_channels,
            config.width,
            config.modes_h,
            config.modes_w,
            config.modes_d,
            config.n_layers,
        )
        self.width = w
        self.modes_h = mh
        self.modes_w = mw
        self.modes_d = md
        self.pad_factor = config.pad_factor

        # Lifting
        self.lift = nn.Linear(c, w)

        # Spectral convs + skip connections
        self.specs = nn.ModuleList(
            [SpectralConv3d(w, w, mh, mw, md) for _ in range(L)]
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
        # x: (batch, h, w, d, in_channels)
        x = self.lift(x)  # (batch, h, w, d, width)
        x = x.permute(0, 4, 1, 2, 3)  # (batch, width, h, w, d)

        # Optionally pad h, w, d to multiples of pad_factor
        if self.pad_factor > 1:
            pf = self.pad_factor
            pad_h = (pf - (x.shape[-3] % pf)) % pf
            pad_w = (pf - (x.shape[-2] % pf)) % pf
            pad_d = (pf - (x.shape[-1] % pf)) % pf
            # F.pad order for last three dims is (left_d, right_d, left_w,
            # right_w, left_h, right_h)
            x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        else:
            pad_h = pad_w = pad_d = 0

        for spec_conv, loc_lin in zip(self.specs, self.locs, strict=True):
            x1 = spec_conv(x)  # (b, w, h, w, d)
            # Local linear along the channel dim only: permute (b, w, h, w, d)
            # -> (b, h, w, d, w), apply Linear on last dim, permute back.
            x2 = loc_lin(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            x = x1 + x2
            x = self.act(x)

        if self.pad_factor > 1 and (pad_h > 0 or pad_w > 0 or pad_d > 0):
            x = x[
                ...,
                : x.shape[-3] - pad_h,
                : x.shape[-2] - pad_w,
                : x.shape[-1] - pad_d,
            ]

        # Project back to (batch, h, w, d, out_channels)
        x = x.permute(0, 2, 3, 4, 1)
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
