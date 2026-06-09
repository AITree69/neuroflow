"""NeuroFlow INT8 post-training quantisation (PTQ) utilities.

This module implements a *fake-quantisation* (also called
"QAT simulation" in the PyTorch literature) path for the
Stage 2 inference runtime:

  * Weights are stored as INT8 with a per-tensor
    `scale` and `zero_point` (asymmetric).
  * Activations are round-to-INT8 at layer boundaries,
    then dequantised back to FP32 for the actual compute
    (which still happens in FP32).  This is the standard
    "fake-quant" trick used by
    `torch.ao.quantization.default_qconfig`.
  * The C++ runtime applies the same fake-quant round-trip
    around its FP32 ops, so C++ vs PyTorch INT8 parity
    reduces to "do we apply the same quantise / dequantise
    round-trip" — not "is the GEMM INT8".

The net effect is that the model weights are 4x smaller
on disk (INT8 vs FP32) and the inference compute is
within the same FP32 envelope, but with a controlled
quantisation noise floor of `~1 / 2^7 = ~7.8e-3` per
tensor.  The Stage 3 quantised runtime will replace the
FP32 GEMM with INT8 GEMM and INT32 accumulation to
recover the speed and memory wins that this Sprint only
achieves on the weight side.

The scheme is "W8A8" (8-bit weights, 8-bit activations)
with per-tensor scale and zero-point.  Per-channel
quantisation is a Stage 3 extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


# INT8 quantisation range
INT8_MIN = -128
INT8_MAX = 127


@dataclass
class TensorQuantParams:
    """Per-tensor quantisation parameters.

    The round-trip is::

        x_q = clamp(round(x / scale) + zero_point, INT8_MIN, INT8_MAX)
        x_dq = (x_q - zero_point) * scale

    `scale` and `zero_point` are derived from the
    `(min, max)` of the source tensor at calibration
    time::

        scale = (max - min) / (INT8_MAX - INT8_MIN)
        zero_point = clamp(round(qmin - min / scale),
                            INT8_MIN, INT8_MAX)

    A `TensorQuantParams` instance is per-tensor (a
    single `scale` and `zero_point` for the whole
    tensor).  For per-channel quantisation of weights,
    use `PerChannelQuantParams` instead.
    """

    scale: float
    zero_point: int
    bit_width: int = 8

    def quantise(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = x
        x_q = np.clip(
            np.round(x_np / self.scale) + self.zero_point,
            INT8_MIN, INT8_MAX,
        ).astype(np.int8)
        return x_q

    def dequantise(self, x_q: np.ndarray) -> np.ndarray:
        return (x_q.astype(np.float32) - self.zero_point) * self.scale

    def fake_quant(self, x: np.ndarray) -> np.ndarray:
        """Quantise-and-dequantise in one step (the
        "fake-quant" round-trip used at layer boundaries)."""
        return self.dequantise(self.quantise(x))

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "zero_point": int(self.zero_point),
            "bit_width": int(self.bit_width),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TensorQuantParams":
        return cls(
            scale=float(d["scale"]),
            zero_point=int(d["zero_point"]),
            bit_width=int(d.get("bit_width", 8)),
        )


@dataclass
class PerChannelQuantParams:
    """Per-channel INT8 quantisation parameters for a
    weight tensor.

    Used for weight tensors of `nn.Linear`
    (`(out_features, in_features)`) and `SpectralConv1d`
    (`(in, out, modes)` — quantised per `(out,)`).
    Per-channel means each *output* channel gets its
    own `(scale, zero_point)`.

    Attributes:
        scales: (n_channels,) float32 — per-channel scale
        zero_points: (n_channels,) int32 — per-channel zp
        channel_axis: int — the axis that defines
            "channels".  For `nn.Linear` weight
            `(out, in)` it is 0; for `SpectralConv1d`
            weight `(in, out, modes)` it is 1.
        bit_width: int — 8 for INT8.

    The round-trip is::

        for c in range(n_channels):
            x_q[c] = clamp(round(x[c] / scale[c]) + zp[c],
                            INT8_MIN, INT8_MAX)
            x_dq[c] = (x_q[c] - zp[c]) * scale[c]
    """

    scales: np.ndarray  # (n_channels,)
    zero_points: np.ndarray  # (n_channels,) int32
    channel_axis: int = 0
    bit_width: int = 8

    def n_channels(self) -> int:
        return int(self.scales.shape[0])

    def quantise(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = x
        # Move channel_axis to the front, quantise along
        # it, then move back.
        moved = np.moveaxis(x_np, self.channel_axis, 0)
        flat = moved.reshape(self.n_channels(), -1)
        scales = self.scales[:, None]
        zps = self.zero_points[:, None].astype(np.float32)
        x_q = np.clip(
            np.round(flat / scales) + zps,
            INT8_MIN, INT8_MAX,
        ).astype(np.int8)
        x_q = x_q.reshape(moved.shape)
        return np.moveaxis(x_q, 0, self.channel_axis)

    def dequantise(self, x_q: np.ndarray) -> np.ndarray:
        moved = np.moveaxis(x_q, self.channel_axis, 0)
        flat = moved.reshape(self.n_channels(), -1)
        scales = self.scales[:, None].astype(np.float32)
        zps = self.zero_points[:, None].astype(np.float32)
        x_dq = (flat.astype(np.float32) - zps) * scales
        x_dq = x_dq.reshape(moved.shape)
        return np.moveaxis(x_dq, 0, self.channel_axis)

    def fake_quant(self, x: np.ndarray) -> np.ndarray:
        return self.dequantise(self.quantise(x))

    def to_dict(self) -> dict:
        return {
            "scales": self.scales.tolist(),
            "zero_points": [int(z) for z in self.zero_points],
            "channel_axis": int(self.channel_axis),
            "bit_width": int(self.bit_width),
            "per_channel": True,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerChannelQuantParams":
        return cls(
            scales=np.asarray(d["scales"], dtype=np.float32),
            zero_points=np.asarray(d["zero_points"], dtype=np.int32),
            channel_axis=int(d.get("channel_axis", 0)),
            bit_width=int(d.get("bit_width", 8)),
        )


@dataclass
class FP8E4M3Params:
    """FP8 E4M3 (1 sign + 4 exponent + 3 mantissa) INT8
    surrogate quantisation parameters.  The actual
    FP8 E4M3 values are 8 distinct magnitudes per
    2^e bucket plus a few subnormals; for our
    surrogate we approximate by quantising the log2
    of the magnitude to the nearest 1/8 (3-bit
    mantissa).

    Symmetric quantisation: no zero_point, just a
    single scale factor.  The scale is the max-abs
    of the source tensor divided by the E4M3 max
    representable value (448).
    """
    scale: float
    bit_width: int = 8

    def quantise(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = x
        return _quantise_fp8_e4m3(x_np, self.scale)

    def dequantise(self, x_q: np.ndarray) -> np.ndarray:
        return x_q  # already in original scale

    def fake_quant(self, x: np.ndarray) -> np.ndarray:
        return _quantise_fp8_e4m3(x, self.scale) if False else (
            self.dequantise(self.quantise(x)))

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "bit_width": int(self.bit_width),
            "format": "fp8_e4m3",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FP8E4M3Params":
        return cls(scale=float(d["scale"]),
                   bit_width=int(d.get("bit_width", 8)))


@dataclass
class PerTokenQuantParams:
    """Per-token (per-spatial-point) INT8 quantisation
    parameters for an activation tensor.

    Used for the post-activation hidden state of every
    FNO1d layer.  Each spatial point `(n_idx, w_idx)` of
    the `(batch, n, width)` activation tensor gets its
    own `(scale, zero_point)`.  The scale and zero_point
    arrays are stored flat in row-major order:
    `scales[ n_idx * width + w_idx ]`.

    Attributes:
        scales: (n_tokens,) float32 — per-token scale
        zero_points: (n_tokens,) int32 — per-token zp
        width: int — the width dimension of the
            activation (last axis).  Used by the C{++}
            runtime to compute `flat_idx = n * width + w`.
        bit_width: int — 8 for INT8.

    The round-trip is::

        for n_idx in range(n):
            for w_idx in range(width):
                idx = n_idx * width + w_idx
                x_q[..., n_idx, w_idx] =
                    clamp(round(x[..., n_idx, w_idx] /
                                 scales[idx]) +
                          zero_points[idx],
                          INT8_MIN, INT8_MAX)
                x_dq[..., n_idx, w_idx] =
                    (x_q - zero_points[idx]) * scales[idx]
    """

    scales: np.ndarray  # (n_tokens,)
    zero_points: np.ndarray  # (n_tokens,) int32
    width: int = 0
    bit_width: int = 8

    def n_tokens(self) -> int:
        return int(self.scales.shape[0])

    def quantise(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = x
        # x: (batch, n, width) — apply per-token
        # quant along the last two axes.
        if x_np.ndim == 3:
            bsz, n, w = x_np.shape
            assert w == self.width, (
                f"width mismatch: {w} != {self.width}")
            # scales / zero_points are (n * w,)
            scales = self.scales.reshape(n, w)
            zps = self.zero_points.reshape(n, w).astype(np.float32)
            x_q = np.clip(
                np.round(x_np / scales[None, :, :]) + zps[None, :, :],
                INT8_MIN, INT8_MAX,
            ).astype(np.int8)
            return x_q
        raise ValueError(
            f"PerTokenQuantParams expects 3D (batch, n, w) "
            f"input, got {x_np.ndim}D")

    def dequantise(self, x_q: np.ndarray) -> np.ndarray:
        if x_q.ndim == 3:
            bsz, n, w = x_q.shape
            scales = self.scales.reshape(n, w).astype(np.float32)
            zps = self.zero_points.reshape(n, w).astype(np.float32)
            return (x_q.astype(np.float32) - zps[None, :, :]) * \
                scales[None, :, :]
        raise ValueError(
            f"PerTokenQuantParams expects 3D (batch, n, w) "
            f"input, got {x_q.ndim}D")

    def fake_quant(self, x: np.ndarray) -> np.ndarray:
        return self.dequantise(self.quantise(x))

    def to_dict(self) -> dict:
        return {
            "scales": self.scales.tolist(),
            "zero_points": [int(z) for z in self.zero_points],
            "width": int(self.width),
            "bit_width": int(self.bit_width),
            "per_token": True,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerTokenQuantParams":
        return cls(
            scales=np.asarray(d["scales"], dtype=np.float32),
            zero_points=np.asarray(d["zero_points"], dtype=np.int32),
            width=int(d.get("width", 0)),
            bit_width=int(d.get("bit_width", 8)),
        )


def compute_fp8_e4m3_qparams(t: torch.Tensor) -> "FP8E4M3Params":
    """Compute FP8 E4M3 quantisation parameters from a
    tensor.  E4M3 has 1 sign + 4 exponent + 3 mantissa
    bits, range ±448, ~256 distinct values per sign.
    Symmetric quantisation (no zero point).

    The scale is the max-abs of the tensor divided by
    the E4M3 max representable value (448).  The
    round-trip error is bounded by `scale / 2`.
    """
    t_max = float(t.detach().abs().max().item())
    if t_max != t_max or t_max == 0.0:  # NaN or all-zero
        return FP8E4M3Params(scale=1.0)
    scale = t_max / 448.0
    return FP8E4M3Params(scale=float(scale))


# E4M3 (1 sign + 4 exponent + 3 mantissa) representable
# values, computed once at import time.  The
# representable values are 8 distinct magnitudes per
# 2^e bucket (0, 1.mmm × 2^e) plus a few subnormals.  For
# quantise/dequantise purposes, we just need the max
# representable magnitude (448) and the saturation
# behaviour.
_E4M3_MAX = 448.0  # 2^(15-7) * (2 - 2^(-3)) = 256 * 1.875 = 480.  # actually 448
# Recheck: 2^7 * 1.875 = 256 * 1.875 = 480.  But the
# standard E4M3 max is 448 (= 2^8 * 1.75).  Use 448.
_E4M3_MAX = 448.0


def _quantise_fp8_e4m3(x: np.ndarray, scale: float) -> np.ndarray:
    """Symmetric FP8 E4M3 quantise.  Returns the
    dequantised FP8 values in the original scale (NOT
    the integer codes — unlike INT8, FP8 stores the
    values directly scaled)."""
    # Scale to [-448, 448].
    x_scaled = x / scale
    x_scaled = np.clip(x_scaled, -_E4M3_MAX, _E4M3_MAX)
    # Round to nearest representable E4M3 value.
    abs_x = np.abs(x_scaled)
    # For values >= 2^(-6) (normal range), quantise the
    # log2 of the magnitude to the nearest 1/8 (3-bit
    # mantissa).  For values < 2^(-6), subnormal range
    # (we keep as-is in this approximation).
    quantized_abs = np.where(
        abs_x >= 2 ** (-6),
        np.power(2.0, np.round(np.log2(np.maximum(abs_x, 1e-30)) * 8) / 8),
        abs_x
    )
    sign = np.where(x_scaled < 0, -1.0, 1.0)
    # Return in original scale (NOT divided by scale —
    # FP8 stores the value directly, unlike INT8).
    return sign * quantized_abs * scale


def compute_per_token_qparams(
    t: torch.Tensor, width: int,
    percentile: float = 100.0,
) -> PerTokenQuantParams:
    """Compute per-token (per-spatial-point) asymmetric
    INT8 qparams for an activation tensor of shape
    `(batch, n, width)`.

    Args:
        t: 2D `(n, width)` or 3D `(batch, n, width)`
            activation tensor.
        width: the `width` dimension (last axis).
        percentile: when 100.0 (default), use the
            strict `(min, max)` of each point.  When
            < 100.0, use the symmetric
            `[(100 - percentile) / 2, percentile +
            (100 - percentile) / 2]` percentile of each
            point.  Percentile-based ranges are more
            robust to outliers and small calibration
            sets than strict min/max.

    The output is a flat `(n * width,)` vector in
    row-major order over `(n, width)`, ready for the
    C{++} runtime to look up by `n_idx * width +
    w_idx`.

    Robust to zp saturation (see
    `compute_tensor_qparams`).
    """
    t_np = t.detach().cpu().float().numpy()
    if t_np.ndim == 3:
        bsz, n, w = t_np.shape
        assert w == width
        # Reduce over the batch axis to get per-spatial-
        # point (n, w) mins / maxs.
        if percentile < 100.0:
            lo_p = (100.0 - percentile) / 2.0
            hi_p = 100.0 - lo_p
            mins = np.percentile(t_np, lo_p, axis=0)
            maxs = np.percentile(t_np, hi_p, axis=0)
        else:
            mins = t_np.min(axis=0)
            maxs = t_np.max(axis=0)
    elif t_np.ndim == 2:
        n, w = t_np.shape
        assert w == width
        if percentile < 100.0:
            lo_p = (100.0 - percentile) / 2.0
            hi_p = 100.0 - lo_p
            mins = np.percentile(t_np, lo_p, axis=0)
            maxs = np.percentile(t_np, hi_p, axis=0)
        else:
            mins = t_np
            maxs = t_np
    else:
        raise ValueError(
            f"PerTokenQuantParams expects 2D or 3D "
            f"input, got {t_np.ndim}D")
    spans = maxs - mins
    scales = np.where(
        spans > 0,
        spans / (INT8_MAX - INT8_MIN),
        1.0,
    ).astype(np.float32)
    raw_zp = np.round(INT8_MIN - mins / scales)
    zero_points = np.clip(raw_zp, INT8_MIN, INT8_MAX).astype(np.int32)
    # Re-derive scale for any (n, w) point where zp was
    # clipped (asymmetric range that doesn't fit).
    clipped_lo = zero_points == INT8_MIN
    clipped_hi = zero_points == INT8_MAX
    scales = np.where(
        clipped_lo & (maxs > 0),
        maxs / (INT8_MAX - INT8_MIN),
        scales,
    ).astype(np.float32)
    scales = np.where(
        clipped_hi & (mins < 0),
        -mins / (INT8_MAX - INT8_MIN),
        scales,
    ).astype(np.float32)
    return PerTokenQuantParams(
        scales=scales.flatten(),
        zero_points=zero_points.flatten(),
        width=width,
    )


def compute_tensor_qparams(t: torch.Tensor) -> TensorQuantParams:
    """Compute per-tensor asymmetric INT8 quantisation
    parameters from a tensor.

    The tensor is reduced to its `(min, max)` range; the
    resulting `scale` and `zero_point` make the round-trip
    `fake_quant` saturate at the INT8 extremes when the
    input spans the full tensor range.

    Standard asymmetric INT8 with `qmin = -128`,
    `qmax = 127`::

        scale     = (max - min) / (qmax - qmin)
        zero_point = round(qmin - min / scale)
                   = round(-128 - min / scale)

    For a symmetric `[-1, 1]` range this gives
    `scale ≈ 1/127` and `zero_point = 0`; for
    `[0, 1]` it gives `scale ≈ 1/255` and
    `zero_point = -128`.

    Robust to zp saturation: when the data range is
    very asymmetric (e.g. all-negative or all-positive)
    the naive zp formula may round outside
    `[-128, 127]`.  When this happens, we re-derive
    the scale from the *clipped* range so the
    round-trip is exact at both ends.
    """
    t_min = float(t.detach().cpu().min().item())
    t_max = float(t.detach().cpu().max().item())
    if t_max == t_min or t_min != t_min or t_max != t_max:
        # Constant tensor or NaN range — pick a no-op
        # scale=1 / zero_point=0 that round-trips
        # correctly (dequantise is identity for any int).
        return TensorQuantParams(scale=1.0, zero_point=0)
    scale = (t_max - t_min) / (INT8_MAX - INT8_MIN)
    zero_point = int(np.clip(
        np.round(INT8_MIN - t_min / scale), INT8_MIN, INT8_MAX
    ))
    if zero_point != np.round(INT8_MIN - t_min / scale):
        # zp was clipped — re-derive scale to fit the
        # data exactly in the asymmetric INT8 range.
        # zp = INT8_MIN, so min' = 0; max' = (qmax - qmin) * scale
        # But the actual max is t_max; we want
        # scale' = (t_max - t_min_adjusted) / 255
        # where t_min_adjusted = (INT8_MIN - zp) * scale.
        if zero_point == INT8_MIN:
            t_min_adj = 0.0
            scale = (t_max - t_min_adj) / (INT8_MAX - INT8_MIN)
        else:  # zero_point == INT8_MAX
            t_max_adj = 0.0
            scale = (t_max_adj - t_min) / (INT8_MAX - INT8_MIN)
    return TensorQuantParams(scale=float(scale),
                              zero_point=zero_point)


def compute_per_channel_qparams(
    t: torch.Tensor, channel_axis: int = 0,
) -> PerChannelQuantParams:
    """Compute per-channel asymmetric INT8 qparams for
    a weight tensor.  Each slice along `channel_axis`
    gets its own `(scale, zero_point)`.

    For `nn.Linear` weights `(out, in)` pass
    `channel_axis=0` (per-output quantisation, the
    standard TensorRT / ONNX pattern).  For
    `SpectralConv1d` weights `(in, out, modes)` pass
    `channel_axis=1`.

    Robust to zp saturation (see `compute_tensor_qparams`).
    """
    t_np = t.detach().cpu().float().numpy()
    moved = np.moveaxis(t_np, channel_axis, 0)
    n_channels = moved.shape[0]
    flat = moved.reshape(n_channels, -1)
    mins = flat.min(axis=1)
    maxs = flat.max(axis=1)
    spans = maxs - mins
    scales = np.where(
        spans > 0,
        spans / (INT8_MAX - INT8_MIN),
        1.0,
    ).astype(np.float32)
    raw_zp = np.round(INT8_MIN - mins / scales)
    zero_points = np.clip(raw_zp, INT8_MIN, INT8_MAX).astype(np.int32)
    # Re-derive scale for any channels where zp was
    # clipped (asymmetric range that doesn't fit).
    clipped_lo = zero_points == INT8_MIN
    clipped_hi = zero_points == INT8_MAX
    # Channel where zp was clipped to INT8_MIN: the
    # source range effectively starts at 0; recompute
    # scale from (0, max).
    scales = np.where(
        clipped_lo & (maxs > 0),
        maxs / (INT8_MAX - INT8_MIN),
        scales,
    ).astype(np.float32)
    # Channel where zp was clipped to INT8_MAX: the
    # source range effectively ends at 0; recompute
    # scale from (min, 0).
    scales = np.where(
        clipped_hi & (mins < 0),
        -mins / (INT8_MAX - INT8_MIN),
        scales,
    ).astype(np.float32)
    return PerChannelQuantParams(
        scales=scales,
        zero_points=zero_points,
        channel_axis=channel_axis,
    )


# ---------------------------------------------------------------------------
# Calibrate — collect activation statistics
# ---------------------------------------------------------------------------


class _CalibrationHook:
    """A forward hook that records the output of an
    `nn.Module` for calibration."""

    def __init__(self) -> None:
        self.observed: list[torch.Tensor] = []

    def __call__(self, module: nn.Module,
                  inputs: tuple, output: torch.Tensor) -> None:
        self.observed.append(output.detach().cpu())


def calibrate(model: nn.Module,
              calib_inputs: Sequence[torch.Tensor],
              per_token: bool = False,
              percentile: float = 100.0,
              ema_decay: float | None = None,
              ) -> dict[str, TensorQuantParams | "PerTokenQuantParams"]:
    """Run the model on a small calibration set and
    compute INT8 quantisation parameters for every
    Linear / Conv1d layer's output.

    The hook is attached to every Linear layer in the
    model; the returned dict maps `layer_name →
    TensorQuantParams` (or `PerTokenQuantParams` if
    `per_token=True`) for the *activation* of that
    layer.  Weight quantisation parameters are derived
    from the layer's `.weight` directly via
    `quantise_model`.

    Args:
        per_token: when True, compute per-spatial-point
            `(scale, zero_point)` for each activation.
        percentile: when < 100, use this percentile of
            the observed distribution as the upper /
            lower bounds (more robust to outliers than
            strict min/max).  E.g. `percentile=99.5`
            uses the 0.25 / 99.5 percentiles.
        ema_decay: when set (e.g. 0.9), apply an
            exponential moving average to the
            per-sample min/max statistics across the
            calibration batch.  This is the standard
            "running min/max" trick used by TensorRT.
            When None (default), use the global
            min/max across all calibration samples
            (the v0.17.0 behaviour).
    """
    activation_qparams: dict[str, TensorQuantParams | "PerTokenQuantParams"] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = _CalibrationHook()
            handle = module.register_forward_hook(hook)
            hooks.append((name, hook, handle))
    model.eval()
    with torch.no_grad():
        for x in calib_inputs:
            model(x)
    for handle in hooks:
        handle[2].remove()
    for name, hook, _ in hooks:
        if per_token and len(hook.observed) > 0:
            # Per-token (per-spatial-point) calibration.
            # Concatenate all observed activations along
            # the batch axis, then compute per-(n, w)
            # qparams.  Reduce is over the *batch* axis,
            # keeping (n, w) as the per-token axes.
            stacked = torch.cat(hook.observed, dim=0)  # (B*calib, n, w)
            bsz, n, w = stacked.shape
            if ema_decay is not None:
                # Running (min, max) per (n, w) point
                # with EMA.  Initialise from the first
                # sample.
                running_min = stacked[0].numpy().astype(np.float32)
                running_max = stacked[0].numpy().astype(np.float32)
                for i in range(1, bsz):
                    sample = stacked[i].numpy().astype(np.float32)
                    running_min = ema_decay * running_min + \
                        (1.0 - ema_decay) * sample
                    running_max = ema_decay * running_max + \
                        (1.0 - ema_decay) * sample
                # The EMA gives a smoothed statistic, not
                # a true min/max.  Treat it as the
                # "expected range" and add a small
                # margin.
                margin = np.abs(running_max - running_min).max() * 0.05 + 1e-6
                mins = running_min - margin
                maxs = running_max + margin
            elif percentile < 100.0:
                lo_p = (100.0 - percentile) / 2.0
                hi_p = 100.0 - lo_p
                mins = np.percentile(stacked.numpy(), lo_p, axis=0)
                maxs = np.percentile(stacked.numpy(), hi_p, axis=0)
            else:
                mins = stacked.min(dim=0).values.numpy()
                maxs = stacked.max(dim=0).values.numpy()
            spans = maxs - mins
            scales = np.where(
                spans > 0,
                spans / (INT8_MAX - INT8_MIN),
                1.0,
            ).astype(np.float32)
            raw_zp = np.round(INT8_MIN - mins / scales)
            zero_points = np.clip(raw_zp, INT8_MIN, INT8_MAX).astype(np.int32)
            clipped_lo = zero_points == INT8_MIN
            clipped_hi = zero_points == INT8_MAX
            scales = np.where(
                clipped_lo & (maxs > 0),
                maxs / (INT8_MAX - INT8_MIN),
                scales,
            ).astype(np.float32)
            scales = np.where(
                clipped_hi & (mins < 0),
                -mins / (INT8_MAX - INT8_MIN),
                scales,
            ).astype(np.float32)
            activation_qparams[name + ".output"] = PerTokenQuantParams(
                scales=scales.flatten(),
                zero_points=zero_points.flatten(),
                width=w,
            )
        else:
            all_observed = torch.cat([t.flatten() for t in hook.observed])
            if percentile < 100.0:
                lo_p = (100.0 - percentile) / 2.0
                hi_p = 100.0 - lo_p
                p = np.percentile(all_observed.numpy(),
                                   [lo_p, hi_p])
                mins = float(p[0])
                maxs = float(p[1])
                spans = maxs - mins
                scale = spans / (INT8_MAX - INT8_MIN) if spans > 0 else 1.0
                raw_zp = np.round(INT8_MIN - mins / scale)
                zp = int(np.clip(raw_zp, INT8_MIN, INT8_MAX))
                if zp == INT8_MIN and maxs > 0:
                    scale = maxs / (INT8_MAX - INT8_MIN)
                elif zp == INT8_MAX and mins < 0:
                    scale = -mins / (INT8_MAX - INT8_MIN)
                activation_qparams[name + ".output"] = TensorQuantParams(
                    scale=float(scale),
                    zero_point=zp,
                )
            else:
                activation_qparams[name + ".output"] = compute_tensor_qparams(all_observed)
    return activation_qparams


def _collect_per_layer_tensors(
    model: nn.Module,
    calib_inputs: Sequence[torch.Tensor],
) -> dict[str, list[torch.Tensor]]:
    """Run the model on the calibration set and return
    the raw per-layer output tensors (one list per
    `nn.Linear` layer, indexed by layer name +
    ".output").

    Used as a primitive by both `calibrate` (which
    derives INT8 qparams from these tensors) and by
    the FP8 path in `quantise_model` (which derives
    FP8 E4M3 qparams from the same tensors)."""
    observed: dict[str, list[torch.Tensor]] = {}
    handles = []

    class _Hook:
        def __init__(self) -> None:
            self.tensors: list[torch.Tensor] = []

        def __call__(self, module, inputs, output):
            self.tensors.append(output.detach().cpu())

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = _Hook()
            handle = module.register_forward_hook(hook)
            handles.append((name, hook, handle))
    model.eval()
    with torch.no_grad():
        for x in calib_inputs:
            model(x)
    for _, _, handle in handles:
        handle.remove()
    for name, hook, _ in handles:
        observed[name + ".output"] = hook.tensors
    return observed


# ---------------------------------------------------------------------------
# Quantise model — produce INT8 weights + activation qparams
# ---------------------------------------------------------------------------


@dataclass
class QuantisedModel:
    """A quantised model bundle.

    Attributes:
        int8_weights: dict of {weight_name: int8 array}
            for every quantisable weight.
        weight_qparams: {weight_name: TensorQuantParams |
            PerChannelQuantParams} — per-tensor or
            per-channel scheme per weight.
        activation_qparams: {layer_name: TensorQuantParams}
            for the Linear layer outputs (calibration
            derived, always per-tensor in this Sprint).
        fp8_qparams: {layer_name: FP8E4M3Params} — when
            `fp8_activations=True` was used, this holds
            the FP8 E4M3 qparams for every activation.
            None / empty when INT8 is used.
        per_channel_weights: bool — whether weights
            use the per-channel scheme.
    """

    int8_weights: dict[str, np.ndarray]
    weight_qparams: dict[str, TensorQuantParams | "PerChannelQuantParams"]
    activation_qparams: dict[str, TensorQuantParams]
    fp8_qparams: dict[str, "FP8E4M3Params"] = None  # type: ignore
    per_channel_weights: bool = False

    def dequantise_weight(self, name: str) -> np.ndarray:
        q = self.int8_weights[name]
        return self.weight_qparams[name].dequantise(q)

    def to_dict(self) -> dict:
        return {
            "int8_weights": {
                k: v.tolist() for k, v in self.int8_weights.items()
            },
            "weight_qparams": {
                k: v.to_dict() for k, v in self.weight_qparams.items()
            },
            "activation_qparams": {
                k: v.to_dict() for k, v in self.activation_qparams.items()
            },
            "fp8_qparams": {
                k: v.to_dict() for k, v in self.fp8_qparams.items()
            } if self.fp8_qparams else {},
            "per_channel_weights": bool(self.per_channel_weights),
        }


def _channel_axis_for_weight(name: str, t: torch.Tensor) -> int:
    """Heuristic: pick the per-channel axis for a weight
    tensor based on its name and shape.

    `nn.Linear.weight` has shape `(out, in)` →
    `channel_axis=0` (per-output).  `SpectralConv1d`
    weights are `(in, out, modes)` → `channel_axis=1`
    (per-output-of-the-spectral-conv).  Everything else
    defaults to `0` (the leading axis).
    """
    if name.endswith(".weight") and t.ndim == 2:
        return 0
    if ("specs." in name or "weights_real" in name or "weights_imag" in name) \
            and t.ndim == 3:
        return 1
    return 0


def quantise_model(model: nn.Module,
                    calib_inputs: Sequence[torch.Tensor],
                    per_channel_weights: bool = False,
                    per_token_activations: bool = False,
                    percentile: float = 100.0,
                    ema_decay: float | None = None,
                    fp8_activations: bool = False,
                    ) -> QuantisedModel:
    """Quantise a model to INT8 (W8A8 fake-quant) and
    return the bundle.

    Args:
        model: the trained PyTorch model.
        calib_inputs: a small calibration set (typically
            8-32 held-out inputs).
        per_channel_weights: when True, quantise every
            weight tensor with a per-output-channel
            `(scale, zero_point)`.
        per_token_activations: when True, quantise
            every activation with a per-spatial-point
            `(scale, zero_point)`.
        percentile: when < 100, use this percentile of
            the observed distribution as the upper /
            lower bounds.  E.g. `percentile=99.5` is
            more robust to outliers than strict min/max
            and avoids the over-aggressive per-point
            calibration that Sprint 3.11 documented.
        ema_decay: when set (e.g. 0.9), apply an
            exponential moving average to the
            per-sample min/max statistics across the
            calibration batch.  The standard TensorRT
            "running min/max" trick.
        fp8_activations: when True, also compute FP8
            E4M3 qparams for every activation layer
            (in addition to the INT8 qparams).  These
            are stored in `QuantisedModel.fp8_qparams`
            and exported as NIRQ kind=3 entries.  The
            C++ runtime uses these via
            `FNO1d::EnableFP8Activation`.

    The bundle is meant to be exported alongside the IR
    in v0.15.0; the C++ runtime reads the bundle and
    applies the same fake-quant round-trip.
    """
    activation_qparams = calibrate(
        model, calib_inputs, per_token=per_token_activations,
        percentile=percentile, ema_decay=ema_decay,
    )
    int8_weights: dict[str, np.ndarray] = {}
    weight_qparams: dict[str, TensorQuantParams | PerChannelQuantParams] = {}
    sd = model.state_dict()
    for name, t in sd.items():
        if per_channel_weights and t.ndim >= 1 and name.endswith((".weight", "weights_real", "weights_imag")):
            axis = _channel_axis_for_weight(name, t)
            qp = compute_per_channel_qparams(t, channel_axis=axis)
        else:
            qp = compute_tensor_qparams(t)
        qp_np = qp.quantise(t)
        int8_weights[name] = qp_np
        weight_qparams[name] = qp
    # Optional FP8 E4M3 qparams for activations.
    # We re-run a small calibration pass to capture the
    # per-layer activation distributions; the per-layer
    # FP8 qparams are derived from the same per-layer
    # tensors that INT8 calibration already collected.
    # To keep this lightweight we recompute them via a
    # second pass with the same hooks, then snap each
    # per-layer activation tensor to FP8 E4M3.
    fp8_qparams: dict[str, FP8E4M3Params] = {}
    if fp8_activations:
        per_layer_tensors = _collect_per_layer_tensors(model, calib_inputs)
        for name, tensors in per_layer_tensors.items():
            cat = torch.cat([t.flatten() for t in tensors])
            fp8_qparams[name] = compute_fp8_e4m3_qparams(cat)
    return QuantisedModel(
        int8_weights=int8_weights,
        weight_qparams=weight_qparams,
        activation_qparams=activation_qparams,
        fp8_qparams=fp8_qparams,
        per_channel_weights=bool(per_channel_weights),
    )


# ---------------------------------------------------------------------------
# Fake-quantised forward — used as a Python reference
# ---------------------------------------------------------------------------


class FakeQuantLinear(nn.Module):
    """A drop-in replacement for `nn.Linear` that does
    fake-quant on its weight and its input activation,
    then runs the standard FP32 matmul.

    The output is then fake-quantised a second time using
    the layer's output quantisation parameters (matching
    the C++ runtime's behaviour).

    The weight fake-quant uses the supplied weight qparam
    (per-tensor or per-channel).  The activation fake-quant
    uses the supplied act_in_qp / act_out_qp, which may
    be per-tensor (`TensorQuantParams`) or per-token
    (`PerTokenQuantParams`).

    The C++ v0.16.0 / v0.17.0 runtime dequantises weights
    with per-tensor / per-channel qparams at load time
    and applies per-token / per-tensor activation fake-
    quant at every layer's boundary.  This wrapper
    matches the same operations in Python.
    """

    def __init__(self, linear: nn.Linear,
                  act_in_qp: TensorQuantParams | "PerTokenQuantParams"
                              | "FP8E4M3Params",
                  act_out_qp: TensorQuantParams | "PerTokenQuantParams"
                              | "FP8E4M3Params",
                  weight_qp: TensorQuantParams | "PerChannelQuantParams"
                              | None = None) -> None:
        super().__init__()
        self.linear = linear
        self.act_in_qp = act_in_qp
        self.act_out_qp = act_out_qp
        self.weight_qp = weight_qp
        self._apply_weight_fake_quant()
        self._cached_x_q: torch.Tensor | None = None

    def _apply_weight_fake_quant(self) -> None:
        if self.weight_qp is None:
            return
        with torch.no_grad():
            w = self.linear.weight.detach().cpu().float().numpy()
            w_dq = self.weight_qp.fake_quant(w)
            self.linear.weight.data = (
                torch.from_numpy(w_dq).to(
                    self.linear.weight.device, dtype=self.linear.weight.dtype)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Fake-quant the input activation.
        x_np = x.detach().cpu().float().numpy()
        x_dq = self.act_in_qp.fake_quant(x_np)
        x = torch.from_numpy(x_dq).to(x.device, dtype=x.dtype)
        # 2. Linear (FP32 compute, but with the
        #    fake-quantised input and weight).
        y = self.linear(x)
        # 3. Fake-quant the output activation.
        y_np = y.detach().cpu().float().numpy()
        y_dq = self.act_out_qp.fake_quant(y_np)
        return torch.from_numpy(y_dq).to(y.device, dtype=y.dtype)


def build_fake_quant_model(model: nn.Module,
                            quantised: QuantisedModel,
                            use_fp8_activations: bool = False,
                            ) -> nn.Module:
    """Build a clone of `model` whose Linear layers are
    replaced with `FakeQuantLinear` wrappers that apply
    the same fake-quant round-trip the C++ runtime does.

    Used as a Python reference for the C++ INT8 parity
    check.  The weight fake-quant uses the
    per-tensor or per-channel weight qparams from
    `quantised`; the activation fake-quant uses the
    per-tensor activation qparams (INT8) — or, when
    `use_fp8_activations=True`, the FP8 E4M3 qparams
    from `quantised.fp8_qparams` (matching the C++
    v0.21.0 runtime's `EnableFP8Activation` path).

    Spectral conv weights (`weights_real`,
    `weights_imag`) are also fake-quantised in place
    when a corresponding qparam is present in the
    bundle.  This matches the C++ v0.16.0 runtime
    which dequantises these tensors at load time.
    """
    layer_order: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_order.append(name)

    int8_out_qp_lookup: dict[str, TensorQuantParams] = dict(
        quantised.activation_qparams
    )
    fp8_qp_lookup: dict[str, FP8E4M3Params] = (
        dict(quantised.fp8_qparams)
        if (use_fp8_activations and quantised.fp8_qparams)
        else {}
    )
    first_input_qp = TensorQuantParams(scale=1.0, zero_point=0)
    first_input_fp8 = FP8E4M3Params(scale=1.0) if fp8_qp_lookup else None
    in_int8_qp_lookup: dict[str, TensorQuantParams] = {
        layer_order[0]: first_input_qp
    }
    in_fp8_qp_lookup: dict[str, FP8E4M3Params] = (
        {layer_order[0]: first_input_fp8} if first_input_fp8 else {}
    )
    for prev, cur in zip(layer_order[:-1], layer_order[1:], strict=True):
        in_int8_qp_lookup[cur] = int8_out_qp_lookup[prev + ".output"]
        if in_fp8_qp_lookup:
            in_fp8_qp_lookup[cur] = fp8_qp_lookup[prev + ".output"]

    import copy
    cloned = copy.deepcopy(model)
    for name, module in list(cloned.named_modules()):
        if isinstance(module, nn.Linear) and name in layer_order:
            parent_path = name.rsplit(".", 1)
            if len(parent_path) == 1:
                parent = cloned
                attr = parent_path[0]
            else:
                parent = cloned
                for p in parent_path[0].split("."):
                    parent = getattr(parent, p)
                attr = parent_path[1]
            in_int8 = in_int8_qp_lookup[name]
            out_int8 = int8_out_qp_lookup[name + ".output"]
            weight_qp = quantised.weight_qparams.get(name + ".weight")
            if in_fp8_qp_lookup:
                in_qp = in_fp8_qp_lookup[name]
                out_qp = fp8_qp_lookup[name + ".output"]
            else:
                in_qp = in_int8
                out_qp = out_int8
            fq = FakeQuantLinear(module, in_qp, out_qp,
                                 weight_qp=weight_qp)
            setattr(parent, attr, fq)

    # Fake-quant the spectral conv weights in place.
    from neuroflow.nn.fno import SpectralConv1d
    for name, module in list(cloned.named_modules()):
        if isinstance(module, SpectralConv1d):
            for w_attr, suffix in [
                ("weights_real", "weights_real"),
                ("weights_imag", "weights_imag"),
            ]:
                full_name = f"{name}.{suffix}"
                qp = quantised.weight_qparams.get(full_name)
                if qp is None:
                    continue
                with torch.no_grad():
                    w = getattr(module, w_attr).detach().cpu().float().numpy()
                    w_dq = qp.fake_quant(w)
                    getattr(module, w_attr).data = (
                        torch.from_numpy(w_dq).to(
                            getattr(module, w_attr).device,
                            dtype=getattr(module, w_attr).dtype)
                    )
    return cloned


__all__ = [
    "INT8_MAX", "INT8_MIN",
    "TensorQuantParams", "PerChannelQuantParams",
    "PerTokenQuantParams", "FP8E4M3Params", "QuantisedModel",
    "compute_tensor_qparams", "compute_per_channel_qparams",
    "compute_per_token_qparams", "compute_fp8_e4m3_qparams",
    "calibrate",
    "quantise_model", "build_fake_quant_model", "FakeQuantLinear",
    "quant_to_ir",
]


# ---------------------------------------------------------------------------
# Convert QuantisedModel to the v0.15.0 IR `quant` block
# ---------------------------------------------------------------------------


def quant_to_ir(qm: QuantisedModel) -> dict:
    """Convert a `QuantisedModel` to the `quant` dict
    shape expected by `NeuroIRSpec.quant` (see
    `export.py` for the binary layout).

    Returns a dict with:
      - `enabled: True` (so the export actually writes
        the NIRQ block)
      - `qparams: {name: {scale, zero_point}}` for every
        weight tensor and every calibrated activation.
    """
    qparams: dict[str, dict] = {}
    for name, qp in qm.weight_qparams.items():
        qparams[name] = qp.to_dict()
    for name, qp in qm.activation_qparams.items():
        qparams[name] = qp.to_dict()
    if qm.fp8_qparams:
        for name, qp in qm.fp8_qparams.items():
            qparams[name] = qp.to_dict()
    return {"enabled": True, "qparams": qparams}
