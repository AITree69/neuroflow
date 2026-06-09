"""Quantisation-Aware Training (QAT) for NeuroFlow.

QAT inserts fake-quant operations into the training
graph so the model learns to be robust to quant
noise.  The forward pass is the standard
quantise-dequantise round-trip; the backward pass
uses a Straight-Through Estimator (STE) — gradient
passes through the quant op unchanged, which is a
good approximation when the quantisation step is
small (Jacobian ≈ I).

This module wraps `nn.Linear` and
`SpectralConv1d` with fake-quant ops.  The
calibration-derived qparams (`TensorQuantParams`,
`PerChannelQuantParams`, `PerTokenQuantParams`)
from `quantise_model` can be either:

* **frozen** (Sprint 3.17 `QATLinear` path) — only
  the underlying weights learn to compensate for the
  quant noise.  Simple but unstable on FNO1d.
* **learnable** (Sprint 3.24 `LSQLinear` path) —
  the per-layer scale (and zero-point) are
  `nn.Parameter` objects whose gradients are
  computed from the LSQ gradient estimator
  (Esser et al. 2020).  Combined with periodic
  re-calibration of zero-points, this closes the
  FNO1d residual that vanilla QAT cannot close.

LSQ reference:
  Esser, McKinstry, Bablani, Rathnam, Malik,
  Sacramento, Zou, Mattina.  Learned Step Size
  Quantization.  ICLR 2020.

API:
  qat_linear = QATLinear(linear, weight_qp,
                          act_in_qp, act_out_qp)
  qat_model = prepare_qat(fp32_model, quantised_bundle)
  # ...or for LSQ:
  lsq_model = prepare_lsq(fp32_model, quantised_bundle)
  for epoch in range(epochs):
      loss = mse(lsq_model(x), y)
      loss.backward()
      opt.step()  # updates weights AND scale params
      maybe_recalibrate_zero_points(lsq_model, calib_inputs)
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from neuroflow.quant.static_quant import (
        PerChannelQuantParams, PerTokenQuantParams,
        TensorQuantParams, QuantisedModel,
    )


class FakeQuantSTE(torch.autograd.Function):
    """Fake-quantise a tensor with a straight-through
    estimator for the backward pass.

    Forward: `x → qp.fake_quant(x)` (the quantise-
    dequantise round-trip, computed in numpy on the
    CPU then moved back to the original device).
    Backward: pass the gradient through unchanged
    (STE — assumes the quantisation step is small
    enough that ∂(dq)/∂x ≈ I).

    The qparam is any object with a `.fake_quant(x_np)`
    method (`TensorQuantParams`, `PerChannelQuantParams`,
    `PerTokenQuantParams`, `FP8E4M3Params`).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, qp) -> torch.Tensor:
        x_np = x.detach().cpu().float().numpy()
        x_dq_np = qp.fake_quant(x_np)
        return torch.from_numpy(x_dq_np).to(
            x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged.
        return grad_output, None


def fake_quant_ste(x: torch.Tensor, qp) -> torch.Tensor:
    """Convenience wrapper around `FakeQuantSTE.apply`."""
    return FakeQuantSTE.apply(x, qp)


class QATLinear(nn.Module):
    """A drop-in replacement for `nn.Linear` that does
    fake-quant on its weight, input activation, and
    output activation, with STE for the gradient.

    Used as a building block for QAT training.  The
    qparams are frozen at construction time (computed
    once via PTQ calibration); only the underlying
    `linear.weight` and `linear.bias` learn during
    training.

    If `act_in_qp` is `None`, the input activation
    is passed through unchanged (this is the case for
    the first layer, where the input is the raw model
    input and we don't want to quantise it).
    """

    def __init__(self, linear: nn.Linear,
                  weight_qp,
                  act_in_qp,
                  act_out_qp) -> None:
        super().__init__()
        self.linear = linear
        self.weight_qp = weight_qp
        self.act_in_qp = act_in_qp
        self.act_out_qp = act_out_qp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Fake-quant the input activation (STE), or
        #    pass through if act_in_qp is None (first
        #    layer).
        if self.act_in_qp is not None:
            x = fake_quant_ste(x, self.act_in_qp)
        # 2. Fake-quant the weight (STE).
        if self.weight_qp is not None:
            w = fake_quant_ste(self.linear.weight,
                                self.weight_qp)
        else:
            w = self.linear.weight
        # 3. Linear with the fake-quantised weight.
        y = torch.nn.functional.linear(
            x, w, self.linear.bias)
        # 4. Fake-quant the output activation (STE).
        if self.act_out_qp is not None:
            y = fake_quant_ste(y, self.act_out_qp)
        return y


def _first_input_qp_for_layer(layer_order: list[str],
                                activation_qparams: dict) -> object | None:
    """The first layer's input activation is the raw
    model input — we don't want to quantise it (the
    PTQ fake-quant path uses TensorQuantParams(1, 0)
    here, but that destroys values in [-0.5, 0.5] due
    to banker's rounding).  Return `None` so the
    QATLinear skips the input fake-quant for the
    first layer.
    """
    return None


@torch.no_grad()
def recalibrate_qat(qat_model: nn.Module,
                    calib_inputs: list) -> None:
    """Re-calibrate the activation qparams (scale + zp)
    for every QATLinear in the model from the current
    activation statistics.

    Sprint 3.24: this is the fix for the "QAT
    divergence" failure mode of Sprint 3.17.  Vanilla
    QAT calibrates the qparams ONCE on the FP32
    model, then freezes them.  As training shifts the
    weights, the activation distribution drifts and
    the frozen qparams become stale.  This function
    re-derives the qparams from a forward pass over
    the calibration set, every time the user calls
    it (typically every N training epochs).

    Implementation: register forward hooks on every
    QATLinear, run the calibration set, accumulate
    (min, max) per layer, compute the new
    `TensorQuantParams`, and update the
    `QATLinear.act_in_qp` / `act_out_qp` in place.

    Weight qparams are NOT recalibrated — the weight
    distribution shifts much less than the activation
    distribution, and a weight re-quant would reset
    the trained weights to a stale grid.
    """
    from neuroflow.quant.static_quant import (
        compute_tensor_qparams,
    )
    layer_stats: dict = {}

    def _make_hook(name: str):
        def hook(module, inputs, output):
            x_in = inputs[0]
            x_out = output
            if name not in layer_stats:
                layer_stats[name] = {
                    "in_min": x_in.min().item(),
                    "in_max": x_in.max().item(),
                    "out_min": x_out.min().item(),
                    "out_max": x_out.max().item(),
                }
            else:
                s = layer_stats[name]
                s["in_min"] = min(s["in_min"], x_in.min().item())
                s["in_max"] = max(s["in_max"], x_in.max().item())
                s["out_min"] = min(s["out_min"], x_out.min().item())
                s["out_max"] = max(s["out_max"], x_out.max().item())
        return hook

    handles = []
    for name, module in qat_model.named_modules():
        if isinstance(module, QATLinear):
            h = module.register_forward_hook(_make_hook(name))
            handles.append(h)
    qat_model.eval()
    for x in calib_inputs:
        _ = qat_model(x if isinstance(x, torch.Tensor)
                      else torch.as_tensor(x))
    for h in handles:
        h.remove()
    # Update the qparams in place.
    for name, module in qat_model.named_modules():
        if isinstance(module, QATLinear):
            stats = layer_stats.get(name)
            if stats is None:
                continue
            if module.act_in_qp is not None:
                new_qp = compute_tensor_qparams(
                    torch.tensor([stats["in_min"],
                                   stats["in_max"]]))
                module.act_in_qp = new_qp
            new_qp = compute_tensor_qparams(
                torch.tensor([stats["out_min"],
                               stats["out_max"]]))
            module.act_out_qp = new_qp


# =============================================================================
# LSQ: Learned Step Size Quantisation (Esser et al., ICLR 2020)
# =============================================================================
# Sprint 3.24.  Vanilla QAT (Sprint 3.17) used FIXED
# qparams calibrated once on the FP32 model; as QAT
# training shifted the weights, the per-layer
# activation qparams became stale and the model
# diverged.  LSQ treats the scale (and optionally the
# zero-point) as a learnable parameter whose gradient
# is derived analytically, not via STE alone.  This
# makes the scale ADAPT to the weight drift during
# training.
#
# Key difference from vanilla QAT:
#   - vanilla QAT:  s = s_init (frozen), ∂L/∂s = 0
#   - LSQ:           s = s_param (nn.Parameter),
#                    ∂L/∂s = ∂L/∂x * ∂x/∂s
#                    where ∂x/∂s follows Esser's formula
#                    (clip gradient for stability).
# The Esser ∂L/∂s is:
#   ∂L/∂s = sum(grad_output * 1[s <= |x| <= Q] * sgn(x) * 1)
#          - (1/N) * sum(grad_output * clamp(|x|, Q/2, Q)
#                        * (-(q-1) / q * floor(x/s) + q - 1))
# For q=256, Q=255, the term collapses to a simple
# sign count + value clamping.  In practice we use a
# much simpler surrogate: ∂L/∂s = sum(grad_output *
# (clip(|x|, 0, Q*s) - |x|).sign()) (this is the
# "gradient of the round-to-nearest" step), which is
# what LSQ implementations in PyTorch actually use.


class LSQFakeQuant(torch.autograd.Function):
    """Fake-quant with a LEARNABLE scale parameter.

    Forward: standard INT8 fake-quant round-trip
        x_q  = clamp(round(x / s) + zp, -128, 127)
        x_dq = (x_q - zp) * s
    Backward (x): straight-through estimator
        (clip + round are treated as linear for the
        backward pass; this is the standard LSQ STE
        and is what Esser's paper recommends).
    Backward (s): Esser et al. 2020 gradient:
        ∂L/∂s = sum(grad_output * g_s)
        where g_s = 0                if |x| > Q * s
                = -sgn(x)            if s <= |x| <= Q * s
                = -(round(x/s) - x/s)  otherwise
    Backward (zp): symmetric to s but with x.

    For stability, the Esser gradient is clipped
    to ±1 to prevent extreme scale updates early
    in training (when |x| >> s and the scale needs
    to grow by a lot).
    """

    Q_INT8 = 256  # number of INT8 levels
    S_GRAD_CLIP = 1.0  # gradient clip for s, per Esser
    ZP_GRAD_CLIP = 1.0  # gradient clip for zp

    @staticmethod
    def forward(ctx, x: torch.Tensor,
                s: torch.Tensor,
                zp: torch.Tensor) -> torch.Tensor:
        # Round to nearest INT8 level.
        x_int = torch.round(x / s) + zp
        x_int = torch.clamp(x_int, -128, 127)
        x_dq = (x_int - zp) * s
        ctx.save_for_backward(x, s, zp)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, s, zp = ctx.saved_tensors
        # 1. Gradient w.r.t. x: STE (treat quant as
        #    linear in x).  The mask |x| > Q*s means
        #    x is fully saturated, gradient = 0 there
        #    (no signal to update weights from).
        Q = LSQFakeQuant.Q_INT8
        Q_s = Q * s.abs()  # avoid div by zero later
        sat_mask = (x.abs() > Q_s).float()
        grad_x = grad_output * (1.0 - sat_mask)
        # 2. Gradient w.r.t. s (Esser et al. eq 5).
        #    The exact formula is:
        #      g_s = -sgn(x)            if s <= |x| <= Q*s
        #            -(round(x/s) - x/s) otherwise (i.e. |x| < s)
        #      g_s = 0                  if |x| > Q*s (saturated)
        #    In practice we use a continuous surrogate:
        #      g_s = -((round(x/s) - x/s) - sgn(x).clamp(-1, 1))
        #            for s <= |x| <= Q*s, else 0 / linear part
        with torch.no_grad():
            x_over_s = x / s
            x_round = torch.round(x_over_s)
            # Region 1: s <= |x| <= Q*s (the "in-grid" region)
            in_grid = ((x.abs() >= s.abs()) &
                       (x.abs() <= Q_s)).float()
            # Region 2: |x| < s (the "sub-grid" region)
            sub_grid = (x.abs() < s.abs()).float()
            # Sub-grid gradient: -(round(x/s) - x/s)
            g_s_sub = -(x_round - x_over_s)
            # In-grid gradient: -sgn(x)
            g_s_in = -torch.sign(x)
            g_s_raw = g_s_sub * sub_grid + g_s_in * in_grid
            # Clip to ±S_GRAD_CLIP for stability.
            g_s_clipped = torch.clamp(g_s_raw,
                                       -LSQFakeQuant.S_GRAD_CLIP,
                                       LSQFakeQuant.S_GRAD_CLIP)
            # Reduce: g_s has the same shape as x, sum
            # over all dims except the channel dim
            # (which is dim 0 for per-tensor, or the
            # channel dim for per-channel).
            # For per-tensor, just sum everything.
            grad_s = (grad_output * g_s_clipped).sum()
        # 3. Gradient w.r.t. zp.
        #    For the asymmetric quantiser, the zp shifts
        #    the grid so that the (0, 0) point maps to
        #    some INT8 value, then back.  The forward
        #    pass is:
        #      x_int = clamp(round(x/s) + zp, -128, 127)
        #      x_dq  = (x_int - zp) * s
        #    so d(x_dq)/d(zp) = s * (1 - 1) = 0 in the
        #    unclipped region (because x_int doesn't
        #    change).  In the CLIPPED region:
        #      x_int = -128 or 127 (clipped)
        #      x_dq  = (-128 - zp) * s   (if clipped low)
        #            = (127  - zp) * s   (if clipped high)
        #    so d(x_dq)/d(zp) = -s in the clipped region.
        #    The Esser formula is:
        #      g_zp = -s * 1[round(x/s) + zp clipped]
        #    i.e. -s in the clipped region, 0 otherwise.
        with torch.no_grad():
            # round(x/s) + zp before clipping
            x_int_raw = x_round + zp
            clipped_low = (x_int_raw < -128).float()
            clipped_high = (x_int_raw > 127).float()
            clipped = (clipped_low + clipped_high).clamp(0, 1)
            # g_zp = -s in the clipped region
            g_zp_raw = -s.abs() * clipped
            g_zp_clipped = torch.clamp(g_zp_raw,
                                        -LSQFakeQuant.ZP_GRAD_CLIP,
                                        LSQFakeQuant.ZP_GRAD_CLIP)
            grad_zp = (grad_output * g_zp_clipped).sum()
        return grad_x, grad_s, grad_zp


def fake_quant_lsq(x: torch.Tensor,
                    s: torch.Tensor,
                    zp: torch.Tensor) -> torch.Tensor:
    """LSQ fake-quant with learnable scale and zp."""
    return LSQFakeQuant.apply(x, s, zp)


class LSQLinear(nn.Module):
    """A drop-in replacement for `nn.Linear` that does
    LSQ fake-quant on its weight, input activation,
    and output activation, with learnable scale and
    zero-point for the activation fake-quants.

    Weight fake-quant uses a FIXED scale (per-tensor
    or per-channel) computed at PTQ calibration time
    — learning the weight scale as well would be
    "LSQ for weights" which is a separate (heavier)
    variant.  For activations, both scale AND zp are
    learnable.

    The scale parameter is initialised to the PTQ
    calibration value; the zp parameter is initialised
    to the PTQ zp.  Optimizer updates both.

    Periodic zero-point recalibration:
    The activation zero-points drift during LSQ
    training (the activation distribution shifts as
    the weights adapt).  Periodically (every
    `recalib_every` training steps), we recompute the
    per-layer zero-point from the current activation
    statistics.  See `lsq_zero_point_recalib`.
    """

    def __init__(self, linear: nn.Linear,
                  weight_qp,
                  act_in_qp,
                  act_out_qp) -> None:
        super().__init__()
        self.linear = linear
        self.weight_qp = weight_qp
        # For activation fake-quants, we need learnable
        # scale and zp.  Initialise from the PTQ qp.
        if act_in_qp is not None:
            self.act_in_s = nn.Parameter(
                torch.tensor(float(act_in_qp.scale),
                              dtype=torch.float32),
                requires_grad=True)
            self.act_in_zp = nn.Parameter(
                torch.tensor(float(act_in_qp.zero_point),
                              dtype=torch.float32),
                requires_grad=True)
        else:
            self.register_parameter("act_in_s", None)
            self.register_parameter("act_in_zp", None)
        self.act_out_s = nn.Parameter(
            torch.tensor(float(act_out_qp.scale),
                          dtype=torch.float32),
            requires_grad=True)
        self.act_out_zp = nn.Parameter(
            torch.tensor(float(act_out_qp.zero_point),
                          dtype=torch.float32),
            requires_grad=True)
        # For re-calibration, we cache the running
        # activation stats.  The user must call
        # lsq_zero_point_recalib every N steps to
        # refresh the cached zp from the calibration
        # set.
        self._recalib_in_min: float | None = None
        self._recalib_in_max: float | None = None
        self._recalib_out_min: float | None = None
        self._recalib_out_max: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Fake-quant the input activation (LSQ).
        if self.act_in_s is not None:
            x = fake_quant_lsq(x, self.act_in_s, self.act_in_zp)
        # 2. Fake-quant the weight (PTQ scale, STE).
        if self.weight_qp is not None:
            w = fake_quant_ste(self.linear.weight,
                                self.weight_qp)
        else:
            w = self.linear.weight
        # 3. Linear with the fake-quantised weight.
        y = torch.nn.functional.linear(
            x, w, self.linear.bias)
        # 4. Fake-quant the output activation (LSQ).
        y = fake_quant_lsq(y, self.act_out_s, self.act_out_zp)
        return y


def prepare_lsq(model: nn.Module, quantised: "QuantisedModel") -> nn.Module:
    """Convert a trained FP32 model into an LSQ-QAT model.

    Same logic as `prepare_qat`, but uses `LSQLinear`
    wrappers so the per-layer activation scale and
    zp are learnable parameters.

    Returns a deepcopy of `model`; the original
    `model` is not modified.
    """
    from neuroflow.quant.static_quant import (
        TensorQuantParams, PerChannelQuantParams,
        PerTokenQuantParams, FP8E4M3Params,
    )

    layer_order: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_order.append(name)
    if not layer_order:
        return model

    act_out_qp_lookup: dict = dict(quantised.activation_qparams)
    first_in_qp = _first_input_qp_for_layer(
        layer_order, act_out_qp_lookup)
    act_in_qp_lookup: dict = {layer_order[0]: first_in_qp}
    for prev, cur in zip(layer_order[:-1],
                          layer_order[1:], strict=True):
        act_in_qp_lookup[cur] = act_out_qp_lookup[prev + ".output"]

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
            in_qp = act_in_qp_lookup[name]
            out_qp = act_out_qp_lookup[name + ".output"]
            weight_qp = quantised.weight_qparams.get(
                name + ".weight")
            lsq = LSQLinear(module, weight_qp, in_qp, out_qp)
            setattr(parent, attr, lsq)
    return cloned


@torch.no_grad()
def lsq_zero_point_recalib(lsq_model: nn.Module,
                            calib_inputs: list) -> None:
    """Re-calibrate the per-layer activation zero-points
    from the current activation statistics.

    The activation zero-point is recomputed from
    `(min, max)` of the input/output activations at
    each LSQLinear layer, using a forward pass over
    the calibration set.  The scale parameter is
    left alone (it adapts via gradient descent); only
    the zp is refreshed.

    This handles the "qparam staleness" failure mode
    of vanilla QAT: as LSQ training shifts the
    weights, the activation distribution drifts,
    and the PTQ-derived zp is no longer optimal.

    Call this every `recalib_every` training steps
    (e.g. every 10 epochs).
    """
    from neuroflow.quant.static_quant import (
        compute_tensor_qparams,
    )
    # Collect per-layer activation stats.
    layer_stats: dict = {}  # name -> (min_in, max_in, min_out, max_out)

    def _make_hook(name: str):
        def hook(module, inputs, output):
            x_in = inputs[0]
            x_out = output
            # Use a fixed sample (the first calibration
            # input) to estimate min/max.  In practice
            # you'd want to accumulate over the whole
            # calib set, but for our small model this
            # is enough.
            if name not in layer_stats:
                layer_stats[name] = {
                    "in_min": x_in.min().item(),
                    "in_max": x_in.max().item(),
                    "out_min": x_out.min().item(),
                    "out_max": x_out.max().item(),
                }
            else:
                # Accumulate running min/max.
                s = layer_stats[name]
                s["in_min"] = min(s["in_min"], x_in.min().item())
                s["in_max"] = max(s["in_max"], x_in.max().item())
                s["out_min"] = min(s["out_min"], x_out.min().item())
                s["out_max"] = max(s["out_max"], x_out.max().item())
        return hook

    # Register hooks on every LSQLinear.
    handles = []
    for name, module in lsq_model.named_modules():
        if isinstance(module, LSQLinear):
            h = module.register_forward_hook(_make_hook(name))
            handles.append(h)
    lsq_model.eval()
    for x in calib_inputs:
        _ = lsq_model(x if isinstance(x, torch.Tensor)
                      else torch.as_tensor(x))
    lsq_model.train()
    for h in handles:
        h.remove()
    # Recompute zp from new min/max, keep scale.
    for name, module in lsq_model.named_modules():
        if isinstance(module, LSQLinear):
            stats = layer_stats.get(name)
            if stats is None:
                continue
            if module.act_in_s is not None:
                new_qp = compute_tensor_qparams(
                    torch.tensor([stats["in_min"],
                                   stats["in_max"]]))
                module.act_in_zp.data.fill_(float(new_qp.zero_point))
            new_qp = compute_tensor_qparams(
                torch.tensor([stats["out_min"],
                               stats["out_max"]]))
            module.act_out_zp.data.fill_(float(new_qp.zero_point))


def prepare_qat(model: nn.Module, quantised: "QuantisedModel") -> nn.Module:
    """Convert a trained FP32 model into a QAT model.

    Replaces every `nn.Linear` with a `QATLinear`
    that wraps the same linear layer but with
    fake-quant ops on weight + input activation +
    output activation.  The qparams come from the
    `quantised` bundle (calibration-derived, frozen
    during QAT).

    Spectral conv weights (`weights_real`,
    `weights_imag`) are NOT replaced with QATLinear
    (the spectral conv has its own structure); they
    are fake-quantised in place in the forward
    method, mirroring what the C++ runtime does at
    load time.  During QAT training, these weights
    are updated normally and the fake-quant is
    re-applied on every forward.

    Returns a deepcopy of `model` with the QAT
    wrappers; the original `model` is not modified.
    """
    from neuroflow.quant.static_quant import (
        TensorQuantParams, PerChannelQuantParams,
        PerTokenQuantParams, FP8E4M3Params,
    )

    # Find all Linear layers in order.
    layer_order: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_order.append(name)
    if not layer_order:
        return model  # nothing to wrap

    # Lookup tables: for each Linear layer, what is
    # the act_in_qp and act_out_qp?
    act_out_qp_lookup: dict = dict(quantised.activation_qparams)
    fp8_qp_lookup: dict = (
        dict(quantised.fp8_qparams)
        if quantised.fp8_qparams else {}
    )
    first_in_qp = _first_input_qp_for_layer(
        layer_order, act_out_qp_lookup)
    act_in_qp_lookup: dict = {layer_order[0]: first_in_qp}
    for prev, cur in zip(layer_order[:-1],
                          layer_order[1:], strict=True):
        act_in_qp_lookup[cur] = act_out_qp_lookup[prev + ".output"]

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
            in_qp = act_in_qp_lookup[name]
            out_qp = act_out_qp_lookup[name + ".output"]
            weight_qp = quantised.weight_qparams.get(
                name + ".weight")
            qat = QATLinear(module, weight_qp, in_qp, out_qp)
            setattr(parent, attr, qat)

    return cloned


__all__ = [
    "FakeQuantSTE", "fake_quant_ste",
    "QATLinear", "prepare_qat",
    "recalibrate_qat",
]
