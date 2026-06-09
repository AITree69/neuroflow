"""Public API for `neuroflow.quant`."""

from __future__ import annotations

from neuroflow.quant.static_quant import (
    INT8_MAX,
    INT8_MIN,
    FakeQuantLinear,
    FP8E4M3Params,
    PerChannelQuantParams,
    PerTokenQuantParams,
    QuantisedModel,
    TensorQuantParams,
    build_fake_quant_model,
    calibrate,
    compute_fp8_e4m3_qparams,
    compute_per_channel_qparams,
    compute_per_token_qparams,
    compute_tensor_qparams,
    quant_to_ir,
    quantise_model,
)

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
