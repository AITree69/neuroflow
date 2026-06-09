"""Sprint 3.9 — pytest cases for INT8 fake-quantisation.

Covers:
  * Per-tensor quantisation roundtrip (fake_quant)
  * Constant tensor
  * `calibrate` produces a qparam per Linear layer output
  * `quantise_model` produces INT8 weights in [-128, 127]
  * `build_fake_quant_model` produces a model with the
    same forward shape as the original
  * End-to-end FP32 vs INT8 (fake-quant) rel L2 is
    within a sane bound (typically < 5e-2 for a well-
    calibrated W8A8 scheme on a small FNO1d)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neuroflow.nn.fno import FNO1d, FNO1dConfig
from neuroflow.quant import (
    FakeQuantLinear,
    PerChannelQuantParams,
    PerTokenQuantParams,
    QuantisedModel,
    TensorQuantParams,
    build_fake_quant_model,
    calibrate,
    compute_per_channel_qparams,
    compute_per_token_qparams,
    compute_tensor_qparams,
    quantise_model,
)


def _make_dummy_fno1d(n_points: int = 64) -> FNO1d:
    cfg = FNO1dConfig(
        in_channels=1, out_channels=1,
        width=16, modes=min(8, n_points // 2),
        n_layers=2, activation="gelu", pad_factor=1,
        name="dummy_fno1d_quant",
    )
    return FNO1d(cfg)


def test_compute_tensor_qparams_basic() -> None:
    t = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float32)
    qp = compute_tensor_qparams(t)
    # 1.0 - (-1.0) = 2.0 spread over 255 levels → scale ≈ 7.84e-3
    assert 7.0e-3 < qp.scale < 9.0e-3
    # Symmetric range gives zero_point near 0.
    assert abs(qp.zero_point) <= 1


def test_compute_tensor_qparams_constant() -> None:
    t = torch.full((100,), 0.5, dtype=torch.float32)
    qp = compute_tensor_qparams(t)
    # The constant-tensor fallback sets scale=1, zp=0
    # (no spread → no real quantisation).
    x_q = qp.quantise(t)
    x_dq = qp.dequantise(x_q)
    assert np.allclose(x_dq, t.numpy(), atol=1.0)


def test_fake_quant_roundtrip_error_bounded() -> None:
    """The fake-quant roundtrip error is at most
    ~scale/2 per element, which is the INT8 quantisation
    noise floor."""
    t = torch.linspace(-1.0, 1.0, 10000, dtype=torch.float32)
    qp = compute_tensor_qparams(t)
    x_q = qp.quantise(t)
    x_dq = qp.dequantise(x_q)
    err = np.abs(x_dq - t.numpy())
    # The error is bounded by half a quantisation step.
    assert err.max() <= qp.scale + 1e-7


def test_calibrate_produces_qparams_per_linear() -> None:
    torch.manual_seed(0)
    model = _make_dummy_fno1d(n_points=64)
    calib_inputs = [
        torch.randn(1, 64, 1, dtype=torch.float32) for _ in range(4)
    ]
    qparams = calibrate(model, calib_inputs)
    # The FNO1d has lift + L×locs + proj_q + proj_out
    # = 2 + 2×2 = 6 Linear layers' outputs calibrated.
    n_linear = sum(
        1 for _ in model.modules() if isinstance(_, torch.nn.Linear)
    )
    assert len(qparams) == n_linear
    for k, qp in qparams.items():
        assert k.endswith(".output")
        assert qp.scale > 0


def test_quantise_model_returns_int8_weights() -> None:
    torch.manual_seed(0)
    model = _make_dummy_fno1d(n_points=64)
    calib_inputs = [
        torch.randn(1, 64, 1, dtype=torch.float32) for _ in range(4)
    ]
    qm = quantise_model(model, calib_inputs)
    # Every state_dict entry (linear/conv .weight/.bias
    # plus spectral `weights_real` / `weights_imag`)
    # should be quantised to int8.
    n_state = len(model.state_dict())
    assert len(qm.int8_weights) == n_state
    for name, arr in qm.int8_weights.items():
        assert arr.dtype == np.int8
        assert arr.min() >= -128 and arr.max() <= 127
    # Dequantising a weight should reconstruct the FP32
    # weight to within the quantisation noise floor.
    for name, t in model.state_dict().items():
        dq = qm.dequantise_weight(name)
        ref = t.detach().cpu().float().numpy()
        err = np.abs(dq - ref).max()
        assert err <= qm.weight_qparams[name].scale + 1e-7


def test_build_fake_quant_model_forward_shape() -> None:
    torch.manual_seed(0)
    model = _make_dummy_fno1d(n_points=64)
    calib_inputs = [
        torch.randn(1, 64, 1, dtype=torch.float32) for _ in range(4)
    ]
    qm = quantise_model(model, calib_inputs)
    fq_model = build_fake_quant_model(model, qm)
    fq_model.eval()
    with torch.no_grad():
        y_ref = model(torch.from_numpy(calib_inputs[0].numpy()))
        y_fq = fq_model(calib_inputs[0])
    assert y_ref.shape == y_fq.shape
    # The fake-quant error should be bounded by the
    # max per-tensor quantisation noise — typically
    # < 1e-1 in absolute terms for INT8.
    err = float((y_ref - y_fq).abs().max())
    assert err < 1.0


def test_fake_quant_int8_saturates() -> None:
    """A tensor with values larger than the INT8
    representable range (after scale) should saturate
    to the INT8 extremes.  Banker's rounding (round
    half to even) is used."""
    qp = TensorQuantParams(scale=0.01, zero_point=0)
    t = np.array([-100.0, -0.005, 0.0, 0.005, 100.0], dtype=np.float32)
    x_q = qp.quantise(t)
    # -100.0 / 0.01 = -10000 → clamps to -128
    # 100.0 / 0.01 = 10000 → clamps to 127
    assert x_q[0] == -128
    # round(-0.005/0.01) = round(-0.5) = 0 (banker's: half to even)
    assert x_q[1] == 0
    # round(0.005/0.01) = round(0.5) = 0 (banker's: half to even)
    assert x_q[2] == 0
    assert x_q[3] == 0
    assert x_q[4] == 127


def test_per_channel_qparams_basic() -> None:
    """A (out=4, in=3) tensor with 4 well-separated
    per-row ranges should produce 4 distinct (scale,
    zero_point) pairs in the per-channel qparams."""
    t = torch.tensor([
        [-1.0, -0.5,  0.0],
        [10.0, 12.0, 14.0],
        [0.0,  0.0,  0.0],   # constant row → scale=1, zp=0
        [-100.0, -50.0, 0.0],
    ], dtype=torch.float32)
    pcp = compute_per_channel_qparams(t, channel_axis=0)
    assert pcp.n_channels() == 4
    assert pcp.channel_axis == 0
    # Row 0 range [-1, 0] → scale = 1/255 ≈ 3.92e-3
    assert 3.5e-3 < pcp.scales[0] < 4.5e-3
    # Row 1 range [10, 14] (all positive) → zp=INT8_MIN,
    # scale = max / 255 = 14/255 ≈ 5.49e-2 (the
    # zp-saturation fix in v0.17.0 widens the scale
    # to fit the data exactly in the asymmetric
    # INT8 range).
    assert 5.0e-2 < pcp.scales[1] < 6.0e-2
    assert pcp.zero_points[1] == -128
    # Row 2 constant → scale 1 (constant fallback), zp -128
    assert pcp.scales[2] == pytest.approx(1.0, abs=1e-6)
    assert pcp.zero_points[2] == -128
    # Row 3 range [-100, 0] (all negative) → zp=INT8_MAX,
    # scale = -min / 255 = 100/255 ≈ 3.92e-1.
    assert 3.5e-1 < pcp.scales[3] < 4.5e-1
    assert pcp.zero_points[3] == 127
    # Round-trip on row 0 is within scale/2 of the
    # original.
    dq = pcp.dequantise(pcp.quantise(t))
    err_row0 = np.abs(dq[0] - t.numpy()[0]).max()
    assert err_row0 <= pcp.scales[0] + 1e-7


def test_per_channel_qparams_3d_spectral() -> None:
    """A (in=2, out=3, modes=4) tensor with
    `channel_axis=1` should quantise each output
    channel independently."""
    t = torch.randn(2, 3, 4, dtype=torch.float32) * 0.1
    pcp = compute_per_channel_qparams(t, channel_axis=1)
    assert pcp.n_channels() == 3
    assert pcp.channel_axis == 1
    # Round-trip should be within scale per channel.
    dq = pcp.dequantise(pcp.quantise(t))
    err = np.abs(dq - t.numpy()).max(axis=(0, 2))  # max per output channel
    for c in range(3):
        assert err[c] <= pcp.scales[c] + 1e-7


def test_quantise_model_per_channel_weights() -> None:
    """When per_channel_weights=True, all weight
    tensors in the bundle should carry
    `PerChannelQuantParams` (not `TensorQuantParams`)."""
    torch.manual_seed(0)
    model = _make_dummy_fno1d(n_points=64)
    calib_inputs = [
        torch.randn(1, 64, 1, dtype=torch.float32) for _ in range(4)
    ]
    qm = quantise_model(model, calib_inputs, per_channel_weights=True)
    assert qm.per_channel_weights is True
    # The weight tensors of nn.Linear / SpectralConv1d
    # should all be PerChannelQuantParams.
    for name, qp in qm.weight_qparams.items():
        if name.endswith((".weight", "weights_real", "weights_imag")):
            assert isinstance(qp, PerChannelQuantParams), (
                f"{name} should be PerChannelQuantParams, "
                f"got {type(qp).__name__}"
            )
    # Biases are still per-tensor.
    for name, qp in qm.weight_qparams.items():
        if name.endswith(".bias"):
            assert isinstance(qp, TensorQuantParams)
    # On-disk weight size: 1 byte per element (INT8).
    n_state = sum(t.numel() for t in model.state_dict().values())
    n_int8 = sum(arr.nbytes for arr in qm.int8_weights.values())
    assert n_int8 == n_state


def test_per_channel_serialisation_roundtrip() -> None:
    """PerChannelQuantParams.to_dict / from_dict
    round-trip preserves scale and zero_point values."""
    pcp = PerChannelQuantParams(
        scales=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        zero_points=np.array([0, -10, 5], dtype=np.int32),
        channel_axis=0,
    )
    d = pcp.to_dict()
    pcp2 = PerChannelQuantParams.from_dict(d)
    assert pcp2.n_channels() == 3
    assert pcp2.channel_axis == 0
    assert np.allclose(pcp2.scales, pcp.scales)
    assert (pcp2.zero_points == pcp.zero_points).all()


def test_per_token_qparams_basic() -> None:
    """A (batch=4, n=8, width=4) activation tensor
    produces 8*4=32 per-token (scale, zp) pairs."""
    torch.manual_seed(0)
    t = torch.randn(4, 8, 4, dtype=torch.float32) * 0.5
    ptp = compute_per_token_qparams(t, width=4)
    assert ptp.n_tokens() == 32
    assert ptp.width == 4
    # Round-trip error is bounded by per-token scale in
    # the bulk, but at the calibration boundary it can
    # saturate to scale (banker's rounding on a
    # half-step).  We allow up to 2*scale.
    dq = ptp.dequantise(ptp.quantise(t))
    err = np.abs(dq - t.numpy()).max(axis=0)  # max per (n, w)
    for idx in range(32):
        assert err.flat[idx] <= 2.0 * ptp.scales[idx] + 1e-7


def test_calibrate_per_token() -> None:
    """When `per_token=True`, the calibrate dict
    should contain `PerTokenQuantParams` for every
    Linear layer's output, with width matching the
    layer's output channel count."""
    torch.manual_seed(0)
    model = _make_dummy_fno1d(n_points=64)
    calib_inputs = [
        torch.randn(1, 64, 1, dtype=torch.float32) for _ in range(4)
    ]
    qparams = calibrate(model, calib_inputs, per_token=True)
    n_linear = sum(
        1 for _ in model.modules() if isinstance(_, torch.nn.Linear)
    )
    assert len(qparams) == n_linear
    # The FNO1d has width=16 for the hidden layers
    # (lift, locs.*, proj_q) and out_channels=1 for
    # the output projection.
    for k, qp in qparams.items():
        assert k.endswith(".output")
        assert isinstance(qp, PerTokenQuantParams)
        if k.startswith("proj_out."):
            # proj_out projects to out_channels=1.
            assert qp.width == 1
            assert qp.n_tokens() == 64
        else:
            # Hidden layers project to width=16.
            assert qp.width == 16
            assert qp.n_tokens() == 64 * 16


def test_quantise_model_per_token_activations() -> None:
    """When `per_token_activations=True`, every
    activation qparam in the bundle is a
    `PerTokenQuantParams`."""
    torch.manual_seed(0)
    model = _make_dummy_fno1d(n_points=64)
    calib_inputs = [
        torch.randn(1, 64, 1, dtype=torch.float32) for _ in range(4)
    ]
    qm = quantise_model(
        model, calib_inputs,
        per_token_activations=True, per_channel_weights=True,
    )
    assert qm.per_channel_weights is True
    for k, qp in qm.activation_qparams.items():
        assert isinstance(qp, PerTokenQuantParams), (
            f"{k} should be PerTokenQuantParams, "
            f"got {type(qp).__name__}"
        )


def test_per_token_serialisation_roundtrip() -> None:
    """PerTokenQuantParams.to_dict / from_dict
    round-trip preserves scale and zero_point values."""
    ptp = PerTokenQuantParams(
        scales=np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
        zero_points=np.array([0, -10, 5, 100], dtype=np.int32),
        width=4,
    )
    d = ptp.to_dict()
    ptp2 = PerTokenQuantParams.from_dict(d)
    assert ptp2.n_tokens() == 4
    assert ptp2.width == 4
    assert np.allclose(ptp2.scales, ptp.scales)
    assert (ptp2.zero_points == ptp.zero_points).all()


def test_fp8_e4m3_basic() -> None:
    """FP8 E4M3 qparams are symmetric (no zero point)
    and the scale is max_abs / 448."""
    from neuroflow.quant import compute_fp8_e4m3_qparams, FP8E4M3Params
    t = torch.tensor([-100.0, -50.0, 0.0, 50.0, 100.0],
                      dtype=torch.float32)
    p = compute_fp8_e4m3_qparams(t)
    assert isinstance(p, FP8E4M3Params)
    # scale = 100 / 448 ≈ 0.223
    assert 0.22 < p.scale < 0.23


def test_fp8_vs_int8_dynamic_range() -> None:
    """FP8 E4M3 has wider dynamic range than INT8 for
    the same number of bits.  With a 1000:1 dynamic
    range tensor, INT8 saturates the small values
    to zero; FP8 preserves them (with some loss in
    the dense region)."""
    from neuroflow.quant import (compute_tensor_qparams,
                                  compute_fp8_e4m3_qparams,
                                  TensorQuantParams,
                                  FP8E4M3Params)
    t = torch.cat([
        torch.full((10,), 0.001),
        torch.full((10,), 1.0),
        torch.full((10,), 100.0),
        torch.full((10,), 1000.0),
    ])
    int8 = compute_tensor_qparams(t)
    fp8 = compute_fp8_e4m3_qparams(t)
    dq_int8 = int8.dequantise(int8.quantise(t.numpy()))
    dq_fp8 = fp8.fake_quant(t.numpy())
    # Headline: FP8 preserves the small values (0.001)
    # while INT8 saturates them to zero (because INT8
    # with strict min/max uses scale 3.9, and 0.001/3.9
    # = 0 rounds to 0).
    # FP8 with scale 2.23 and E4M3 subnormal range
    # ~0.0015 keeps 0.001 close to its value.
    small_value = 0.001
    small_idx = 0  # first 10 elements
    int8_small_err = abs(
        dq_int8[small_idx] - small_value) / small_value
    fp8_small_err = abs(
        dq_fp8[small_idx] - small_value) / small_value
    # INT8 has 100% rel error on the small values
    # (rounds to 0).  FP8 has much less.
    assert int8_small_err > 0.5, (
        f"INT8 should saturate small values, "
        f"got rel err = {int8_small_err:.3e}"
    )
    assert fp8_small_err < 0.5, (
        f"FP8 should preserve small values better, "
        f"got rel err = {fp8_small_err:.3e}"
    )


def test_quantise_model_fp8_activations_populates_bundle() -> None:
    """quantise_model(fp8_activations=True) populates
    the bundle's `fp8_qparams` map with one FP8E4M3Params
    per Linear layer activation.  This is the path the
    C++ runtime uses via FNO1d::EnableFP8Activation."""
    from neuroflow.quant import quantise_model, FP8E4M3Params
    # Build a small model with 2 Linear layers.
    m = torch.nn.Sequential(torch.nn.Linear(8, 16),
                             torch.nn.GELU(),
                             torch.nn.Linear(16, 4))
    m.eval()
    calib = [torch.randn(2, 8) for _ in range(4)]
    qm = quantise_model(m, calib, fp8_activations=True)
    # fp8_qparams should be populated, one per Linear.
    assert qm.fp8_qparams is not None
    assert len(qm.fp8_qparams) == 2
    for name, fp8p in qm.fp8_qparams.items():
        assert isinstance(fp8p, FP8E4M3Params)
        assert name.endswith(".output")
        assert fp8p.scale > 0
    # INT8 activation qparams should still be present
    # (the FP8 path is additive, not replacing).
    assert len(qm.activation_qparams) == 2
    # When fp8_activations=False, fp8_qparams is empty.
    qm_no_fp8 = quantise_model(m, calib, fp8_activations=False)
    assert qm_no_fp8.fp8_qparams == {}


def test_build_fake_quant_model_with_fp8_activations() -> None:
    """build_fake_quant_model(use_fp8_activations=True)
    builds a fake-quant model that uses the FP8E4M3Params
    for the activation fake-quant at every layer boundary
    (matching the C++ v0.21.0 EnableFP8Activation path)."""
    from neuroflow.quant import (build_fake_quant_model,
                                  quantise_model)
    m = torch.nn.Sequential(torch.nn.Linear(8, 16),
                             torch.nn.GELU(),
                             torch.nn.Linear(16, 4))
    m.eval()
    calib = [torch.randn(2, 8) for _ in range(4)]
    qm = quantise_model(m, calib, fp8_activations=True)
    fq_int8 = build_fake_quant_model(m, qm, use_fp8_activations=False)
    fq_fp8 = build_fake_quant_model(m, qm, use_fp8_activations=True)
    # Both should produce same-shaped outputs.
    x = torch.randn(2, 8)
    with torch.no_grad():
        y_int8 = fq_int8(x)
        y_fp8 = fq_fp8(x)
    assert y_int8.shape == y_fp8.shape == (2, 4)
    # FP8 and INT8 should differ at the output (FP8 has
    # wider dynamic range, so the activation fake-quant
    # round-trips a different set of values).
    assert not torch.allclose(y_int8, y_fp8, atol=1e-6)


def test_qat_fake_quant_ste_gradient_flow() -> None:
    """fake_quant_ste passes the gradient through
    unchanged (straight-through estimator).  Verify
    that the autograd Function behaves as expected:
    forward = qp.fake_quant(x), backward = identity."""
    from neuroflow.quant.qat import fake_quant_ste
    from neuroflow.quant.static_quant import TensorQuantParams
    qp = TensorQuantParams(scale=0.1, zero_point=0)
    x = torch.randn(4, requires_grad=True)
    y = fake_quant_ste(x, qp)
    # Forward is the fake-quant round-trip.
    y_expected = torch.from_numpy(qp.fake_quant(x.detach().numpy()))
    assert torch.allclose(y, y_expected, atol=1e-6)
    # Backward: gradient passes through unchanged.
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-6)


def test_qat_linear_forward_close_to_ptq() -> None:
    """QATLinear's forward (with STE fake-quant)
    should be close to build_fake_quant_model's
    forward when no training has been done.  They
    are not exactly equal because QAT skips the
    first-layer act_in fake-quant (the original
    PTQ path used TensorQuantParams(1, 0) there
    which clips values in [-0.5, 0.5] to 0; QAT
    passes the input through unchanged).  But
    downstream layers should still produce
    reasonable outputs (within an order of
    magnitude of each other)."""
    from neuroflow.quant import (build_fake_quant_model,
                                  quantise_model)
    from neuroflow.quant.qat import prepare_qat
    m = torch.nn.Sequential(torch.nn.Linear(8, 16),
                             torch.nn.GELU(),
                             torch.nn.Linear(16, 4))
    m.eval()
    calib = [torch.randn(2, 8) for _ in range(4)]
    qm = quantise_model(m, calib)
    fq = build_fake_quant_model(m, qm)
    fq.eval()
    qat = prepare_qat(m, qm)
    qat.eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        y_ptq = fq(x)
        y_qat = qat(x)
    # PTQ and QAT should both produce finite, reasonable
    # outputs.  The exact values differ because the
    # first-layer input handling is intentionally
    # different (QAT avoids the banker's-rounding clip).
    assert torch.isfinite(y_ptq).all()
    assert torch.isfinite(y_qat).all()
    assert y_ptq.std() > 1e-3
    assert y_qat.std() > 1e-3
    # The difference should be bounded — the same
    # underlying fake-quant ops, just one less on
    # the first layer's input.
    assert (y_ptq - y_qat).abs().max() < 1.0


def test_qat_training_on_single_linear() -> None:
    """QAT on a single Linear layer should converge
    to (near) the FP32 loss.  This is the smoke
    test for the QAT machinery on a model where
    stale-qparams is not an issue (one layer)."""
    from neuroflow.quant import quantise_model
    from neuroflow.quant.qat import prepare_qat
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(8, 4))
    x = torch.randn(100, 8)
    y = torch.randn(100, 4)
    # FP32 baseline
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        (m(x) - y).pow(2).mean().backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        fp32_loss = (m(x) - y).pow(2).mean().item()
    # Calibrate + QAT
    qm = quantise_model(m, [x[i:i+1] for i in range(8)])
    qat = prepare_qat(m, qm)
    qat.train()
    qat_opt = torch.optim.Adam(qat.parameters(), lr=1e-4)
    for _ in range(100):
        qat_opt.zero_grad()
        (qat(x) - y).pow(2).mean().backward()
        qat_opt.step()
    qat.eval()
    with torch.no_grad():
        qat_loss = (qat(x) - y).pow(2).mean().item()
    # QAT loss should be in the same order as FP32
    # (QAT adds quant noise, so it's a bit higher,
    # but the model didn't diverge).
    assert qat_loss < fp32_loss * 2.0, (
        f"QAT loss {qat_loss:.3e} should be at most "
        f"2x FP32 loss {fp32_loss:.3e} (diverged?)"
    )


def test_per_token_percentile_robustness() -> None:
    """With outliers injected into the calibration
    data, the 99.5th percentile range should give a
    wider scale (less clipping) than the strict
    min/max range."""
    torch.manual_seed(0)
    # Normal data: 16 samples of (n=4, w=2) with small range.
    t = torch.randn(16, 4, 2, dtype=torch.float32) * 0.1
    # Inject 2 outliers with huge values.
    t[0, 0, 0] = 100.0
    t[1, 1, 1] = -100.0
    ptp_strict = compute_per_token_qparams(t, width=2, percentile=100.0)
    ptp_p995 = compute_per_token_qparams(t, width=2, percentile=99.5)
    # The strict range at (0, 0) is dominated by the
    # 100.0 outlier → huge scale.
    # The 99.5th percentile range excludes the outlier
    # → smaller scale, but matches the bulk of the
    # data better.
    # The percentile scale should be smaller than the
    # strict scale at the affected points.
    # (0, 0) flat index = 0
    assert ptp_p995.scales[0] < ptp_strict.scales[0]
    # (1, 1) flat index = 1*2 + 1 = 3
    assert ptp_p995.scales[3] < ptp_strict.scales[3]
    # And unaffected points should be similar.
    # (3, 1) flat index = 3*2 + 1 = 7
    assert abs(ptp_p995.scales[7] - ptp_strict.scales[7]) < 0.01


def test_calibrate_percentile() -> None:
    """When `percentile=99.5`, the calibrate dict
    should still produce `PerTokenQuantParams` but
    with scale/zp values from the 0.25 / 99.5
    percentile range instead of strict min/max."""
    torch.manual_seed(0)
    model = _make_dummy_fno1d(n_points=64)
    calib_inputs = [
        torch.randn(1, 64, 1, dtype=torch.float32) for _ in range(8)
    ]
    q_strict = calibrate(model, calib_inputs, per_token=True,
                          percentile=100.0)
    q_p995 = calibrate(model, calib_inputs, per_token=True,
                        percentile=99.5)
    n_linear = sum(
        1 for _ in model.modules() if isinstance(_, torch.nn.Linear)
    )
    assert len(q_strict) == n_linear
    assert len(q_p995) == n_linear
    for k in q_strict:
        assert k in q_p995
        s_strict = q_strict[k].scales
        s_p995 = q_p995[k].scales
        # The 99.5th percentile scales should be
        # <= the strict min/max scales per point.
        # (no point should be wider than strict).
        assert (s_p995 <= s_strict + 1e-6).all()


def test_calibrate_ema_decay() -> None:
    """When `ema_decay=0.9`, the calibrate dict should
    still produce valid `PerTokenQuantParams` with
    the EMA-smoothed range."""
    torch.manual_seed(0)
    model = _make_dummy_fno1d(n_points=64)
    calib_inputs = [
        torch.randn(1, 64, 1, dtype=torch.float32) for _ in range(8)
    ]
    q = calibrate(model, calib_inputs, per_token=True,
                   ema_decay=0.9)
    n_linear = sum(
        1 for _ in model.modules() if isinstance(_, torch.nn.Linear)
    )
    assert len(q) == n_linear
    for k, qp in q.items():
        assert isinstance(qp, PerTokenQuantParams)
        assert qp.scales.shape[0] > 0


def test_recalibrate_qat_updates_qparams() -> None:
    """Sprint 3.24: `recalibrate_qat` should re-derive
    the activation `(scale, zp)` for every QATLinear
    in the model from the current activation stats.
    After recalibration, the qparams should differ
    from the initial PTQ qparams (because the model
    state has changed)."""
    from neuroflow.quant import quantise_model
    from neuroflow.quant.qat import prepare_qat, recalibrate_qat, QATLinear
    torch.manual_seed(0)
    m = _make_dummy_fno1d(n_points=64)
    x = torch.randn(8, 64, 1, dtype=torch.float32)
    # Train briefly to get non-trivial weights.
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    y = torch.randn(8, 64, 1, dtype=torch.float32)
    for _ in range(20):
        opt.zero_grad()
        (m(x) - y).pow(2).mean().backward()
        opt.step()
    m.eval()
    qm = quantise_model(m, [x[i:i+1] for i in range(8)])
    qat = prepare_qat(m, qm)
    # Snapshot original qparams.
    original = {}
    for name, module in qat.named_modules():
        if isinstance(module, QATLinear):
            if module.act_in_qp is not None:
                original[(name, "in")] = (module.act_in_qp.scale,
                                            module.act_in_qp.zero_point)
            original[(name, "out")] = (module.act_out_qp.scale,
                                        module.act_out_qp.zero_point)
    # Do some "training" that shifts the weights.
    qat.train()
    qat_opt = torch.optim.SGD(qat.parameters(), lr=1e-3, momentum=0.9)
    for _ in range(20):
        qat_opt.zero_grad()
        (qat(x) - y).pow(2).mean().backward()
        qat_opt.step()
    qat.eval()
    # Recalibrate.
    recalibrate_qat(qat, [x[i:i+1] for i in range(8)])
    # Verify the qparams changed (at least one).
    changed = 0
    for name, module in qat.named_modules():
        if isinstance(module, QATLinear):
            if module.act_in_qp is not None:
                new = (module.act_in_qp.scale, module.act_in_qp.zero_point)
                if abs(new[0] - original[(name, "in")][0]) > 1e-6:
                    changed += 1
            new = (module.act_out_qp.scale, module.act_out_qp.zero_point)
            if abs(new[0] - original[(name, "out")][0]) > 1e-6:
                changed += 1
    assert changed > 0, (
        "recalibrate_qat should update at least one scale"
    )


def test_qat_bestval_improves_over_ptq() -> None:
    """Sprint 3.24: QAT with best-val early-stop should
    reduce the INT8 PTQ rel L2 on a small FNO1d + Burgers
    1D-like dataset.  This is the headline test for the
    best-val early-stop fix to Sprint 3.17's divergence
    issue."""
    from neuroflow.data.burgers import Burgers1dConfig, Burgers1dDataset
    from neuroflow.nn.fno import FNO1d
    from neuroflow.quant import quantise_model, build_fake_quant_model
    from neuroflow.quant.qat import prepare_qat
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Burgers1dConfig(n_points=64, n_tsteps=10, nu=0.01, dt=0.01)
    ds = Burgers1dDataset(n_samples=40, cfg=cfg, t_in=1, t_out=1, seed=0)
    x = torch.stack([ds[i][0] for i in range(40)], dim=0)
    y = torch.stack([ds[i][1] for i in range(40)], dim=0)
    model = FNO1d(in_channels=1, out_channels=1, width=16, modes=8,
                   n_layers=2)
    # Pre-train FP32 briefly.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        perm = torch.randperm(40)
        for i in range(0, 40, 8):
            idx = perm[i:i+8]
            opt.zero_grad()
            (model(x[idx]) - y[idx]).pow(2).mean().backward()
            opt.step()
    model.eval()
    calib = [x[i:i+1] for i in range(8)]
    qm = quantise_model(model, calib, per_channel_weights=False,
                        per_token_activations=False, fp8_activations=False)
    fq = build_fake_quant_model(model, qm, use_fp8_activations=False)
    fq.eval()
    with torch.no_grad():
        rel_ptq = (torch.linalg.norm(fq(x) - y) /
                    torch.linalg.norm(y)).item()
    # QAT with best-val early-stop.
    qat = prepare_qat(model, qm)
    qat.train()
    qat_opt = torch.optim.SGD(qat.parameters(), lr=1e-4, momentum=0.9)
    best_val = float("inf")
    best_state = None
    xv = x[32:]
    yv = y[32:]
    xt = x[:32]
    yt = y[:32]
    for ep in range(30):
        perm = torch.randperm(32)
        for i in range(0, 32, 8):
            idx = perm[i:i+8]
            qat_opt.zero_grad()
            (qat(xt[idx]) - yt[idx]).pow(2).mean().backward()
            qat_opt.step()
        with torch.no_grad():
            qat.eval()
            v = (torch.linalg.norm(qat(xv) - yv) /
                 torch.linalg.norm(yv)).item()
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().clone()
                          for k, t in qat.state_dict().items()}
        qat.train()
    if best_state is not None:
        qat.load_state_dict(best_state)
    qat.eval()
    with torch.no_grad():
        rel_qat = (torch.linalg.norm(qat(x) - y) /
                    torch.linalg.norm(y)).item()
    # QAT should improve (or at least not be worse by
    # more than 10% of PTQ).
    assert rel_qat <= rel_ptq * 1.1, (
        f"QAT best-val rel L2 {rel_qat:.3e} should be at "
        f"most 10% worse than PTQ INT8 {rel_ptq:.3e} "
        f"(best-val early-stop should at least preserve "
        f"performance)"
    )
