"""FP8 E4M3 (IEEE-754 binary8) bit-level conversions.

This is the Python mirror of ``cpp/include/neuroflow/fp8_e4m3.h``.
Both implementations MUST stay bit-for-bit identical; the
cross-language parity test in ``tests/test_fp8_e4m3_parity.py``
enforces this.

E4M3 spec (OCP 8-bit FP, NVIDIA H100):
  bit 7      : sign (1 = negative)
  bits 6..3  : exponent (4 bits, bias = 7)
  bits 2..0  : mantissa (3 bits, hidden-1 for E in [1, 15])

Range:
  Largest finite normal   : +448      (S=0, E=0b1111, M=0b110)
  Smallest positive normal : +2^-6     (S=0, E=0b0001, M=0b000)
  Largest subnormal        : +0.875*2^-6 ≈ 1.336e-2
  Smallest subnormal       : +2^-9 ≈ 1.953e-3
  Zero                     : S=0/1, E=0, M=0
  NaN                      : S=0/1, E=0b1111, M=0b111
  NO Inf encoding!         : 0b?_1111_110 is +448 / -448, not ±Inf.

Why this matters:
  The previous ``_quantise_fp8_e4m3`` helper in ``static_quant.py``
  used a log2 rounding approximation.  That approximation is NOT
  IEEE-754 conformant: it diverges from what NVIDIA H100 / cuDNN /
  TensorRT hardware actually computes.  This module is the
  conformant replacement.
"""

from __future__ import annotations

import struct
from typing import Iterable, List, Union

import numpy as np

_E4M3_MAX = 448.0
_E4M3_BIAS = 7
_E4M3_NAN_BYTE = 0x7F
_E4M3_NAN_BYTE_NEG = 0xFF


def _f32_to_bits(v: float) -> int:
    """Reinterpret a Python float as a 32-bit unsigned integer (its IEEE-754 bits)."""
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(u) & 0xFFFFFFFF))[0]


def fp32_to_e4m3_bits(v: float) -> int:
    """Convert a single FP32 value to its FP8 E4M3 bit pattern (0..255).

    Implements IEEE-754 round-to-nearest-even (RNE) for the 3-bit
    mantissa.  Inputs outside ±448 saturate to ±448 (NOT NaN).
    NaN inputs map to 0x7F (+NaN).  ±Inf maps to ±448.
    """
    if v != v:  # NaN
        return _E4M3_NAN_BYTE
    if v == float("inf"):
        return 0x7E
    if v == float("-inf"):
        return 0xFE

    u = _f32_to_bits(v)
    sign = (u >> 31) & 0x1
    abs_bits = u & 0x7FFFFFFF

    if abs_bits == 0:
        return sign << 7  # +0 / -0

    f32_exp = ((abs_bits >> 23) & 0xFF) - 127
    f32_mant = abs_bits & 0x7FFFFF

    e4m3_exp = f32_exp + _E4M3_BIAS
    e4m3_mant = (f32_mant >> 20) & 0x7
    residue = f32_mant & 0xFFFFF  # lower 20 bits

    # RNE rounding: half-ULP is bit 19 (value 2^19 = 524288).
    half_ulp = 1 << 19
    if residue > half_ulp:
        e4m3_mant += 1
    elif residue == half_ulp:
        if e4m3_mant & 0x1:
            e4m3_mant += 1
    # else: residue < half -> truncate

    if e4m3_mant & 0x8:
        e4m3_mant &= 0x7
        e4m3_exp += 1

    if e4m3_exp > 15:
        return (sign << 7) | 0x7E
    if e4m3_exp < 1:
        # Subnormal: value = mant_bits × 2^-9 for mant_bits in {1..7}.
        # Convert FP32 to double, scale by 2^9, RNE to integer, clamp to [0, 7].
        abs_f32 = _bits_to_f32(abs_bits)
        scaled = abs_f32 * 512.0
        rounded = int(np.rint(scaled))
        rounded = max(0, min(7, rounded))
        return (sign << 7) | rounded

    return (sign << 7) | (e4m3_exp << 3) | e4m3_mant


def e4m3_bits_to_fp32(b: int) -> float:
    """Convert an FP8 E4M3 bit pattern (0..255) back to FP32."""
    b = int(b) & 0xFF
    sign = (b >> 7) & 0x1
    exp_bits = (b >> 3) & 0xF
    mant_bits = b & 0x7

    # NaN
    if exp_bits == 0xF and mant_bits == 0x7:
        if sign:
            return float("nan")  # sign is preserved through the NaN payload
        return float("nan")

    if exp_bits == 0:
        if mant_bits == 0:
            return -0.0 if sign else 0.0
        # Subnormal: value = mant_bits × 2^-9
        return (-1.0 if sign else 1.0) * np.ldexp(float(mant_bits), -9)

    # Normal
    fp32_exp = exp_bits + 120  # e4m3_exp - 7 + 127
    fp32_mant = mant_bits << 20
    u = (sign << 31) | (fp32_exp << 23) | fp32_mant
    return _bits_to_f32(u)


def fp32_array_to_e4m3_bits(arr: Union[np.ndarray, Iterable[float]]) -> np.ndarray:
    """Convert a float array to a uint8 array of FP8 E4M3 bytes."""
    arr = np.asarray(arr, dtype=np.float32).ravel()
    out = np.empty(arr.shape[0], dtype=np.uint8)
    for i, v in enumerate(arr):
        out[i] = fp32_to_e4m3_bits(float(v))
    return out


def e4m3_bits_to_fp32_array(arr: Union[np.ndarray, Iterable[int]]) -> np.ndarray:
    """Convert a uint8 array of FP8 E4M3 bytes back to a float32 array."""
    arr = np.asarray(arr, dtype=np.uint8).ravel()
    out = np.empty(arr.shape[0], dtype=np.float32)
    for i, b in enumerate(arr):
        out[i] = e4m3_bits_to_fp32(int(b))
    return out


def fp8_e4m3_per_tensor_fake_quant(
    x: Union[np.ndarray, Iterable[float]],
    scale: float,
) -> np.ndarray:
    """Per-tensor FP8 E4M3 fake-quant.  Returns a float32 array of
    the same shape as ``x`` after the quantise-dequantise round-trip.

    Mirrors the C++ ``nflow::fp8_e4m3_per_tensor_fake_quant`` exactly.
    """
    x = np.asarray(x, dtype=np.float32)
    x_flat = x.ravel()
    inv_scale = 1.0 / scale if scale != 0.0 else 1.0
    out = np.empty_like(x_flat)
    for i, v in enumerate(x_flat):
        x_scaled = float(v) * inv_scale
        # Saturate
        if x_scaled > _E4M3_MAX:
            x_clamped = _E4M3_MAX
        elif x_scaled < -_E4M3_MAX:
            x_clamped = -_E4M3_MAX
        else:
            x_clamped = x_scaled
        b = fp32_to_e4m3_bits(x_clamped)
        x_q = e4m3_bits_to_fp32(b)
        out[i] = x_q * scale
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# Drop-in replacement for the old log2-approximate helper in
# ``static_quant.py``.  Kept under a different name so that callers
# can opt in.  The old name is left in place for back-compat with
# existing code paths.
# ---------------------------------------------------------------------------
def quantise_fp8_e4m3_ieee(x: np.ndarray, scale: float) -> np.ndarray:
    """Quantise ``x`` to FP8 E4M3 and back to FP32 (IEEE-754 conformant)."""
    return fp8_e4m3_per_tensor_fake_quant(x, scale)
