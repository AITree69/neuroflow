"""Sprint 3.30 — FP8 E4M3 IEEE-754 bit-level conversion tests.

These tests verify the Python ``neuroflow.quant.fp8_e4m3`` module
against the C++ implementation in
``cpp/include/neuroflow/fp8_e4m3.h``.  The two implementations
MUST stay bit-for-bit identical; this file enforces that.

If the optional ``neuroflow_cpp`` C++ extension is built, an
additional cross-language parity test runs; otherwise it is
skipped with a clear message.
"""

# *** DLL search-path setup must happen before numpy / scipy
# imports, because those packages themselves load DLLs on Windows.
# In particular, importing `neuroflow_cpp.pyd` requires libstdc++-6
# to be discoverable.  Do this BEFORE the rest of the imports.
from __future__ import annotations

import math
import os
import struct
import sys
from pathlib import Path

# On Windows, the MinGW-built ``neuroflow_cpp.pyd`` depends on
# libstdc++-6.dll and friends, which are NOT in the default DLL
# search path.  Pre-register the MinGW ``bin/`` directory so the
# import doesn't fail with "DLL load failed".
if sys.platform == "win32":
    _mingw_bin = Path(r"C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin")
    if _mingw_bin.exists():
        os.add_dll_directory(str(_mingw_bin))

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

# Try to import the C++ extension at module load time.  If this
# fails, the cross-language parity test will be skipped.
_NEUROFLOW_CPP = None
_NEUROFLOW_CPP_FP8 = False
os.write(2, b"[test_fp8_e4m3 module load] attempting import...\n")
try:
    sys.path.insert(0, str(_PROJECT_ROOT / "cpp" / "build_pybind"))
    import neuroflow_cpp as _NEUROFLOW_CPP_MOD
    os.write(2, f"[test_fp8_e4m3 module load] imported: {_NEUROFLOW_CPP_MOD}\n".encode())
    os.write(2, f"[test_fp8_e4m3 module load] has fp8_e4m3_to_bits: "
                f"{hasattr(_NEUROFLOW_CPP_MOD, 'fp8_e4m3_to_bits')}\n".encode())
    if hasattr(_NEUROFLOW_CPP_MOD, "fp8_e4m3_to_bits"):
        _NEUROFLOW_CPP = _NEUROFLOW_CPP_MOD
        _NEUROFLOW_CPP_FP8 = True
except ImportError as _e:
    import traceback
    os.write(2, f"[test_fp8_e4m3 module load] ImportError: {_e!r}\n".encode())
    traceback.print_exc(file=sys.stderr)

from neuroflow.quant.fp8_e4m3 import (
    e4m3_bits_to_fp32,
    e4m3_bits_to_fp32_array,
    fp32_array_to_e4m3_bits,
    fp32_to_e4m3_bits,
    fp8_e4m3_per_tensor_fake_quant,
)


# ---------------------------------------------------------------------------
# Reference table for the special values
# ---------------------------------------------------------------------------

def test_zero_roundtrip() -> None:
    """+0 and -0 round-trip exactly, and the sign is preserved."""
    assert fp32_to_e4m3_bits(0.0) == 0x00
    assert fp32_to_e4m3_bits(-0.0) == 0x80
    v_pos = e4m3_bits_to_fp32(0x00)
    v_neg = e4m3_bits_to_fp32(0x80)
    assert v_pos == 0.0 and not math.copysign(1.0, v_pos) < 0
    assert v_neg == 0.0 and math.copysign(1.0, v_neg) < 0


def test_unit_values() -> None:
    """±1.0 round-trip exactly (1.000_bin × 2^0, E=7, M=0)."""
    assert fp32_to_e4m3_bits(1.0) == 0x38
    assert fp32_to_e4m3_bits(-1.0) == 0xB8
    assert e4m3_bits_to_fp32(0x38) == 1.0
    assert e4m3_bits_to_fp32(0xB8) == -1.0


def test_max_finite() -> None:
    """±448 are the largest finite values (E4M3 has NO Inf)."""
    assert fp32_to_e4m3_bits(448.0) == 0x7E
    assert fp32_to_e4m3_bits(-448.0) == 0xFE
    # 0x7E and 0xFE are finite (NOT Inf)
    assert math.isfinite(e4m3_bits_to_fp32(0x7E))
    assert math.isfinite(e4m3_bits_to_fp32(0xFE))


def test_nan_bytes() -> None:
    """0x7F and 0xFF are the two NaN bytes (E=0b1111, M=0b111)."""
    v = e4m3_bits_to_fp32(0x7F)
    assert math.isnan(v)
    v = e4m3_bits_to_fp32(0xFF)
    assert math.isnan(v)
    # FP32 NaN -> 0x7F
    assert fp32_to_e4m3_bits(float("nan")) == 0x7F


def test_saturation() -> None:
    """±Inf and any value above ±448 saturate to ±448 (NOT NaN)."""
    assert fp32_to_e4m3_bits(float("inf")) == 0x7E
    assert fp32_to_e4m3_bits(float("-inf")) == 0xFE
    assert fp32_to_e4m3_bits(1000.0) == 0x7E
    assert fp32_to_e4m3_bits(-1000.0) == 0xFE


def test_rne_half_ulp() -> None:
    """Round-to-nearest-even on the 1.0 / 1.125 boundary."""
    # Halfway between 1.0 (0x38) and 1.125 (0x39) is 1.0625.
    # mantissa LSB of 0x38 is 0 (even), so 1.0625 rounds DOWN to 1.0.
    assert fp32_to_e4m3_bits(1.0625) == 0x38
    # Halfway between 1.125 (0x39) and 1.25 (0x3A) is 1.1875.
    # mantissa LSB of 0x39 is 1 (odd), so 1.1875 rounds UP to 1.25.
    assert fp32_to_e4m3_bits(1.1875) == 0x3A


def test_subnormal_roundtrip() -> None:
    """Subnormal range (E=0, M=1..7): round-trip is exact."""
    for b in range(0x01, 0x08):
        v = e4m3_bits_to_fp32(b)
        assert v > 0.0
        assert v < 2 ** -6  # strictly less than min normal
        b2 = fp32_to_e4m3_bits(v)
        assert b2 == b, f"subnormal byte 0x{b:02X}: v={v}, rt=0x{b2:02X}"


def test_min_subnormal_value() -> None:
    """Smallest subnormal = 2^-9 (mant_bits=1)."""
    v = e4m3_bits_to_fp32(0x01)
    expected = np.ldexp(1.0, -9)
    assert abs(v - expected) < 1e-12
    # And the FP32 encoding of 2^-9 round-trips back to 0x01
    assert fp32_to_e4m3_bits(np.float32(expected)) == 0x01


def test_fake_quant_idempotence() -> None:
    """fake_quant(fake_quant(x)) == fake_quant(x)."""
    np.random.seed(42)
    x = (np.random.rand(2048).astype(np.float32) - 0.5) * 600.0
    y1 = fp8_e4m3_per_tensor_fake_quant(x, 1.0)
    y2 = fp8_e4m3_per_tensor_fake_quant(y1, 1.0)
    np.testing.assert_array_equal(y1, y2)


def test_fake_quant_noise_floor() -> None:
    """With scale=1 and inputs in [-1, 1], max abs err is bounded by
    0.0625 (half-ULP of the 1.0 ULP step in E4M3)."""
    x = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    y = fp8_e4m3_per_tensor_fake_quant(x, 1.0)
    err = np.abs(y - x).max()
    assert err <= 0.0625 + 1e-6


def test_bulk_path_matches_scalar() -> None:
    """The bulk array path is identical to the scalar element-wise path."""
    np.random.seed(0)
    x = (np.random.rand(2048).astype(np.float32) - 0.5) * 500.0
    scalar = np.array([fp32_to_e4m3_bits(float(v)) for v in x], dtype=np.uint8)
    bulk = fp32_array_to_e4m3_bits(x)
    np.testing.assert_array_equal(scalar, bulk)


# ---------------------------------------------------------------------------
# Cross-language parity test — requires the C++ extension to be built.
# Skipped if ``neuroflow_cpp`` is not importable.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _NEUROFLOW_CPP_FP8,
    reason=(
        "neuroflow_cpp C++ extension not built or built without "
        "FP8 bindings.  Run "
        "`cmake -S cpp -B cpp/build_pybind -G 'MinGW Makefiles' "
        "-DNFLOW_BUILD_PYBIND=ON ...` and copy/symlink the .pyd to "
        "cpp/build_pybind/ to enable this test."
    ),
)
def test_cross_language_parity() -> None:
    """The C++ fp8_e4m3 routines produce bit-identical output to the
    Python implementation.  This is the **single most important
    test** in the FP8 E4M3 work — it guarantees that an IR produced
    by the Python quantiser can be quantised in C++ to the same
    byte stream (and vice versa), which is what makes the cross-
    language deployment story work.
    """
    assert _NEUROFLOW_CPP is not None
    neuroflow_cpp = _NEUROFLOW_CPP
    np.random.seed(7)
    x = (np.random.rand(4096).astype(np.float32) - 0.5) * 1000.0
    bits_py = fp32_array_to_e4m3_bits(x)
    bits_cpp = np.asarray(neuroflow_cpp.fp8_e4m3_to_bits(x), dtype=np.uint8)
    np.testing.assert_array_equal(bits_py, bits_cpp)
    # And the fake-quant round-trip matches
    y_py = fp8_e4m3_per_tensor_fake_quant(x, 1.0)
    y_cpp = np.asarray(neuroflow_cpp.fp8_e4m3_fake_quant(x, 1.0), dtype=np.float32)
    np.testing.assert_array_almost_equal(y_py, y_cpp, decimal=6)
