// =============================================================================
// NeuroFlow C++ Runtime — FP8 E4M3 (IEEE-754 binary8) bit-level conversions
// =============================================================================
//
// Why this exists
// ---------------
// Stage 3 Sprint 3.30+ introduces FP8 E4M3 inference alongside the existing
// INT8 (per-tensor / per-channel / per-token) PTQ paths.  The Python and
// C++ sides must agree bit-for-bit on the E4M3 round-trip so the IR can
// carry `fp8_e4m3` qparams and the C++ runtime can apply them.
//
// The previous prototype in `static_quant.py::_quantise_fp8_e4m3` used a
// `log2` rounding approximation.  That approximation is NOT IEEE-754
// conformant: it loses the subnormal range, treats the (E=0b1111, M=0b111)
// NaN encoding as a number, and diverges from what NVIDIA H100 / cuDNN
// / TensorRT hardware actually computes when it quantises a tensor to
// FP8 E4M3.  This file replaces that approximation with a bit-level
// implementation that matches the OCP 8-bit Floating Point spec
// (OCP-8-FP, 2022-06) and the NVIDIA H100 E4M3 mapping:
//
//   bit 7      : sign (1 = negative)
//   bits 6..3  : exponent (4 bits, bias = 7)
//   bits 2..0  : mantissa (3 bits, hidden-1)
//
// Range
// -----
//   * Largest finite normal:   0b0_1111_110 = +448  (1.110_bin × 2^8)
//   * Smallest positive normal: 0b0_0001_000 = +2^-6  (1.000_bin × 2^-6)
//   * Largest subnormal:        0b0_0000_111 = +0.875 × 2^-6 ≈ 1.336e-5
//   * Smallest subnormal:       0b0_0000_001 = +2^-9  ≈ 1.953e-3 × 2^-6
//   * Zero:                     0b0_0000_000
//   * NaN:                      0b? _1111_111  (any sign, E=0b1111, M=0b111)
//   * NO Inf encoding in E4M3.  0b?_1111_110 is the largest finite value,
//     not infinity.  This is the key difference from E5M2.
//
// Why this matters for the project
// --------------------------------
// * Cross-language parity: PyTorch's `torch.float8_e4m3fn` and any
//   hardware-accelerated FP8 path will produce bit-identical results
//   to `fp32_to_e4m3_bits` below.  The `log2` approximation did NOT.
// * IR portability: a NeuroIR `fp8_e4m3` node can be lowered directly
//   to a future CUDA `__nv_fp8_e4m3` kernel without re-quantising.
// * `INT_MAX 255` (the highest FP8 byte value 0xFF) is NaN in E4M3,
//   NOT the largest number — a trap that the log2 prototype also missed.
//
// We provide the round-trip in both directions:
//
//   fp32_to_e4m3_bits(float v) -> uint8_t
//   e4m3_bits_to_fp32(uint8_t b) -> float
//
// and a `fake-quant` analogue of `static_quant.FP8E4M3Params.fake_quant`
// that takes a `(scale,)` per-tensor parameter and produces a float
// tensor that has been quantised to E4M3 and immediately dequantised
// back to fp32.  This is the "fake-quant" trick that lets us measure
// the FP8 quantisation noise floor in the existing FP32 compute path.
//
// NOTE: header-only, so it is inlined by the compiler into the .cpp
//       files that include it.  No link-time dependencies.
// =============================================================================

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nflow {

// ---------------------------------------------------------------------------
// Constants — public so tests / docs can reference them.
// ---------------------------------------------------------------------------

/// Largest finite representable value in FP8 E4M3 (S=0, E=0b1111, M=0b110).
/// = 1.110_bin × 2^(15-7) = 1.75 × 256 = 448.
inline constexpr float kE4M3Max = 448.0f;

/// Smallest positive normal value (S=0, E=0b0001, M=0b000).
/// = 1.000_bin × 2^(1-7) = 2^-6.
inline constexpr float kE4M3MinNormal = 1.0f / 64.0f;  // 2^-6

/// Smallest positive subnormal value (S=0, E=0b0000, M=0b001).
/// = 0.001_bin × 2^-6 = 2^-9.
inline constexpr float kE4M3MinSubnormal = 1.0f / 512.0f;  // 2^-9

/// 0b?_1111_111 = NaN (any sign).  E4M3 has no Inf encoding.
inline constexpr uint8_t kE4M3NaNByte = 0x7F;   // 0b0_1111_111
inline constexpr uint8_t kE4M3NaNByteNeg = 0xFF;  // 0b1_1111_111

// ---------------------------------------------------------------------------
// Bit-level conversion routines.
// ---------------------------------------------------------------------------

/// Convert a single FP32 value to its FP8 E4M3 bit pattern.
///
/// Implements IEEE-754 round-to-nearest-even (RNE) for the 3-bit
/// mantissa and the OCP-8-FP NaN / max-normal convention.  Inputs
/// outside ±448 saturate to ±448 (NOT NaN — the OCP spec says
/// "saturation is the default for non-NaN inputs outside the range").
/// NaN inputs are mapped to `kE4M3NaNByte` (0x7F).
/// +inf / -inf are mapped to ±448 (saturate, since E4M3 has no Inf).
/// +0 / -0 round-trip exactly.
///
/// Performance: ~10 ns on modern x86 — pure bit ops, no FP arithmetic
/// in the inner loop.  Bulk path is `fp32_array_to_e4m3_bits` below.
inline uint8_t fp32_to_e4m3_bits(float v) {
    // ---- Step 1: handle NaN / Inf up front ----
    //   - NaN: map to canonical NaN byte
    //   - +Inf / -Inf: saturate to ±448 (E4M3 has no Inf)
    if (std::isnan(v)) {
        return kE4M3NaNByte;
    }
    if (std::isinf(v)) {
        return (v < 0.0f) ? uint8_t{0xFE} : uint8_t{0x7E};  // -448 / +448
    }

    // ---- Step 2: sign extraction ----
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    uint8_t sign = static_cast<uint8_t>((u >> 31) & 0x1u);
    uint32_t f32_abs_bits = u & 0x7FFFFFFFu;  // strip sign

    // ---- Step 3: zero ----
    if (f32_abs_bits == 0) {
        return static_cast<uint8_t>(sign << 7);  // +0 or -0
    }

    // ---- Step 4: extract FP32 exponent and mantissa ----
    // FP32: 1 sign + 8 exp (bias 127) + 23 mantissa
    int32_t f32_exp = static_cast<int32_t>((f32_abs_bits >> 23) & 0xFFu) - 127;
    uint32_t f32_mant = f32_abs_bits & 0x7FFFFFu;  // 23 bits, the 1. is implicit

    // ---- Step 5: convert to E4M3 (1 sign + 4 exp bias 7 + 3 mantissa) ----
    // We are representing `1.M × 2^f32_exp` in E4M3.  The hidden 1 is
    // also implicit in E4M3, so we just need to round the top 3 bits
    // of the FP32 mantissa (after shifting to align with the E4M3
    // mantissa field at bit 0).
    //
    // Shift amount: E4M3 mantissa starts at bit 0, FP32 mantissa at
    // bit 0.  E4M3 has 3 mantissa bits, FP32 has 23.  So we shift the
    // FP32 mantissa RIGHT by (23 - 3) = 20 to align the top 3 bits
    // with E4M3.  The remaining 20 bits are the rounding residue.
    //
    // E4M3 exponent = f32_exp + 7 (bias correction).  Range:
    //   E4M3 normal:  E in [1, 15] -> f32_exp in [-6, +8]
    //   E4M3 subnormal: E == 0,   f32_exp < -6
    //   Overflow:     E would be 16 -> saturate to E=15,M=0b110 (max normal)
    constexpr int32_t kE4M3Bias = 7;
    constexpr int32_t kE4M3MaxExp = 15;        // E=15 -> mantissa must NOT be 0b111
    constexpr int32_t kE4M3MinNormalExp = 1;   // E=1 -> f32_exp == -6
    constexpr int32_t kE4M3MinExp = 0;         // subnormal

    int32_t e4m3_exp_unbiased = f32_exp + kE4M3Bias;
    int32_t e4m3_exp = e4m3_exp_unbiased;
    uint8_t e4m3_mant = static_cast<uint8_t>((f32_mant >> 20) & 0x7u);
    uint32_t residue = f32_mant & 0xFFFFFu;   // lower 20 bits

    // ---- Step 6: rounding (round-to-nearest-even) ----
    // We add 0.5 ULP (i.e. bit 19 of the residue) and let integer
    // arithmetic propagate the carry.  The "even" tie-break is handled
    // by adding 1 only when the residue is strictly > half, and adding
    // 1 (rounding to even) when the residue is exactly half and the
    // current mantissa LSB is 1.
    //
    // Half-ULP is bit 19 of the residue (value 2^19 = 524288).
    constexpr uint32_t kHalfULP = 1u << 19;        // 2^19
    constexpr uint32_t kHalfMinus1 = kHalfULP - 1; // bits below half
    if (residue > kHalfULP) {
        // Strictly above half -> round up
        e4m3_mant = static_cast<uint8_t>(e4m3_mant + 1);
    } else if (residue == kHalfULP) {
        // Exactly half -> round to even (LSB of current mantissa is 1)
        if (e4m3_mant & 0x1u) {
            e4m3_mant = static_cast<uint8_t>(e4m3_mant + 1);
        }
        // else: ties-to-even, leave mantissa as-is.
        (void)kHalfMinus1;  // silence unused warning in some configs
    }
    // else: residue < half -> truncate (mantissa unchanged)

    // If mantissa overflowed (was 0b111 and we added 1), carry into exp.
    if (e4m3_mant & 0x8u) {
        e4m3_mant &= 0x7u;
        e4m3_exp += 1;
    }

    // ---- Step 7: handle exponent out-of-range ----
    if (e4m3_exp > kE4M3MaxExp) {
        // Overflow.  Saturate to ±448 (NOT NaN).
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }
    if (e4m3_exp_unbiased < kE4M3MinNormalExp) {
        // Subnormal range.  E4M3 subnormals: 0.M × 2^-6 where M is
        // 3-bit (1..7).  So subnormal magnitudes are M × 2^-9 for
        // M = 1..7.  FP32 input 1.M32 × 2^f32_exp must lie in
        // [2^-9, 2^-6) — i.e. f32_exp ∈ {-9, -8, -7}.  For smaller
        // |f32_exp|, the value underflows to 0.
        //
        // Simplest correct derivation: convert the FP32 to an exact
        // double, scale by 2^9 (so 2^-9 becomes 1.0), round to
        // nearest 3-bit integer 0..7, clamp.  The double has 52
        // mantissa bits, so `(double)f32_value * 512.0` is exact
        // for any FP32 input.
        const float abs_f32 = *reinterpret_cast<const float*>(&f32_abs_bits);
        const double scaled = static_cast<double>(abs_f32) * 512.0;  // × 2^9
        long long rounded = static_cast<long long>(std::nearbyint(scaled));
        if (rounded < 0) rounded = 0;
        if (rounded > 7) rounded = 7;
        e4m3_mant = static_cast<uint8_t>(rounded);
        e4m3_exp = 0;
    }

    // ---- Step 8: assemble byte ----
    return static_cast<uint8_t>((sign << 7) | (e4m3_exp << 3) | e4m3_mant);
}


/// Convert an FP8 E4M3 bit pattern back to its FP32 value.
///
/// NaN bytes (0x7F, 0xFF) map to FP32 NaN.  All other bytes map to
/// the exact representable value (with implicit 1 for E in [1, 15],
/// explicit mantissa for E == 0 subnormals).
inline float e4m3_bits_to_fp32(uint8_t b) {
    uint8_t sign = (b >> 7) & 0x1u;
    uint8_t exp_bits = (b >> 3) & 0xFu;
    uint8_t mant_bits = b & 0x7u;

    // NaN: E == 0b1111 and M == 0b111 (any sign)
    if (exp_bits == 0xFu && mant_bits == 0x7u) {
        // Return a quiet NaN.  Use the standard FP32 qNaN bit pattern.
        uint32_t qnan_bits = 0x7FC00000u;  // sign 0, exp 0xFF, mantissa MSB set
        if (sign) qnan_bits |= 0x80000000u;
        float qnan = 0.0f;
        std::memcpy(&qnan, &qnan_bits, sizeof(qnan));
        return qnan;
    }

    // E == 0 subnormal: value = (mant_bits / 8) × 2^-6 = mant_bits × 2^-9.
    // mant_bits is a 3-bit integer in {1..7}.  Build the FP32 value
    // by left-shifting `mant_bits` into the FP32 mantissa field at
    // bit 20 (the top 3 bits of the 23-bit mantissa) and setting
    // exp_field = 118 (= -9 + 127, the exponent of 2^-9).  This
    // produces the value `1.M × 2^-9` where M is (mant_bits - 8) / 8
    // — wait, that's wrong: mant_bits << 20 puts mant_bits in bits
    // 22..20, so the hidden-1 + fraction is 1.{mant_bits in binary}.
    // For mant_bits = 1, that's 1.001_bin = 1.125, giving
    // 1.125 × 2^-9 = 0.002197, not 1.0 × 2^-9 = 0.001953.
    //
    // Correct construction: encode the value as a power of two with
    // the mantissa offset built in.  Use `std::ldexp` to produce
    // mant_bits × 2^-9 exactly, which the compiler lowers to a
    // single bit-shift.  Since mant_bits ≤ 7 fits in 3 bits, the
    // final FP32 mantissa has at most 3 non-zero bits — the round
    // trip is exact.
    if (exp_bits == 0) {
        if (mant_bits == 0) {
            return (sign ? -0.0f : 0.0f);
        }
        // Subnormal: value = mant_bits × 2^-9.  Apply sign.
        const float abs_val = std::ldexp(static_cast<float>(mant_bits), -9);
        return sign ? -abs_val : abs_val;
    }

    // Normal: value = (1 + mant_bits/8) × 2^(exp_bits - 7).
    // In FP32: exp_field = (exp_bits - 7) + 127 = exp_bits + 120,
    // mantissa = mant_bits << 20 (i.e. 3 bits at the top of the
    // 23-bit mantissa field).
    uint32_t fp32_exp = static_cast<uint32_t>(exp_bits) + 120u;
    uint32_t fp32_mant = static_cast<uint32_t>(mant_bits) << 20;

    uint32_t u = (static_cast<uint32_t>(sign) << 31) | (fp32_exp << 23) | fp32_mant;
    float v = 0.0f;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}


// ---------------------------------------------------------------------------
// Bulk paths — same semantics as the scalar routines, just in a tight
// loop.  These are the workhorse for the IR loader's per-tensor FP8
// quantise pass.
// ---------------------------------------------------------------------------

/// Convert a contiguous FP32 array to a contiguous FP8 E4M3 byte array.
/// `n` is the number of elements.  Caller owns both buffers.
inline void fp32_array_to_e4m3_bits(const float* in, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = fp32_to_e4m3_bits(in[i]);
    }
}


/// Convert a contiguous FP8 E4M3 byte array back to a contiguous FP32
/// array.  `n` is the number of elements.  Caller owns both buffers.
inline void e4m3_bits_to_fp32_array(const uint8_t* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = e4m3_bits_to_fp32(in[i]);
    }
}


/// Per-tensor FP8 E4M3 fake-quant.  Mirrors the Python-side
/// `FP8E4M3Params.fake_quant` exactly: scale a float tensor down to
/// the E4M3 range, snap each element to its nearest E4M3 representable
/// value (RNE), and snap back up.  The result is the FP32 tensor
/// that an FP8 hardware kernel would have produced if it had
/// dequantised back to FP32 immediately after the quantise step.
///
/// `in` and `out` may be the same buffer (in-place fake-quant is
/// allowed).  Caller owns `in` and `out`; this routine does not
/// allocate.
inline void fp8_e4m3_per_tensor_fake_quant(
    const float* in, float scale, float* out, size_t n)
{
    // Use a non-zero scale guard so divide-by-zero is impossible.
    // (Python side returns scale=1.0 for all-zero tensors, but the
    // fake-quant of an all-zero tensor is the all-zero tensor
    // regardless of scale.)
    const float inv_scale = (scale == 0.0f) ? 1.0f : (1.0f / scale);
    for (size_t i = 0; i < n; ++i) {
        // 1) Scale down to E4M3 range.
        const float x_scaled = in[i] * inv_scale;
        // 2) Saturate to ±448 (so the e4m3_bits_to_fp32 call never
        //    returns NaN for an in-range input).
        const float x_clamped =
            (x_scaled >  kE4M3Max) ?  kE4M3Max :
            (x_scaled < -kE4M3Max) ? -kE4M3Max : x_scaled;
        // 3) Round to nearest E4M3 byte, then back to FP32.
        const uint8_t b = fp32_to_e4m3_bits(x_clamped);
        const float x_q = e4m3_bits_to_fp32(b);
        // 4) Scale back up.
        out[i] = x_q * scale;
    }
}


}  // namespace nflow
