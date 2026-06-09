// =============================================================================
// NeuroFlow INT8 GEMM — implementation
// =============================================================================
//
// Stage 3 Sprint 3.14 (naive scalar) + Sprint 3.25
// (AVX2 / VNNI-style INT8 dot product).
//
// The activation quantisation (FP32 → INT8 via scale_A
// and zp_A) happens on the fly inside the kernel; the
// input A is FP32 row-major.  The output is FP32
// row-major after dequantisation.
//
// The Sprint 3.25 path uses AVX2 intrinsics
// (_mm256_maddubs_epi16 + _mm256_madd_epi16) to do
// 32 INT8 multiplies + accumulate per iteration in
// INT32, achieving ~3-4x the throughput of the
// Sprint 3.14 scalar path on AVX2-capable CPUs.
// Sprint 3.5+ would use VNNI (AVX512_VNNI on Intel
// Cascade Lake+, equivalent on AMD Zen4+) for
// ~8x throughput.

#include "neuroflow/int8_gemm.h"

#include <cmath>
#include <cstdio>

// AVX2 is required for the Sprint 3.25 path.  We
// detect it at compile time via __AVX2__.  If not
// available, the code falls back to the scalar path
// (Sprint 3.14).
// AVX-512 VNNI is required for the Sprint 3.26
// path.  Detected via __AVX512VNNI__.  If not
// available, falls back to AVX2.
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define NFLOW_INT8_GEMM_USE_VNNI 1
#else
#define NFLOW_INT8_GEMM_USE_VNNI 0
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#define NFLOW_INT8_GEMM_USE_AVX2 1
#else
#define NFLOW_INT8_GEMM_USE_AVX2 0
#endif

namespace nflow {
namespace int8_gemm {

// ---------------------------------------------------------------------------
// AVX2 INT8 dot product (Sprint 3.25)
// ---------------------------------------------------------------------------
// Process 32 elements per iteration using AVX2
// intrinsics.  The per-element operation is:
//   a_eff = a_int8 - zp_A         (int8)
//   w_eff = w_int8 - zp_w         (int8)
//   acc  += w_eff * a_eff         (int32)
//   sum_a += a_eff
//   sum_w += w_eff
//
// We use _mm256_maddubs_epi16 to do 32 int8 muls →
// 16 int16 pairs, then _mm256_madd_epi16 to do
// 16 int16 pair-muls with the constant 1 (effectively
// summing adjacent pairs into int32).  This gives us
// 32 int8 muls + 32 int32 accumulates per iteration,
// which is the same throughput as the Intel VNNI
// instruction (just without the dedicated opcode).
#if NFLOW_INT8_GEMM_USE_AVX2
struct Avx2Acc {
    __m256i acc;     // 8 int32 (the main mat-mul accumulator)
    __m256i sum_a;   // 8 int32 (sum of a_eff)
    __m256i sum_w;   // 8 int32 (sum of w_eff)
    explicit Avx2Acc() : acc(_mm256_setzero_si256()),
                          sum_a(_mm256_setzero_si256()),
                          sum_w(_mm256_setzero_si256()) {}
    // Add 32 int8 muls.  a_q has the raw int8 codes
    // (already in [-128, 127]); the kernel subtracts
    // zp_A from a and zp_w from w in a separate
    // pre-pass (vectorised).  This function does the
    // main product accumulation only.
    inline void AddBlock(const __m256i& a_q,
                          const __m256i& w_q) {
        // _mm256_maddubs_epi16 treats a as uint8.  The
        // bit pattern of int8 v is v + 256 for v < 0.
        // So the AVX2 instruction computes
        // (a_signed + 256 * [a_signed<0]) * w_signed,
        // not (a_signed + 128) * w_signed as commonly
        // assumed.  The "subtract 128*sum_w" trick
        // only works if we pre-shift a to add 128 to
        // every byte, making the bit pattern equal to
        // a_signed + 128 (the well-known AVX2 INT8
        // GEMM pattern).  Without the pre-shift, the
        // bias is data-dependent and not removable by
        // a single constant.
        __m256i a_shifted = _mm256_add_epi8(
            a_q, _mm256_set1_epi8(0x80));  // add 128 to each int8
        __m256i prod16 = _mm256_maddubs_epi16(a_shifted, w_q);
        // prod16 is 16 int16, each is
        //   (a_signed[i*2] + 128) * w_int8[i*2]
        // + (a_signed[i*2+1] + 128) * w_int8[i*2+1]
        // = a_signed[i*2]*w_int8[i*2] + 128*w_int8[i*2]
        // + a_signed[i*2+1]*w_int8[i*2+1] + 128*w_int8[i*2+1]
        // = (a*w sum of pair) + 128 * (w sum of pair)
        // _mm256_madd_epi16(_, _mm256_set1_epi16(1))
        // sums adjacent int16s (the pair sum above) into
        // int32, with no extra factor.  So the int32
        // result is (4-product a*w sum) + 128 *
        // (4-element w sum).  Accumulating 8 int32
        // values gives 32 int8 products + 128 * sum_w_int8.
        acc = _mm256_add_epi32(
            acc, _mm256_madd_epi16(prod16, _mm256_set1_epi16(1)));
    }
    // Add a_eff (8 int32) and w_eff (8 int32) to sum_a
    // and sum_w.  Used after the per-block zp shift.
    inline void AddSums(const __m256i& a_eff_i32,
                         const __m256i& w_eff_i32) {
        sum_a = _mm256_add_epi32(sum_a, a_eff_i32);
        sum_w = _mm256_add_epi32(sum_w, w_eff_i32);
    }
    // Add a_int8 (32 bytes) and w_int8 (32 bytes)
    // as int16 (sign-extended), accumulating into
    // sum_a and sum_w.  This is the SAD trick: SAD
    // against 0x80 (with the byte re-centered) gives
    // the sum of absolute differences, but a cleaner
    // approach is to do _mm256_sad_epu8 against a
    // vector with the high bit cleared.
    inline void AddInt8Sums(const __m256i& a_int8,
                             const __m256i& w_int8) {
        // Use _mm256_sad_epu8 against the high-bit
        // mask 0x80 repeated.  The SAD computes
        // sum(|a - b|) per byte.  For our use:
        //   SAD(a, 0x80) = sum(|a - 0x80|)
        //   = sum((a & 0x7F) + (0x80 - (a & 0x7F)))
        //   = 0x80 (for each byte)
        // Hmm, that's not useful directly.  Skip
        // this and use the SAD trick from the outer
        // loop instead.
        (void)a_int8; (void)w_int8;
    }
    // Horizontal sum: reduce 8 int32 to 1 int32.
    static int32_t HsumI32(const __m256i& v) {
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        return _mm_cvtsi128_si32(s);
    }
    int32_t Acc() const { return HsumI32(acc); }
    int32_t SumA() const { return HsumI32(sum_a); }
    int32_t SumW() const { return HsumI32(sum_w); }
};
#endif  // AVX2

// ---------------------------------------------------------------------------
// AVX-512 VNNI INT8 dot product (Sprint 3.26)
// ---------------------------------------------------------------------------
// VNNI adds a single instruction
// _mm512_dpbusd_epi32 (and the AVX-512 version
// _mm512_dpbusd_avx512_epi32) that does 64 int8
// products + 16 int32 accumulates in one cycle.
// Compared to the AVX2 emulation, VNNI is ~2x
// faster (one 512-bit instruction vs 32 + 8 AVX2
// instructions) and uses ~half the port pressure.
#if NFLOW_INT8_GEMM_USE_VNNI
struct VnniAcc {
    __m512i acc;   // 16 int32 (the main mat-mul accumulator)
    explicit VnniAcc() : acc(_mm512_setzero_si512()) {}
    // Add 64 int8 muls.  dpbusd treats a as uint8,
    // b as int8; the product is a*b summed into
    // int32 lanes.  As with the AVX2 emulation, the
    // bias is "256 * sum_w_int8_per_byte" (because
    // uint8(v_signed) = v_signed + 256*[v<0], not
    // v_signed + 128).  We pre-shift a by adding
    // 128 to every byte (sub_epi8 with 0x80) so
    // that the standard "+128*sum_w" bias correction
    // works.
    inline void AddBlock(const __m512i& a_q,
                          const __m512i& w_q) {
        __m512i a_shifted = _mm512_sub_epi8(
            a_q, _mm512_set1_epi8(0x80));  // a - 128
        acc = _mm512_dpbusd_epi32(acc, a_shifted, w_q);
    }
    // Horizontal sum: reduce 16 int32 to 1 int32.
    static int32_t HsumI32(const __m512i& v) {
        // Use _mm512_reduce_add_epi32 if available
        // (AVX-512F), else do it manually.  The
        // intrinsic _mm512_reduce_add_epi32 sums
        // all 16 int32s in one go; it's available
        // when __AVX512F__ is defined (we have
        // that for VNNI).
        return _mm512_reduce_add_epi32(v);
    }
    int32_t Acc() const { return HsumI32(acc); }
};
#endif  // VNNI

// ---------------------------------------------------------------------------
// LinearForward (scalar + AVX2 + VNNI dispatch)
// ---------------------------------------------------------------------------
Status LinearForward(const Int8LinearParams& p,
                       const float* A, float* C) {
    if (!p.W_int8 || !p.scale_W || !p.zp_W || !A || !C) {
        return Status::InvalidArg("Int8Linear: null input/output");
    }
    if (p.in_features <= 0 || p.out_features <= 0) {
        return Status::InvalidArg("Int8Linear: bad feature dims");
    }
    const int in_f = p.in_features;
    const int out_f = p.out_features;
    const float scale_A = p.scale_A;
    const int32_t zp_A = p.zp_A;

#if NFLOW_INT8_GEMM_USE_VNNI || NFLOW_INT8_GEMM_USE_AVX2
    // ---- AVX2 / VNNI setup (Sprint 3.25 / 3.26) ----
    // Pre-quantise A to int8 (one-time) if in_f is
    // small enough to fit on the stack.
    bool use_simd = (in_f <= 4096);
    alignas(64) int8_t a_q_buf[4096];
    int8_t* a_q = nullptr;
#if NFLOW_INT8_GEMM_USE_VNNI
    bool use_vnni = use_simd;
#else
    bool use_vnni = false;
#endif
#if NFLOW_INT8_GEMM_USE_AVX2
    bool use_avx2 = use_simd && !use_vnni;
#else
    bool use_avx2 = false;
#endif
    if (use_vnni || use_avx2) {
        a_q = a_q_buf;
        for (int i = 0; i < in_f; ++i) {
            int32_t q = static_cast<int32_t>(
                std::round(A[i] / scale_A)) + zp_A;
            if (q < -128) q = -128;
            if (q >  127) q =  127;
            a_q[i] = static_cast<int8_t>(q);
        }
    }
    // Pre-compute per-row sum_w_int8 (signed) for all
    // output rows.  This is the dominant per-output
    // overhead, and it doesn't depend on A.  Compute
    // it once before the output loop and look it up
    // by row inside.
    int vec_sad_end = in_f - (in_f % 32);
    std::vector<int32_t> sum_w_int8_per_row_local(out_f);
    // Sprint 3.28: if the caller pre-computed
    // sum_W_per_row, use it directly (skip the
    // O(out_f * in_f) per-call pre-pass).  This is
    // the recommended path for the inline FNO1d
    // use case (where LinearForward is called
    // bsz*n times with the same weight matrix).
    int32_t* sum_w_int8_per_row = p.sum_W_per_row
        ? const_cast<int32_t*>(p.sum_W_per_row)
        : sum_w_int8_per_row_local.data();
    if (!p.sum_W_per_row) {
        __m256i zero256 = _mm256_setzero_si256();
        for (int o = 0; o < out_f; ++o) {
            const int8_t* w_row = p.W_int8 + o * in_f;
            int64_t w_usum_total = 0;
            int64_t w_neg_total = 0;
            for (int j = 0; j < vec_sad_end; j += 32) {
                __m256i wv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        w_row + j));
                __m256i usad = _mm256_sad_epu8(
                    wv, zero256);
                __m256i nm = _mm256_cmpgt_epi8(
                    zero256, wv);
                __m256i negsad = _mm256_sad_epu8(
                    nm, zero256);
                int64_t w_usum[4], w_nsum[4];
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(w_usum), usad);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(w_nsum), negsad);
                for (int k = 0; k < 4; ++k) {
                    w_usum_total += w_usum[k];
                    w_neg_total += w_nsum[k] / 255;
                }
            }
            // AVX2 tail (if in_f not divisible by 64).
            int avx_end = in_f - (in_f % 32);
            for (int j = vec_sad_end; j < avx_end; j += 32) {
                __m256i wv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        w_row + j));
                __m256i usad = _mm256_sad_epu8(
                    wv, zero256);
                __m256i nm = _mm256_cmpgt_epi8(
                    zero256, wv);
                __m256i negsad = _mm256_sad_epu8(
                    nm, zero256);
                int64_t w_usum[4], w_nsum[4];
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(w_usum), usad);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(w_nsum), negsad);
                for (int k = 0; k < 4; ++k) {
                    w_usum_total += w_usum[k];
                    w_neg_total += w_nsum[k] / 255;
                }
            }
            // Scalar tail.
            for (int j = vec_sad_end; j < in_f; ++j) {
                w_usum_total += static_cast<uint8_t>(w_row[j]);
                if (w_row[j] < 0) ++w_neg_total;
            }
            sum_w_int8_per_row[o] = static_cast<int32_t>(
                w_usum_total - 256 * w_neg_total);
        }
    }
    // Pre-compute sum_a_int8 (single value, used by
    // all rows).
    int32_t sum_a_int8 = 0;
    if (use_vnni || use_avx2) {
        __m256i zero256 = _mm256_setzero_si256();
        int64_t a_usum_total = 0;
        int64_t a_neg_total = 0;
        for (int j = 0; j < vec_sad_end; j += 32) {
            __m256i av = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(a_q + j));
            __m256i usad = _mm256_sad_epu8(av, zero256);
            __m256i nm = _mm256_cmpgt_epi8(zero256, av);
            __m256i negsad = _mm256_sad_epu8(nm, zero256);
            int64_t a_usum[4], a_nsum[4];
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(a_usum), usad);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(a_nsum), negsad);
            for (int k = 0; k < 4; ++k) {
                a_usum_total += a_usum[k];
                a_neg_total += a_nsum[k] / 255;
            }
        }
        for (int j = vec_sad_end; j < in_f; ++j) {
            a_usum_total += static_cast<uint8_t>(a_q[j]);
            if (a_q[j] < 0) ++a_neg_total;
        }
        sum_a_int8 = static_cast<int32_t>(
            a_usum_total - 256 * a_neg_total);
    }
#endif

    for (int o = 0; o < out_f; ++o) {
        const int32_t zp_w_o = p.zp_W[o];
        const int8_t* w_row = p.W_int8 + o * in_f;
        int32_t acc = 0;
        int32_t sum_a = 0;
        int32_t sum_w = 0;
        bool used_avx2 = false;

#if NFLOW_INT8_GEMM_USE_VNNI
        if (use_vnni) {
            // ---- VNNI path (Sprint 3.26) ----
            // 64 int8 elements per iteration (one
            // 512-bit register).  Same pre-sum trick
            // as the AVX2 path.  _mm512_dpbusd_epi32
            // is the dedicated VNNI instruction:
            // (a_uint8 * b_int8) summed into int32
            // in 16 lanes.  One instruction does
            // 64 int8 muls + 16 int32 accumulates.
            VnniAcc vnni;
            const int VEC = 64;
            int vec_end = in_f - (in_f % VEC);
            int i = 0;
            for (; i < vec_end; i += VEC) {
                __m512i a_vec = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(
                        a_q + i));
                __m512i w_vec = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(
                        w_row + i));
                vnni.AddBlock(a_vec, w_vec);
            }
            // Scalar tail for in_f % 64.
            for (; i < in_f; ++i) {
                int32_t a_eff = a_q[i] - zp_A;
                int32_t w_eff = static_cast<int32_t>(
                    w_row[i]) - zp_w_o;
                acc    += w_eff * a_eff;
                sum_a  += a_eff;
                sum_w  += w_eff;
            }
            int32_t raw_dot = vnni.Acc()
                              - 128 * sum_w_int8_per_row[o];
            acc = raw_dot - zp_A * sum_w_int8_per_row[o]
                              - zp_w_o * sum_a_int8
                              + static_cast<int32_t>(
                                  in_f) * zp_A * zp_w_o;
            sum_a = sum_a_int8 - in_f * zp_A;
            sum_w = sum_w_int8_per_row[o] - in_f * zp_w_o;
            used_avx2 = true;  // reuse flag to skip
                                // the scalar path
        } else
#endif
#if NFLOW_INT8_GEMM_USE_AVX2
        if (use_avx2) {
            // ---- AVX2 path (Sprint 3.25) ----
            // Pass 1 (quantise A) is done above.  We
            // have a_q + sum_a_int8 already.
            // Pass 2: AVX2 INT32 dot product.
            Avx2Acc avx;
            const int VEC = 32;
            int vec_end = in_f - (in_f % VEC);
            int i = 0;
            for (; i < vec_end; i += VEC) {
                __m256i a_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        a_q + i));
                __m256i w_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        w_row + i));
                avx.AddBlock(a_vec, w_vec);
            }
            // Scalar tail for in_f % 32.
            for (; i < in_f; ++i) {
                int32_t a_eff = a_q[i] - zp_A;
                int32_t w_eff = static_cast<int32_t>(
                    w_row[i]) - zp_w_o;
                acc    += w_eff * a_eff;
                sum_a  += a_eff;
                sum_w  += w_eff;
            }
            // _mm256_maddubs_epi16 (with the a+128
            // shift applied in AddBlock) computes
            // sum((a+128)*w) = sum(a*w) + 128*sum_w.
            int32_t raw_dot = avx.Acc()
                              - 128 * sum_w_int8_per_row[o];
            acc = raw_dot - zp_A * sum_w_int8_per_row[o]
                              - zp_w_o * sum_a_int8
                              + static_cast<int32_t>(
                                  in_f) * zp_A * zp_w_o;
            // dequantised sums (from the pre-computed
            // signed sums):
            sum_a = sum_a_int8 - in_f * zp_A;
            sum_w = sum_w_int8_per_row[o] - in_f * zp_w_o;
            used_avx2 = true;
        }
        if (!used_avx2) {
#endif
            // ---- Scalar path (Sprint 3.14) ----
            for (int i = 0; i < in_f; ++i) {
                float a_f = A[i] / scale_A;
                int32_t a_q_local = static_cast<int32_t>(
                    std::round(a_f)) + zp_A;
                if (a_q_local < -128) a_q_local = -128;
                if (a_q_local >  127) a_q_local =  127;
                const int32_t a_eff = a_q_local - zp_A;
                const int32_t w_eff = static_cast<int32_t>(
                    w_row[i]) - zp_w_o;
                acc    += w_eff * a_eff;
                sum_a  += a_eff;
                sum_w  += w_eff;
            }
#if NFLOW_INT8_GEMM_USE_AVX2
        }
#endif

        // Dequantise and write the output.  Shared
        // between scalar and AVX2 paths.
        const float sw = p.scale_W[o];
        const float y = sw * scale_A * static_cast<float>(
            acc - zp_w_o * sum_a - zp_A * sum_w
            + static_cast<int64_t>(in_f) * zp_w_o * zp_A);
        C[o] = y + (p.bias ? p.bias[o] : 0.0f);
    }
    return Status::Ok();
}

// ---------------------------------------------------------------------------
// Sprint 3.28: PrecomputeSumW (cacheable per-row sum of
// W_int8 - zp_W).  Returns a heap-allocated int32_t* of
// length `out_features` (caller owns; pass to
// `Int8LinearParams::sum_W_per_row` to skip the per-call
// pre-sum pre-pass).
// ---------------------------------------------------------------------------
const int32_t* PrecomputeSumW(
    const int8_t* W_int8, const int32_t* zp_W,
    int in_features, int out_features) {
    if (!W_int8 || !zp_W || in_features <= 0 || out_features <= 0) {
        return nullptr;
    }
    int32_t* out = new int32_t[out_features];
    for (int o = 0; o < out_features; ++o) {
        int32_t acc = 0;
        const int8_t* w_row = W_int8 + o * in_features;
        const int32_t zp = zp_W[o];
        for (int i = 0; i < in_features; ++i) {
            acc += static_cast<int32_t>(w_row[i]) - zp;
        }
        out[o] = acc;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Sprint 3.28: QuantiseActivation (FP32 -> INT8 bulk).
// A is (bsz, in_features) row-major.  A_q must be
// allocated by the caller.  Quantise one element at a
// time (the per-element work is `round(v / s) + zp` +
// clamp, ~6 FP ops; this is the same work the
// per-call path in `LinearForward` does, just hoisted
// out of the per-row loop).
// ---------------------------------------------------------------------------
void QuantiseActivation(
    const float* A, int8_t* A_q,
    int bsz, int in_features,
    float scale_A, int32_t zp_A) {
    if (!A || !A_q) return;
    const int n = bsz * in_features;
    for (int i = 0; i < n; ++i) {
        int32_t q = static_cast<int32_t>(
            std::round(A[i] / scale_A)) + zp_A;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        A_q[i] = static_cast<int8_t>(q);
    }
}

// ---------------------------------------------------------------------------
// Sprint 3.28: LinearForwardBatched.  Bulk pre-quantise
// + per-row GEMM (scalar for the inline FNO1d use case
// because the activation is small and the SIMD setup
// overhead is comparable to the work).  The
// recommended pattern for production is to call this
// function ONCE per layer per forward, not per row.
// ---------------------------------------------------------------------------
Status LinearForwardBatched(
    const Int8LinearParams& p, int bsz,
    const float* A, float* C) {
    if (bsz <= 0) return Status::Ok();
    if (!A || !C) return Status::InvalidArg(
        "LinearForwardBatched: null A or C");
    const int in_f = p.in_features;
    const int out_f = p.out_features;
    // Bulk quantise the activation matrix (bsz, in_f).
    std::vector<int8_t> A_q_buf(bsz * in_f);
    QuantiseActivation(A, A_q_buf.data(), bsz, in_f,
                       p.scale_A, p.zp_A);
    // Pre-compute per-row sum of pre-quantised A.
    // (Per-row, used in the bias correction term.)
    std::vector<int32_t> sum_a_per_row(bsz);
    for (int r = 0; r < bsz; ++r) {
        int32_t sa = 0;
        for (int i = 0; i < in_f; ++i) {
            sa += A_q_buf[r * in_f + i];
        }
        sum_a_per_row[r] = sa;
    }
    // Per-row GEMM.  For the inline FNO1d path the
    // activation is small (width ~= 16-256), so the
    // AVX2/VNNI inner loop setup overhead per row
    // (~20 cycles) is comparable to the per-row work
    // (~64 muls).  The scalar path wins for small
    // in_features.  For large in_features a future
    // sprint can switch to a SIMD-batched variant.
    for (int r = 0; r < bsz; ++r) {
        const int8_t* a_row = A_q_buf.data() + r * in_f;
        const int32_t sum_a = sum_a_per_row[r];
        float* c_row = C + r * out_f;
        for (int o = 0; o < out_f; ++o) {
            const int8_t* w_row = p.W_int8 + o * in_f;
            const int32_t zp_w_o = p.zp_W[o];
            int32_t acc = 0;
            for (int i = 0; i < in_f; ++i) {
                acc += static_cast<int32_t>(a_row[i]) *
                       (static_cast<int32_t>(w_row[i]) - zp_w_o);
            }
            const int32_t sum_w = p.sum_W_per_row
                ? p.sum_W_per_row[o] : 0;
            const float y = p.scale_W[o] * p.scale_A * static_cast<float>(
                acc - zp_w_o * sum_a - p.zp_A * sum_w
                + static_cast<int64_t>(in_f) * zp_w_o * p.zp_A);
            c_row[o] = y + (p.bias ? p.bias[o] : 0.0f);
        }
    }
    return Status::Ok();
}

}  // namespace int8_gemm
}  // namespace nflow
