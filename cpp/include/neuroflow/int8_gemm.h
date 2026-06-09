// =============================================================================
// NeuroFlow INT8 GEMM (Stage 3 Sprint 3.14)
// =============================================================================
//
// Real INT8 GEMM with INT32 accumulation.  W8A8
// (per-channel INT8 weights × per-tensor INT8
// activations) → FP32 output.
//
// The TensorRT / ONNX runtime pattern:
//
//   y[o] = sum_i scale_W[o] * scale_A *
//                (W_int8[o, i] - zp_W[o]) *
//                (A_int8[i] - zp_A)
//         + b[o]
//
//   = scale_W[o] * scale_A *
//     (sum_i W_int8[o, i] * A_int8[i]
//      - zp_W[o] * sum_i A_int8[i]
//      - zp_A     * sum_i W_int8[o, i]
//      + n * zp_W[o] * zp_A)
//     + b[o]
//
// All four sums are INT32; the final scaling + bias is
// FP32.  The W_int8, A_int8, and accumulators are INT8 /
// INT32; only the final scaling is FP32.  This is
// 4x less weight bandwidth than FP32 and uses INT32
// accumulators (which are 4x wider than INT8 but
// still fit in a single register on most CPUs).
//
// This is a naive scalar implementation; the SIMD /
// BLAS-accelerated path is a Stage 3.5+ item.
// =============================================================================

#pragma once

#include <cstdint>

#include "neuroflow/tensor.h"

namespace nflow {
namespace int8_gemm {

/// Per-output-channel INT8 weight quantisation
/// parameters (already loaded by the IR loader).
struct Int8LinearParams {
    const int8_t* W_int8;       // (out_features, in_features) row-major
    const float* scale_W;       // (out_features,) per-channel scale
    const int32_t* zp_W;        // (out_features,) per-channel zp
    const float* bias;          // (out_features,) FP32 bias, or nullptr
    int in_features = 0;
    int out_features = 0;
    float scale_A = 1.0f;       // per-tensor activation scale
    int32_t zp_A = 0;          // per-tensor activation zp
    // Sprint 3.28: optional pre-computed per-row
    // sum(W_int8_signed[o, :]).  When non-null,
    // `LinearForward` skips the per-call vectorised
    // pre-sum pre-pass (which costs O(out_f * in_f)
    // and would be repeated for every batch row).
    // The recommended pattern for production
    // (e.g. FNO1d with bsz*n rows) is:
    //   1. call `PrecomputeSumW` ONCE at construction,
    //   2. set `p.sum_W_per_row` to the result,
    //   3. call `LinearForward` per row.
    const int32_t* sum_W_per_row = nullptr;  // (out_features,) or nullptr
};

/// Pre-compute the per-row signed-sum of W_int8:
///   sum_W[o] = sum_i (W_int8[o, i] - zp_W[o])
/// Returns a vector of length `out_features`.  The
/// caller may pass the returned pointer in
/// `Int8LinearParams::sum_W_per_row` to skip the
/// per-call pre-sum pre-pass inside `LinearForward`.
/// Sprint 3.28 addition for the inline FNO1d path.
const int32_t* PrecomputeSumW(
    const int8_t* W_int8, const int32_t* zp_W,
    int in_features, int out_features);

/// INT8 GEMM with INT32 accumulation + FP32 final
/// scaling.  The input is FP32 (we quantise on the
/// fly using `scale_A` and `zp_A`).  The output is
/// FP32.
///
/// `A` is shape (in_features,).  `C` is shape
/// (out_features,).  Both are FP32 row-major.
///
/// When `p.sum_W_per_row` is non-null, the per-call
/// pre-sum pre-pass is skipped (saves O(out_f * in_f)
/// per call; essential for the inline FNO1d path which
/// calls this function bsz*n times).
Status LinearForward(const Int8LinearParams& p,
                       const float* A, float* C);

/// Sprint 3.28: batched INT8 GEMM.  `A` is shape
/// (bsz, in_features) FP32 row-major; `C` is shape
/// (bsz, out_features) FP32 row-major.  Quantises ALL
/// of A to INT8 in one bulk pass (saves the per-row
/// pre-quantise overhead that `LinearForward` would
/// incur when called bsz times) and reuses the
/// per-row `sum_W_per_row` cache when present.
/// Recommended for the inline FNO1d use case where
/// the per-layer Linear sees bsz*n rows at once.
Status LinearForwardBatched(
    const Int8LinearParams& p, int bsz,
    const float* A, float* C);

/// Sprint 3.28: pre-quantise a (bsz, in_features) FP32
/// activation matrix to a (bsz, in_features) INT8
/// matrix using `scale_A` and `zp_A`.  Used by
/// `LinearForwardBatched` and exposed so the inline
/// FNO1d path can pre-quantise once and reuse the
/// int8 buffer across multiple Linear layers (when
/// the input distribution is shared).
/// `A_q` must be allocated by the caller with at
/// least `bsz * in_features` int8_t.
void QuantiseActivation(
    const float* A, int8_t* A_q,
    int bsz, int in_features,
    float scale_A, int32_t zp_A);

}  // namespace int8_gemm
}  // namespace nflow
