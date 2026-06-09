// =============================================================================
// NeuroFlow C++ Runtime — common types
// =============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nflow {

/// Per-tensor INT8 (W8A8 fake-quant) quantisation
/// parameters — see neuroflow/quant/static_quant.py.
struct QuantParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
};

/// Per-channel INT8 quantisation parameters.  Used
/// for weight tensors of `nn.Linear` (`(out, in)`)
/// and `SpectralConv1d` (`(in, out, modes)`).  Each
/// output channel gets its own `(scale, zero_point)`.
struct PerChannelQuantParams {
    std::vector<float> scales;       // (n_channels,)
    std::vector<int32_t> zero_points;  // (n_channels,)
    int32_t channel_axis = 0;
    int32_t n_channels() const { return static_cast<int32_t>(scales.size()); }
};

/// Per-token (per-spatial-point) INT8 quantisation
/// parameters for an activation tensor of shape
/// `(batch, n, width)`.  Each spatial point
/// `(n_idx, w_idx)` gets its own `(scale, zero_point)`.
/// The scales / zero_points vectors are stored flat
/// in row-major order: `scales[ n_idx * width + w_idx ]`.
struct PerTokenQuantParams {
    std::vector<float> scales;       // (n_tokens,)
    std::vector<int32_t> zero_points;  // (n_tokens,)
    int32_t width = 0;  // last-axis dim
    int32_t n_tokens() const { return static_cast<int32_t>(scales.size()); }
};

/// FP8 E4M3 (1 sign + 4 exponent + 3 mantissa) per-
/// tensor quantisation parameters.  Symmetric (no
/// zero point).  The scale is the max-abs of the
/// source tensor divided by the E4M3 max
/// representable value (448).  E4M3 has 8 distinct
/// magnitudes per 2^e bucket plus a few subnormals,
/// range ±448, ~256 distinct values per sign.
struct FP8E4M3Params {
    float scale = 1.0f;
};

}  // namespace nflow
