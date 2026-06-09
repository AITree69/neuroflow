// =============================================================================
// NeuroFlow C++ Runtime — FNO1d forward
// =============================================================================
//
// Mirrors the Python reference (neuroflow/nn/fno.py:FNO1d). The forward pass:
//   y = proj_out(act(proj_q(h))) , h = repeated spectral+local blocks with GELU/ReLU
// =============================================================================

#include "neuroflow/fno.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#ifdef NFLOW_HAS_EIGEN
#include <Eigen/Dense>
#endif

// MinGW (and strict-mode MSVC) does not expose M_PI from <cmath> by default.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "neuroflow/fft.h"
#include "neuroflow/int8_gemm.h"

namespace nflow::fno {

namespace {

inline void ApplyActivation(float* data, int64_t n, const std::string& act) {
    if (act == "relu") {
        for (int64_t i = 0; i < n; ++i) {
            if (data[i] < 0.0f) data[i] = 0.0f;
        }
    } else {  // gelu (tanh approximation)
        const float c = std::sqrt(2.0f / static_cast<float>(M_PI));
        for (int64_t i = 0; i < n; ++i) {
            float z = data[i];
            data[i] = 0.5f * z * (1.0f + std::tanh(c * (z + 0.044715f * z * z * z)));
        }
    }
}

/// y = x @ W^T + b. x: (b, k), W: (n, k), b: (n), y: (b, n). All row-major.
inline void Linear(const float* x, const float* W, const float* b, int bsz, int k, int n,
                   float* y) {
#ifdef NFLOW_HAS_EIGEN
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X(x, bsz, k);
    // W is (n, k) row-major (PyTorch nn.Linear layout). X * W^T = (b, k) * (k, n)
    // = (b, n). Use Eigen's transpose() on the (n, k) Map to avoid an extra
    // buffer; Eigen handles the transposed-view multiplication in-place via
    // expression templates.
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        W_mat(W, n, k);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Y(y, bsz, n);
    Y.noalias() = X * W_mat.transpose();
    if (b) {
        Y.rowwise() += Eigen::Map<const Eigen::VectorXf>(b, n).transpose();
    }
#else
    for (int i = 0; i < bsz; ++i) {
        const float* xi = x + i * k;
        float* yi = y + i * n;
        for (int j = 0; j < n; ++j) {
            float acc = b ? b[j] : 0.0f;
            const float* wj = W + j * k;
            for (int p = 0; p < k; ++p) acc += xi[p] * wj[p];
            yi[j] = acc;
        }
    }
#endif
}

/// Sprint 3.28: y = x @ W^T + b in INT8 (W8A8, per-channel W +
/// per-tensor A).  Routes the GEMM through
/// `int8_gemm::LinearForward` (which selects VNNI / AVX2 /
/// scalar at compile time).  Input is FP32 (we quantise on
/// the fly); output is FP32.  Per-row.
inline Status LinearInt8(
    const float* x_row, const Int8GemmLayer& layer,
    const float scale_A, const int32_t zp_A, float* y_row) {
    nflow::int8_gemm::Int8LinearParams p;
    p.W_int8 = layer.W_int8.data();
    p.scale_W = layer.scale_W.data();
    p.zp_W = layer.zp_W.data();
    p.bias = nullptr;  // bias added after the GEMM
    p.in_features = layer.in_features;
    p.out_features = layer.out_features;
    p.scale_A = scale_A;
    p.zp_A = zp_A;
    // Sprint 3.28: pass the pre-computed per-row sum
    // (W_int8 - zp_W) so LinearForward skips the
    // O(out_f * in_f) per-call pre-sum pre-pass.
    // This is the difference between "5x speedup on
    // the bench" and "0.5x slowdown in production"
    // when LinearForward is called bsz*n times per
    // forward pass.
    p.sum_W_per_row = layer.sum_W_per_row.empty()
        ? nullptr
        : layer.sum_W_per_row.data();
    return nflow::int8_gemm::LinearForward(p, x_row, y_row);
}

/// Sprint 3.28: dispatch a (b, k) @ (k, n) Linear through
/// either the FP32 path or the inline INT8 GEMM path.
/// Decides based on:
///   - `int8_gemm_enabled` (set by EnableInt8Gemm),
///   - the cache lookup by `cache_key` (e.g. "lift.weight"),
///   - the per-tensor activation qparam lookup by
///     `act_key` (e.g. "lift.output"),
///   - the absence of per-token / FP8 qparams for `act_key`
///     (those paths are NOT compatible with per-tensor
///     INT8 GEMM and must use the FP32 fallback).
/// Returns true if it dispatched through INT8 GEMM, false
/// if it fell back to FP32 (caller must then call Linear()
/// or the per-row dispatch directly).
inline bool LinearDispatchTryInt8(
    const float* x, float* y, int bsz, int k, int n,
    const std::string& cache_key, const std::string& act_key,
    const FNO1d* self) {
    if (!self->IsInt8GemmEnabled()) return false;
    if (self->HasPerTokenQparam(act_key)) return false;
    if (self->HasFP8Qparam(act_key)) return false;
    const Int8GemmLayer* layer_ptr = self->GetInt8GemmLayer(cache_key);
    if (!layer_ptr) return false;
    const QuantParams* qp_ptr = self->GetActivationQparam(act_key);
    if (!qp_ptr) return false;
    const auto& layer = *layer_ptr;
    const float scale_A = qp_ptr->scale;
    const int32_t zp_A = qp_ptr->zero_point;
    // Sprint 3.28: route through `LinearForwardBatched`
    // which bulk-quantises the activation matrix and
    // reuses the cached `sum_W_per_row`.  This is the
    // recommended production pattern for the inline
    // FNO1d use case (one call per layer per forward,
    // not one call per row).
    nflow::int8_gemm::Int8LinearParams p;
    p.W_int8 = layer.W_int8.data();
    p.scale_W = layer.scale_W.data();
    p.zp_W = layer.zp_W.data();
    p.bias = nullptr;  // bias added after the GEMM
    p.in_features = layer.in_features;
    p.out_features = layer.out_features;
    p.scale_A = scale_A;
    p.zp_A = zp_A;
    p.sum_W_per_row = layer.sum_W_per_row.empty()
        ? nullptr
        : layer.sum_W_per_row.data();
    (void)nflow::int8_gemm::LinearForwardBatched(p, bsz, x, y);
    return true;
}

/// h: (b, w, n). Apply spectral conv on each (b, w) channel: FFT, truncate
/// to first `modes` freq, multiply by per-channel complex weight (w, w, modes),
/// IFFT.
void SpectralConv1d(const float* h, int bsz, int w, int n, int modes,
                    const float* w_real, const float* w_imag, float* out) {
    const int half = n / 2 + 1;
    std::vector<float> H(w * half * 2);          // interleaved complex
    std::vector<float> H_trunc(modes * 2);       // one channel's truncated spec
    std::vector<float> Out_ft(w * half * 2);     // output freq
    std::vector<float> time_scratch(n);

    for (int bi = 0; bi < bsz; ++bi) {
        // Reset Out_ft for this batch (truncated bins must stay zero).
        std::memset(Out_ft.data(), 0, Out_ft.size() * sizeof(float));

        // 1) Forward FFT of all w channels for this batch
        for (int wi = 0; wi < w; ++wi) {
            fft::Rfft(h + bi * w * n + wi * n, n, H.data() + wi * half * 2);
        }
        // 2) Multiply in frequency domain, truncated to first `modes` bins.
        // Mirrors np.einsum('bim,iom->bom', H_trunc, Wspec) where Wspec has
        // shape (in_ch, out_ch, modes). In the IR, spec weights are stored in
        // this exact (in, out, modes) layout (row-major), so element
        // (i, o, m) lives at index (i*w + o) * modes + m.
        for (int mi = 0; mi < modes; ++mi) {
            for (int wo = 0; wo < w; ++wo) {
                float sum_re = 0.0f, sum_im = 0.0f;
                for (int wi2 = 0; wi2 < w; ++wi2) {
                    float hr = H[wi2 * half * 2 + 2 * mi];
                    float hi = H[wi2 * half * 2 + 2 * mi + 1];
                    float wr = w_real[((wi2 * w) + wo) * modes + mi];
                    float wi3 = w_imag[((wi2 * w) + wo) * modes + mi];
                    sum_re += hr * wr - hi * wi3;
                    sum_im += hr * wi3 + hi * wr;
                }
                Out_ft[wo * half * 2 + 2 * mi] = sum_re;
                Out_ft[wo * half * 2 + 2 * mi + 1] = sum_im;
            }
        }
        // 3) Inverse FFT
        for (int wo = 0; wo < w; ++wo) {
            fft::Irfft(Out_ft.data() + wo * half * 2, n, time_scratch.data());
            std::memcpy(out + bi * w * n + wo * n, time_scratch.data(),
                        n * sizeof(float));
        }
    }
}

}  // namespace

FNO1d::FNO1d(FNO1dConfig cfg, FNO1dWeights w) : cfg_(cfg), weights_(std::move(w)) {}

namespace {
// Dequantise a single weight tensor in place using
// either a per-tensor or per-channel qparam.  Returns
// true if any dequantise was applied.
bool DequantiseWeight(
    Tensor& t,
    const std::unordered_map<std::string, QuantParams>& per_tensor,
    const std::unordered_map<std::string, PerChannelQuantParams>& per_channel,
    const std::string& name) {
    // Apply the FULL fake-quant round-trip to the FP32
    // weight (Sprint 3.21 fix).  Earlier this function
    // only did `(v - zp) * s` (the dequantise step),
    // which assumed the IR stored int8 codes; in
    // practice the Python exporter stores the FP32
    // weights directly (the NIRQ block carries the
    // qparams, the weights are the original FP32
    // values).  To match the Python reference's
    // `build_fake_quant_model` (which round-trips the
    // weight through the qparam), we now do:
    //   v' = (round(v / s + zp) - zp) * s
    // which snaps the FP32 weight to the nearest
    // representable INT8 value and dequantises back to
    // FP32.
    auto fake_quant_pt = [](float v, float s, float zp) {
        float q = std::round(v / s) + zp;
        if (q < -128.0f) q = -128.0f;
        if (q >  127.0f) q =  127.0f;
        return (q - zp) * s;
    };

    // Per-channel takes precedence when present.
    auto it_pc = per_channel.find(name);
    if (it_pc != per_channel.end()) {
        const auto& pcp = it_pc->second;
        const int axis = pcp.channel_axis;
        const int64_t n_ch = pcp.n_channels();
        std::vector<int64_t> shape(t.shape().begin(), t.shape().end());
        if (t.shape().size() == 2 && axis == 0) {
            for (int64_t c = 0; c < n_ch; ++c) {
                const float s = pcp.scales[c];
                const float zp = static_cast<float>(pcp.zero_points[c]);
                for (int64_t i = 0; i < shape[1]; ++i) {
                    const int64_t off = c * shape[1] + i;
                    t.data()[off] = fake_quant_pt(t.data()[off], s, zp);
                }
            }
            return true;
        } else if (t.shape().size() == 3 && axis == 1) {
            const int64_t in_dim = shape[0];
            const int64_t out_dim = shape[1];
            const int64_t modes_dim = shape[2];
            for (int64_t i = 0; i < in_dim; ++i) {
                for (int64_t c = 0; c < out_dim; ++c) {
                    const float s = pcp.scales[c];
                    const float zp = static_cast<float>(pcp.zero_points[c]);
                    for (int64_t k = 0; k < modes_dim; ++k) {
                        const int64_t off =
                            i * (out_dim * modes_dim) + c * modes_dim + k;
                        t.data()[off] = fake_quant_pt(t.data()[off], s, zp);
                    }
                }
            }
            return true;
        }
        // Unsupported layout — fall through to per-tensor.
    }
    auto it_pt = per_tensor.find(name);
    if (it_pt != per_tensor.end()) {
        const float s = it_pt->second.scale;
        const float zp = static_cast<float>(it_pt->second.zero_point);
        const int64_t numel = t.numel();
        for (int64_t i = 0; i < numel; ++i) {
            t.data()[i] = fake_quant_pt(t.data()[i], s, zp);
        }
        return true;
    }
    return false;
}
}  // namespace

void FNO1d::EnablePerChannelWeightDequant(
    const std::unordered_map<std::string, QuantParams>& per_tensor,
    const std::unordered_map<std::string, PerChannelQuantParams>& per_channel) {
    // Dequantise the weights in place.  After this call,
    // the weights are stored as FP32 in the Tensor
    // buffers and Forward runs as usual.
    DequantiseWeight(weights_.lift_w, per_tensor, per_channel, "lift.weight");
    DequantiseWeight(weights_.lift_b, per_tensor, per_channel, "lift.bias");
    DequantiseWeight(weights_.proj_q_w, per_tensor, per_channel, "proj_q.weight");
    DequantiseWeight(weights_.proj_q_b, per_tensor, per_channel, "proj_q.bias");
    DequantiseWeight(weights_.proj_out_w, per_tensor, per_channel, "proj_out.weight");
    DequantiseWeight(weights_.proj_out_b, per_tensor, per_channel, "proj_out.bias");
    for (size_t i = 0; i < weights_.spec_w_real.size(); ++i) {
        DequantiseWeight(weights_.spec_w_real[i], per_tensor, per_channel,
                          "specs." + std::to_string(i) + ".weights_real");
        DequantiseWeight(weights_.spec_w_imag[i], per_tensor, per_channel,
                          "specs." + std::to_string(i) + ".weights_imag");
    }
    for (size_t i = 0; i < weights_.loc_w.size(); ++i) {
        DequantiseWeight(weights_.loc_w[i], per_tensor, per_channel, "locs." + std::to_string(i) + ".weight");
        DequantiseWeight(weights_.loc_b[i], per_tensor, per_channel, "locs." + std::to_string(i) + ".bias");
    }
}

// ---------------------------------------------------------------------------
// Sprint 3.28: quantise a 2D FP32 weight (out, in) to INT8 per-channel.
// Writes the int8 codes into `dst.W_int8`, the per-channel
// scales / zero_points into `dst.scale_W` / `dst.zp_W`,
// and stores in/out dims.  Uses the standard
//   q = round(v / s) + zp   (clamped to [-128, 127])
// formula.  Returns true on success.
// ---------------------------------------------------------------------------
static bool QuantiseWeight2D(
    const Tensor& t,
    const PerChannelQuantParams& pcp,
    Int8GemmLayer& dst) {
    if (t.shape().size() != 2 || pcp.channel_axis != 0) {
        return false;
    }
    const int64_t n_ch = t.shape()[0];   // out_features
    const int64_t k = t.shape()[1];      // in_features
    dst.in_features = static_cast<int>(k);
    dst.out_features = static_cast<int>(n_ch);
    dst.W_int8.assign(n_ch * k, 0);
    dst.scale_W.assign(n_ch, 1.0f);
    dst.zp_W.assign(n_ch, 0);
    dst.sum_W_per_row.assign(n_ch, 0);  // Sprint 3.28
    for (int64_t c = 0; c < n_ch; ++c) {
        const float s = pcp.scales[c];
        const float zp = static_cast<float>(pcp.zero_points[c]);
        dst.scale_W[c] = s;
        dst.zp_W[c] = pcp.zero_points[c];
        int32_t row_sum = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t off = c * k + i;
            float q = std::round(t.data()[off] / s) + zp;
            if (q < -128.0f) q = -128.0f;
            if (q >  127.0f) q =  127.0f;
            const int8_t q8 = static_cast<int8_t>(q);
            dst.W_int8[off] = q8;
            row_sum += static_cast<int32_t>(q8) - pcp.zero_points[c];
        }
        dst.sum_W_per_row[c] = row_sum;
    }
    return true;
}

void FNO1d::EnableInt8Gemm(
    const std::unordered_map<std::string, PerChannelQuantParams>&
        per_channel) {
    // One-shot: quantise each Linear's FP32 weight to
    // INT8 (per-channel) and cache it.  We keep the
    // original FP32 weights alongside the cache so the
    // Forward path can fall back to FP32 Linear when
    // the per-tensor activation qparam is missing (i.e.
    // the per-token / FP8 paths are active).
    int8_gemm_enabled_ = true;
    int8_gemm_cache_.clear();
    // Lift
    auto it_lift = per_channel.find("lift.weight");
    if (it_lift != per_channel.end()) {
        Int8GemmLayer layer;
        if (QuantiseWeight2D(weights_.lift_w, it_lift->second, layer)) {
            int8_gemm_cache_["lift.weight"] = std::move(layer);
        }
    }
    // Per-layer loc
    for (size_t i = 0; i < weights_.loc_w.size(); ++i) {
        const std::string key = "locs." + std::to_string(i) + ".weight";
        auto it = per_channel.find(key);
        if (it != per_channel.end()) {
            Int8GemmLayer layer;
            if (QuantiseWeight2D(weights_.loc_w[i], it->second, layer)) {
                int8_gemm_cache_[key] = std::move(layer);
            }
        }
    }
    // proj_q
    auto it_pq = per_channel.find("proj_q.weight");
    if (it_pq != per_channel.end()) {
        Int8GemmLayer layer;
        if (QuantiseWeight2D(weights_.proj_q_w, it_pq->second, layer)) {
            int8_gemm_cache_["proj_q.weight"] = std::move(layer);
        }
    }
    // proj_out
    auto it_po = per_channel.find("proj_out.weight");
    if (it_po != per_channel.end()) {
        Int8GemmLayer layer;
        if (QuantiseWeight2D(weights_.proj_out_w, it_po->second, layer)) {
            int8_gemm_cache_["proj_out.weight"] = std::move(layer);
        }
    }
}

Status FNO1d::Forward(const Tensor& x, Tensor& y) const {
    if (x.shape().size() != 3) {
        return Status::ShapeMismatch("expected 3D input (batch, n, in_ch)");
    }
    int64_t bsz = x.shape()[0];
    int64_t n = x.shape()[1];
    int64_t in_ch = x.shape()[2];
    if (in_ch != cfg_.in_channels) {
        return Status::ShapeMismatch("input channels mismatch");
    }
    if (y.shape().size() != 3 || y.shape()[0] != bsz || y.shape()[1] != n ||
        y.shape()[2] != cfg_.out_channels) {
        return Status::ShapeMismatch("output buffer shape mismatch");
    }
    if (!fft::IsPow2(n)) {
        return Status::InvalidArg("Stage 1: input length n must be a power of two");
    }
    const int w = cfg_.width;
    const int modes = cfg_.modes;
    const int L = cfg_.n_layers;
    const std::string& act = cfg_.activation;

    // Lifting: x (b, n, in_ch) @ lift_w^T (in_ch, w) + lift_b (w)
    // -> h0 (b, n, w) -> permute to (b, w, n)
    Tensor h = Tensor::Zeros({bsz, n, w});
    // Sprint 3.28: lift has no INPUT qparam (the input
    // is the raw `x`); we use the FP32 Linear path
    // unconditionally.  (If we ever want INT8 GEMM on
    // the lift, the caller would need to supply a
    // per-tensor qparam for the input distribution,
    // which is not the case in the current FNO1d
    // architecture.)
    Linear(x.data(), weights_.lift_w.data(), weights_.lift_b.data(),
           static_cast<int>(bsz) * static_cast<int>(n), static_cast<int>(in_ch), w,
           h.data());

    Tensor h_perm = Tensor::Zeros({bsz, w, n});
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int wi = 0; wi < w; ++wi) {
            for (int64_t ni = 0; ni < n; ++ni) {
                h_perm.data()[bi * w * n + wi * n + ni] =
                    h.data()[bi * n * w + ni * w + wi];
            }
        }
    }

    // Optional pad to pad_factor
    int64_t pad_len = 0;
    Tensor padded;
    if (cfg_.pad_factor > 1) {
        int64_t r = n % cfg_.pad_factor;
        pad_len = (r == 0) ? 0 : (cfg_.pad_factor - r);
        if (pad_len > 0) {
            padded = Tensor::Zeros({bsz, w, n + pad_len});
            for (int64_t bi = 0; bi < bsz; ++bi) {
                for (int wi = 0; wi < w; ++wi) {
                    std::memcpy(padded.data() + bi * w * (n + pad_len) + wi * (n + pad_len),
                                h_perm.data() + bi * w * n + wi * n, n * sizeof(float));
                }
            }
            h_perm = padded;
            n += pad_len;
        }
    }

    Tensor x1 = Tensor::Zeros({bsz, w, n});
    Tensor x2 = Tensor::Zeros({bsz, w, n});
    Tensor x2_perm_in = Tensor::Zeros({bsz, n, w});
    Tensor x2_perm_out = Tensor::Zeros({bsz, n, w});
    Tensor add = Tensor::Zeros({bsz, w, n});

    for (int i = 0; i < L; ++i) {
        // Optional input fake-quant: snap `h_perm` to
        // the calibrated range of the *previous* layer's
        // output.  This matches the Python
        // `FakeQuantLinear.act_in_qp.fake_quant(x)` call
        // before the Linear matmul.  Layer i's input
        // qparam is layer (i-1)'s output qparam (or
        // `lift.output` for i=0).  When `h_perm` is
        // already in range this is a no-op.
        if (this->quant_enabled_) {
            std::string in_key;
            if (i == 0) {
                in_key = "lift.output";
            } else {
                in_key = "locs." + std::to_string(i - 1) + ".output";
            }
            auto it_in = this->activation_qparams_.find(in_key);
            if (it_in != this->activation_qparams_.end()) {
                const auto& qp_in = it_in->second;
                const float scale = qp_in.scale;
                const float zp = static_cast<float>(qp_in.zero_point);
                for (int64_t k = 0; k < bsz * w * n; ++k) {
                    float v = h_perm.data()[k];
                    float q = std::round(v / scale) + zp;
                    if (q < -128.0f) q = -128.0f;
                    if (q >  127.0f) q =  127.0f;
                    h_perm.data()[k] = (q - zp) * scale;
                }
            }
        }

        SpectralConv1d(h_perm.data(), static_cast<int>(bsz), w, static_cast<int>(n), modes,
                       weights_.spec_w_real[i].data(),
                       weights_.spec_w_imag[i].data(), x1.data());

        // x2 = permute(h_perm, (0,2,1)) @ loc_w^T + loc_b -> (b, n, w) -> permute back
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int64_t ni = 0; ni < n; ++ni) {
                for (int wi = 0; wi < w; ++wi) {
                    x2_perm_in.data()[bi * n * w + ni * w + wi] =
                        h_perm.data()[bi * w * n + wi * n + ni];
                }
            }
        }
        // Linear into a separate output buffer to avoid in-place aliasing.
        // Sprint 3.28: try inline INT8 GEMM (per-channel W
        // + per-tensor A) when configured and the
        // per-tensor activation qparam is available for
        // this layer's input.  Falls back to FP32 Linear
        // for the per-token / FP8 paths.
        const std::string in_key_for_i = (i == 0)
            ? std::string("lift.output")
            : std::string("locs.") + std::to_string(i - 1) + ".output";
        const std::string cache_key = "locs." + std::to_string(i) + ".weight";
        if (!LinearDispatchTryInt8(
                x2_perm_in.data(), x2_perm_out.data(),
                static_cast<int>(bsz) * static_cast<int>(n), w, w,
                cache_key, in_key_for_i, this)) {
            Linear(x2_perm_in.data(), weights_.loc_w[i].data(),
                   weights_.loc_b[i].data(),
                   static_cast<int>(bsz) * static_cast<int>(n), w, w,
                   x2_perm_out.data());
        } else if (weights_.loc_b[i].data() != nullptr) {
            // LinearDispatchTryInt8 doesn't add bias; do it here.
            const float* b = weights_.loc_b[i].data();
            for (int64_t r = 0; r < bsz * n; ++r) {
                for (int j = 0; j < w; ++j) {
                    x2_perm_out.data()[r * w + j] += b[j];
                }
            }
        }
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int wi = 0; wi < w; ++wi) {
                for (int64_t ni = 0; ni < n; ++ni) {
                    x2.data()[bi * w * n + wi * n + ni] =
                        x2_perm_out.data()[bi * n * w + ni * w + wi];
                }
            }
        }

        // INT8 fake-quant on the Linear's pre-activation
        // output (x2 in the FNO1d design).  This matches
        // the Python `FakeQuantLinear` wrapper which
        // applies `act_out_qp.fake_quant(y)` after the
        // Linear matmul.  When the qparam is per-token,
        // each spatial point's (n_idx, w_idx) gets its
        // own scale / zero_point; when per-tensor, the
        // whole tensor snaps to a single (scale, zp).
        // The qparams are calibrated on the Linear's
        // pre-activation output (the hook in
        // `calibrate` is on the Linear module), so the
        // per-token scheme is well-matched to the data
        // range at this point in the pipeline.
        if (this->quant_enabled_) {
            const std::string key =
                std::string("locs.") + std::to_string(i) + ".output";
            // FP8 E4M3 has highest precedence when
            // configured (it's the most accurate of the
            // three INT8 schemes because of the wider
            // dynamic range).
            auto it_fp8 = this->activation_fp8_qparams_.find(key);
            if (it_fp8 != this->activation_fp8_qparams_.end()) {
                // FP8 E4M3: quantise by snapping
                // log2(|x|) to the nearest 1/8.  Range
                // ±448.  Symmetric (no zero point).
                const float scale = it_fp8->second.scale;
                if (scale > 0.0f) {
                    const int64_t numel = bsz * n * w;
                    for (int64_t k = 0; k < numel; ++k) {
                        const float v = x2.data()[k] / scale;
                        // Snap |v| to the nearest 2^(e +
                        // m/8) value (3-bit mantissa),
                        // preserving sign.
                        const float abs_v = std::abs(v);
                        const float sign = v < 0.0f ? -1.0f : 1.0f;
                        // For abs_v < 2^(-6) (subnormal
                        // range), keep as-is.  For
                        // abs_v >= 2^(-6), snap to
                        // nearest 2^(round(log2 * 8) / 8).
                        const float quant_v = (abs_v < 0.015625f)
                            ? abs_v
                            : std::ldexp(1.0f,
                                  std::round(std::log2(abs_v) * 8.0f) / 8.0f);
                        // Saturate to ±448.
                        const float clamped = std::min(quant_v, 448.0f);
                        x2.data()[k] = sign * clamped * scale;
                    }
                }
            } else {
            auto it_ptok = this->activation_per_token_qparams_.find(key);
            const int32_t w_axis = it_ptok != this->activation_per_token_qparams_.end()
                                      ? it_ptok->second.width
                                      : static_cast<int32_t>(w);
            if (it_ptok != this->activation_per_token_qparams_.end()) {
                const auto& ptp = it_ptok->second;
                // x2 has shape (bsz, n, w) — different
                // convention from `add` (bsz, w, n).
                // The Python qparam is indexed by
                // (n_idx * w + w_idx); here we have
                // x2.data()[ bi * n * w_axis + n_i * w_axis + w_i ].
                for (int64_t bi = 0; bi < bsz; ++bi) {
                    for (int64_t ni = 0; ni < n; ++ni) {
                        for (int64_t wi = 0; wi < w_axis; ++wi) {
                            const int32_t qp_idx = static_cast<int32_t>(
                                ni * w_axis + wi);
                            const float scale = ptp.scales[qp_idx];
                            const float zp = static_cast<float>(
                                ptp.zero_points[qp_idx]);
                            const int64_t off =
                                bi * n * w_axis + ni * w_axis + wi;
                            float v = x2.data()[off];
                            float q = std::round(v / scale) + zp;
                            if (q < -128.0f) q = -128.0f;
                            if (q >  127.0f) q =  127.0f;
                            x2.data()[off] = (q - zp) * scale;
                        }
                    }
                }
            } else {
                auto it = this->activation_qparams_.find(key);
                if (it != this->activation_qparams_.end()) {
                    const auto& qp = it->second;
                    const float scale = qp.scale;
                    const float zp = static_cast<float>(qp.zero_point);
                    const int64_t numel = bsz * n * w_axis;
                    for (int64_t k = 0; k < numel; ++k) {
                        float v = x2.data()[k];
                        float q = std::round(v / scale) + zp;
                        if (q < -128.0f) q = -128.0f;
                        if (q >  127.0f) q =  127.0f;
                        x2.data()[k] = (q - zp) * scale;
                    }
                }
            }
            }  // close FP8 / non-FP8 else dispatch
        }

        // h = act(x1 + x2)
        for (int64_t i2 = 0; i2 < bsz * w * n; ++i2) {
            add.data()[i2] = x1.data()[i2] + x2.data()[i2];
        }
        ApplyActivation(add.data(), bsz * w * n, act);

        // No additional post-activation fake-quant: the
        // Python reference's `FakeQuantLinear` only
        // fake-quants the *pre-activation* linear output
        // (via `act_out_qp.fake_quant(y)`); the
        // post-activation is left in FP32 and the next
        // iteration's `act_in_qp.fake_quant(x)` snaps it
        // to the previous layer's range.
        std::memcpy(h_perm.data(), add.data(), bsz * w * n * sizeof(float));
    }

    if (pad_len > 0) {
        Tensor trimmed = Tensor::Zeros({bsz, w, n - pad_len});
        int64_t orig = n - pad_len;
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int wi = 0; wi < w; ++wi) {
                std::memcpy(trimmed.data() + bi * w * orig + wi * orig,
                            h_perm.data() + bi * w * n + wi * n, orig * sizeof(float));
            }
        }
        h_perm = trimmed;
        n -= pad_len;
    }

    // h (b, w, n) -> permute (b, n, w) -> proj_q -> act -> proj_out -> y
    Tensor h_back = Tensor::Zeros({bsz, n, w});
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int64_t ni = 0; ni < n; ++ni) {
            for (int wi = 0; wi < w; ++wi) {
                h_back.data()[bi * n * w + ni * w + wi] =
                    h_perm.data()[bi * w * n + wi * n + ni];
            }
        }
    }

    Tensor q = Tensor::Zeros({bsz, n, w});
    Linear(h_back.data(), weights_.proj_q_w.data(), weights_.proj_q_b.data(),
           static_cast<int>(bsz) * static_cast<int>(n), w, w, q.data());
    ApplyActivation(q.data(), bsz * n * w, act);

    Linear(q.data(), weights_.proj_out_w.data(), weights_.proj_out_b.data(),
           static_cast<int>(bsz) * static_cast<int>(n), w, static_cast<int>(cfg_.out_channels),
           y.data());

    return Status::Ok();
}

// =============================================================================
// FNO2d (Stage 2 Sprint 1 Phase 2)
// =============================================================================

namespace {

/// h: (b, w, H, W). Apply 2D spectral conv on each (b, w) channel:
/// 2D FFT, truncate to a (modes_h, modes_w) box, multiply by per-channel
/// complex weight (in=w, out=w, modes_h, modes_w), 2D IFFT.
void SpectralConv2d(const float* h, int bsz, int w, int H, int W,
                    int modes_h, int modes_w,
                    const float* w_real, const float* w_imag, float* out) {
    const int half_w = W / 2 + 1;
    // (w, H, half_w) interleaved complex spectrum per batch.
    std::vector<float> H_spec(static_cast<size_t>(w) * H * half_w * 2);
    // (w, H, half_w) interleaved output spectrum per batch.
    std::vector<float> Out_ft(static_cast<size_t>(w) * H * half_w * 2);

    for (int bi = 0; bi < bsz; ++bi) {
        // Reset Out_ft (modes_h*modes_w box is overwritten; bins above it
        // must stay zero for the inverse FFT to produce the right signal).
        std::memset(Out_ft.data(), 0, Out_ft.size() * sizeof(float));

        // 1) Forward 2D FFT of all w channels for this batch.
        for (int wi = 0; wi < w; ++wi) {
            fft::Rfft2(h + (static_cast<int64_t>(bi) * w + wi) * H * W, H, W,
                       H_spec.data() + (static_cast<int64_t>(wi) * H) * half_w * 2);
        }
        // 2) Multiply in frequency domain, truncated to (modes_h, modes_w).
        // Mirrors np.einsum('bimn,iomn->bomn', H_trunc, Wspec) where Wspec
        // has shape (in, out, mh, mw) in row-major layout. Element
        // (wi, wo, mi, mj) lives at index (((wi * w) + wo) * mh + mi) * mw + mj.
        for (int mi = 0; mi < modes_h; ++mi) {
            for (int mj = 0; mj < modes_w; ++mj) {
                for (int wo = 0; wo < w; ++wo) {
                    float sum_re = 0.0f, sum_im = 0.0f;
                    for (int wi = 0; wi < w; ++wi) {
                        const float* H_wi = H_spec.data()
                            + ((static_cast<int64_t>(wi) * H + mi) * half_w + mj) * 2;
                        float hr = H_wi[0];
                        float hi = H_wi[1];
                        const float* w_wi = w_real
                            + (((static_cast<int64_t>(wi) * w) + wo) * modes_h + mi) * modes_w + mj;
                        const float* wi_wi = w_imag
                            + (((static_cast<int64_t>(wi) * w) + wo) * modes_h + mi) * modes_w + mj;
                        float wr = w_wi[0];
                        float wii = wi_wi[0];
                        sum_re += hr * wr - hi * wii;
                        sum_im += hr * wii + hi * wr;
                    }
                    float* Out_wo = Out_ft.data()
                        + ((static_cast<int64_t>(wo) * H + mi) * half_w + mj) * 2;
                    Out_wo[0] = sum_re;
                    Out_wo[1] = sum_im;
                }
            }
        }
        // 3) Inverse 2D FFT, writing directly into the output buffer.
        for (int wo = 0; wo < w; ++wo) {
            fft::Irfft2(Out_ft.data() + static_cast<int64_t>(wo) * H * half_w * 2, H, W,
                        out + ((static_cast<int64_t>(bi) * w + wo) * H) * W);
        }
    }
}

}  // namespace

FNO2d::FNO2d(FNO2dConfig cfg, FNO2dWeights w) : cfg_(cfg), weights_(std::move(w)) {}

Status FNO2d::Forward(const Tensor& x, Tensor& y) const {
    if (x.shape().size() != 4) {
        return Status::ShapeMismatch("expected 4D input (batch, h, w, in_ch)");
    }
    int64_t bsz = x.shape()[0];
    int64_t H = x.shape()[1];
    int64_t W = x.shape()[2];
    int64_t in_ch = x.shape()[3];
    if (in_ch != cfg_.in_channels) {
        return Status::ShapeMismatch("input channels mismatch");
    }
    if (y.shape().size() != 4 || y.shape()[0] != bsz || y.shape()[1] != H ||
        y.shape()[2] != W || y.shape()[3] != cfg_.out_channels) {
        return Status::ShapeMismatch("output buffer shape mismatch");
    }
    if (!fft::IsPow2(H) || !fft::IsPow2(W)) {
        return Status::InvalidArg("FNO2d: H and W must each be a power of two");
    }
    const int w = cfg_.width;
    const int mh = cfg_.modes_h;
    const int mw = cfg_.modes_w;
    const int L = cfg_.n_layers;
    const std::string& act = cfg_.activation;

    // Helper: apply INT8 / FP8 activation fake-quant
    // to a flat buffer (Sprint 3.20).  Looks up the
    // qparams in `activation_fp8_qparams_` (FP8
    // highest precedence) or `activation_qparams_`
    // (INT8 per-tensor fallback).  Mirrors the
    // FNO1d dispatch but extracted to a helper so
    // the 4 Linear boundaries in FNO2d (lift,
    // locs.<i>, proj_q, proj_out) can share the
    // same logic.
    const auto apply_fake_quant = [this](
        float* data, int64_t numel, const std::string& key) {
        if (!this->quant_enabled_) return;
        auto it_fp8 = this->activation_fp8_qparams_.find(key);
        if (it_fp8 != this->activation_fp8_qparams_.end()) {
            const float scale = it_fp8->second.scale;
            if (scale > 0.0f) {
                for (int64_t k = 0; k < numel; ++k) {
                    const float v = data[k] / scale;
                    const float abs_v = std::abs(v);
                    const float sign = v < 0.0f ? -1.0f : 1.0f;
                    const float quant_v = (abs_v < 0.015625f)
                        ? abs_v
                        : std::ldexp(1.0f,
                              std::round(std::log2(abs_v) * 8.0f) / 8.0f);
                    const float clamped = std::min(quant_v, 448.0f);
                    data[k] = sign * clamped * scale;
                }
            }
        } else {
            auto it = this->activation_qparams_.find(key);
            if (it != this->activation_qparams_.end()) {
                const auto& qp = it->second;
                const float scale = qp.scale;
                const float zp = static_cast<float>(qp.zero_point);
                for (int64_t k = 0; k < numel; ++k) {
                    float v = data[k];
                    float q = std::round(v / scale) + zp;
                    if (q < -128.0f) q = -128.0f;
                    if (q >  127.0f) q =  127.0f;
                    data[k] = (q - zp) * scale;
                }
            }
        }
    };

    // Lifting: x (b, H, W, in_ch) @ lift_w^T (in_ch, w) + lift_b (w)
    //   -> h0 (b, H, W, w) -> permute to (b, w, H, W)
    Tensor h = Tensor::Zeros({bsz, H, W, w});
    Linear(x.data(), weights_.lift_w.data(), weights_.lift_b.data(),
           static_cast<int>(bsz) * static_cast<int>(H) * static_cast<int>(W),
           static_cast<int>(in_ch), w, h.data());
    // Fake-quant at lift.output (Sprint 3.20).
    apply_fake_quant(h.data(), bsz * H * W * w, "lift.output");

    Tensor h_perm = Tensor::Zeros({bsz, w, H, W});
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int wi = 0; wi < w; ++wi) {
            for (int64_t hi = 0; hi < H; ++hi) {
                for (int64_t wj = 0; wj < W; ++wj) {
                    h_perm.data()[((bi * w + wi) * H + hi) * W + wj] =
                        h.data()[((bi * H + hi) * W + wj) * w + wi];
                }
            }
        }
    }

    // Pad to pad_factor multiples along H and W.
    int64_t pad_h = 0, pad_w = 0;
    Tensor padded;
    if (cfg_.pad_factor > 1) {
        const int pf = cfg_.pad_factor;
        pad_h = (pf - (H % pf)) % pf;
        pad_w = (pf - (W % pf)) % pf;
        if (pad_h > 0 || pad_w > 0) {
            const int64_t Hp = H + pad_h;
            const int64_t Wp = W + pad_w;
            padded = Tensor::Zeros({bsz, w, Hp, Wp});
            for (int64_t bi = 0; bi < bsz; ++bi) {
                for (int wi = 0; wi < w; ++wi) {
                    for (int64_t hi = 0; hi < H; ++hi) {
                        std::memcpy(
                            padded.data() + ((bi * w + wi) * Hp + hi) * Wp,
                            h_perm.data() + ((bi * w + wi) * H + hi) * W,
                            W * sizeof(float));
                    }
                }
            }
            h_perm = padded;
            H += pad_h;
            W += pad_w;
        }
    }

    Tensor x1 = Tensor::Zeros({bsz, w, H, W});
    Tensor x2 = Tensor::Zeros({bsz, w, H, W});
    Tensor x2_perm_in = Tensor::Zeros({bsz, H, W, w});
    Tensor x2_perm_out = Tensor::Zeros({bsz, H, W, w});
    Tensor add = Tensor::Zeros({bsz, w, H, W});

    for (int i = 0; i < L; ++i) {
        SpectralConv2d(h_perm.data(), static_cast<int>(bsz), w, static_cast<int>(H),
                       static_cast<int>(W), mh, mw,
                       weights_.spec_w_real[i].data(),
                       weights_.spec_w_imag[i].data(), x1.data());

        // x2 = permute(h_perm, (0, 2, 3, 1)) @ loc_w^T + loc_b -> (b, H, W, w)
        //     -> permute back to (b, w, H, W)
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int64_t hi = 0; hi < H; ++hi) {
                for (int64_t wj = 0; wj < W; ++wj) {
                    for (int wi = 0; wi < w; ++wi) {
                        x2_perm_in.data()[((bi * H + hi) * W + wj) * w + wi] =
                            h_perm.data()[((bi * w + wi) * H + hi) * W + wj];
                    }
                }
            }
        }
        Linear(x2_perm_in.data(), weights_.loc_w[i].data(), weights_.loc_b[i].data(),
               static_cast<int>(bsz) * static_cast<int>(H) * static_cast<int>(W), w, w,
               x2_perm_out.data());
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int wi = 0; wi < w; ++wi) {
                for (int64_t hi = 0; hi < H; ++hi) {
                    for (int64_t wj = 0; wj < W; ++wj) {
                        x2.data()[((bi * w + wi) * H + hi) * W + wj] =
                            x2_perm_out.data()[((bi * H + hi) * W + wj) * w + wi];
                    }
                }
            }
        }

        // INT8 / FP8 activation fake-quant at the
        // locs.<i>.output boundary (Sprint 3.20 uses
        // the shared apply_fake_quant helper).
        apply_fake_quant(x2.data(), bsz * w * H * W,
            std::string("locs.") + std::to_string(i) + ".output");

        // h = act(x1 + x2)
        const int64_t total = bsz * w * H * W;
        for (int64_t k = 0; k < total; ++k) {
            add.data()[k] = x1.data()[k] + x2.data()[k];
        }
        ApplyActivation(add.data(), total, act);
        std::memcpy(h_perm.data(), add.data(), total * sizeof(float));
    }

    if (pad_h > 0 || pad_w > 0) {
        const int64_t orig_h = H - pad_h;
        const int64_t orig_w = W - pad_w;
        Tensor trimmed = Tensor::Zeros({bsz, w, orig_h, orig_w});
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int wi = 0; wi < w; ++wi) {
                for (int64_t hi = 0; hi < orig_h; ++hi) {
                    std::memcpy(
                        trimmed.data() + ((bi * w + wi) * orig_h + hi) * orig_w,
                        h_perm.data() + ((bi * w + wi) * H + hi) * W,
                        orig_w * sizeof(float));
                }
            }
        }
        h_perm = trimmed;
        H = orig_h;
        W = orig_w;
    }

    // h (b, w, H, W) -> permute (b, H, W, w) -> proj_q -> act -> proj_out -> y
    Tensor h_back = Tensor::Zeros({bsz, H, W, w});
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int64_t hi = 0; hi < H; ++hi) {
            for (int64_t wj = 0; wj < W; ++wj) {
                for (int wi = 0; wi < w; ++wi) {
                    h_back.data()[((bi * H + hi) * W + wj) * w + wi] =
                        h_perm.data()[((bi * w + wi) * H + hi) * W + wj];
                }
            }
        }
    }

    Tensor q = Tensor::Zeros({bsz, H, W, w});
    Linear(h_back.data(), weights_.proj_q_w.data(), weights_.proj_q_b.data(),
           static_cast<int>(bsz) * static_cast<int>(H) * static_cast<int>(W), w, w, q.data());
    // Fake-quant at proj_q.output (Sprint 3.20).
    apply_fake_quant(q.data(), bsz * H * W * w, "proj_q.output");
    ApplyActivation(q.data(), bsz * H * W * w, act);

    Linear(q.data(), weights_.proj_out_w.data(), weights_.proj_out_b.data(),
           static_cast<int>(bsz) * static_cast<int>(H) * static_cast<int>(W), w,
           static_cast<int>(cfg_.out_channels), y.data());
    // Fake-quant at proj_out.output (Sprint 3.20).
    apply_fake_quant(y.data(),
        bsz * H * W * static_cast<int64_t>(cfg_.out_channels),
        "proj_out.output");

    return Status::Ok();
}

void FNO2d::EnablePerChannelWeightDequant(
    const std::unordered_map<std::string, QuantParams>& per_tensor,
    const std::unordered_map<std::string, PerChannelQuantParams>& per_channel) {
    // Dequantise the weights in place.  After this call,
    // the weights are stored as FP32 in the Tensor
    // buffers and Forward runs as usual.  Mirrors
    // FNO1d::EnablePerChannelWeightDequant (Sprint 3.21).
    DequantiseWeight(weights_.lift_w, per_tensor, per_channel, "lift.weight");
    DequantiseWeight(weights_.lift_b, per_tensor, per_channel, "lift.bias");
    DequantiseWeight(weights_.proj_q_w, per_tensor, per_channel, "proj_q.weight");
    DequantiseWeight(weights_.proj_q_b, per_tensor, per_channel, "proj_q.bias");
    DequantiseWeight(weights_.proj_out_w, per_tensor, per_channel, "proj_out.weight");
    DequantiseWeight(weights_.proj_out_b, per_tensor, per_channel, "proj_out.bias");
    for (size_t i = 0; i < weights_.spec_w_real.size(); ++i) {
        DequantiseWeight(weights_.spec_w_real[i], per_tensor, per_channel,
                          "specs." + std::to_string(i) + ".weights_real");
        DequantiseWeight(weights_.spec_w_imag[i], per_tensor, per_channel,
                          "specs." + std::to_string(i) + ".weights_imag");
    }
    for (size_t i = 0; i < weights_.loc_w.size(); ++i) {
        DequantiseWeight(weights_.loc_w[i], per_tensor, per_channel, "locs." + std::to_string(i) + ".weight");
        DequantiseWeight(weights_.loc_b[i], per_tensor, per_channel, "locs." + std::to_string(i) + ".bias");
    }
}

// =============================================================================
// FNO3d (Stage 2 Sprint 2)
// =============================================================================

namespace {

/// h: (b, w, H, W, D). Apply 3D spectral conv on each (b, w) channel:
/// 3D FFT, truncate to a (modes_h, modes_w, modes_d) box, multiply by
/// per-channel complex weight (in=w, out=w, mh, mw, md), 3D IFFT.
void SpectralConv3d(const float* h, int bsz, int w, int H, int W, int D,
                    int modes_h, int modes_w, int modes_d,
                    const float* w_real, const float* w_imag, float* out) {
    const int half_d = D / 2 + 1;
    // (w, H, W, half_d) interleaved complex spectrum per batch.
    std::vector<float> H_spec(
        static_cast<size_t>(w) * H * W * half_d * 2);
    // (w, H, W, half_d) interleaved output spectrum per batch.
    std::vector<float> Out_ft(
        static_cast<size_t>(w) * H * W * half_d * 2);

    for (int bi = 0; bi < bsz; ++bi) {
        std::memset(Out_ft.data(), 0, Out_ft.size() * sizeof(float));

        // 1) Forward 3D FFT of all w channels for this batch.
        for (int wi = 0; wi < w; ++wi) {
            fft::Rfft3(
                h + (static_cast<int64_t>(bi) * w + wi) * H * W * D, H, W, D,
                H_spec.data()
                    + (static_cast<int64_t>(wi) * H * W * half_d) * 2);
        }
        // 2) Multiply in frequency domain, truncated to the
        // (modes_h, modes_w, modes_d) box. Weight element
        // (wi, wo, mi, mj, mk) lives at index
        // ((((wi*w + wo) * mh + mi) * mw + mj) * md + mk).
        for (int mi = 0; mi < modes_h; ++mi) {
            for (int mj = 0; mj < modes_w; ++mj) {
                for (int mk = 0; mk < modes_d; ++mk) {
                    for (int wo = 0; wo < w; ++wo) {
                        float sum_re = 0.0f, sum_im = 0.0f;
                        for (int wi = 0; wi < w; ++wi) {
                            const float* H_wi = H_spec.data()
                                + (((static_cast<int64_t>(wi) * H + mi) * W + mj) * half_d + mk) * 2;
                            float hr = H_wi[0];
                            float hi = H_wi[1];
                            const float* w_wi = w_real
                                + ((((static_cast<int64_t>(wi) * w) + wo) * modes_h + mi) * modes_w + mj) * modes_d + mk;
                            const float* wi_wi = w_imag
                                + ((((static_cast<int64_t>(wi) * w) + wo) * modes_h + mi) * modes_w + mj) * modes_d + mk;
                            float wr = w_wi[0];
                            float wii = wi_wi[0];
                            sum_re += hr * wr - hi * wii;
                            sum_im += hr * wii + hi * wr;
                        }
                        float* Out_wo = Out_ft.data()
                            + (((static_cast<int64_t>(wo) * H + mi) * W + mj) * half_d + mk) * 2;
                        Out_wo[0] = sum_re;
                        Out_wo[1] = sum_im;
                    }
                }
            }
        }
        // 3) Inverse 3D FFT, writing directly into the output buffer.
        for (int wo = 0; wo < w; ++wo) {
            fft::Irfft3(
                Out_ft.data() + static_cast<int64_t>(wo) * H * W * half_d * 2, H, W, D,
                out + ((static_cast<int64_t>(bi) * w + wo) * H * W) * D);
        }
    }
}

}  // namespace

FNO3d::FNO3d(FNO3dConfig cfg, FNO3dWeights w) : cfg_(cfg), weights_(std::move(w)) {}

Status FNO3d::Forward(const Tensor& x, Tensor& y) const {
    if (x.shape().size() != 5) {
        return Status::ShapeMismatch("expected 5D input (batch, h, w, d, in_ch)");
    }
    int64_t bsz = x.shape()[0];
    int64_t H = x.shape()[1];
    int64_t W = x.shape()[2];
    int64_t D = x.shape()[3];
    int64_t in_ch = x.shape()[4];
    if (in_ch != cfg_.in_channels) {
        return Status::ShapeMismatch("input channels mismatch");
    }
    if (y.shape().size() != 5 || y.shape()[0] != bsz || y.shape()[1] != H ||
        y.shape()[2] != W || y.shape()[3] != D || y.shape()[4] != cfg_.out_channels) {
        return Status::ShapeMismatch("output buffer shape mismatch");
    }
    if (!fft::IsPow2(H) || !fft::IsPow2(W) || !fft::IsPow2(D)) {
        return Status::InvalidArg("FNO3d: H, W, and D must each be a power of two");
    }
    const int w = cfg_.width;
    const int mh = cfg_.modes_h;
    const int mw = cfg_.modes_w;
    const int md = cfg_.modes_d;
    const int L = cfg_.n_layers;
    const std::string& act = cfg_.activation;

    // Lifting: x (b, H, W, D, in_ch) @ lift_w^T (in_ch, w) + lift_b (w)
    //   -> h0 (b, H, W, D, w) -> permute to (b, w, H, W, D)
    Tensor h = Tensor::Zeros({bsz, H, W, D, w});
    Linear(
        x.data(), weights_.lift_w.data(), weights_.lift_b.data(),
        static_cast<int>(bsz) * static_cast<int>(H) * static_cast<int>(W) * static_cast<int>(D),
        static_cast<int>(in_ch), w, h.data());

    Tensor h_perm = Tensor::Zeros({bsz, w, H, W, D});
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int wi = 0; wi < w; ++wi) {
            for (int64_t hi = 0; hi < H; ++hi) {
                for (int64_t wj = 0; wj < W; ++wj) {
                    for (int64_t dk = 0; dk < D; ++dk) {
                        h_perm.data()[(((bi * w + wi) * H + hi) * W + wj) * D + dk] =
                            h.data()[(((bi * H + hi) * W + wj) * D + dk) * w + wi];
                    }
                }
            }
        }
    }

    // Pad to pad_factor multiples along H, W, D.
    int64_t pad_h = 0, pad_w = 0, pad_d = 0;
    Tensor padded;
    if (cfg_.pad_factor > 1) {
        const int pf = cfg_.pad_factor;
        pad_h = (pf - (H % pf)) % pf;
        pad_w = (pf - (W % pf)) % pf;
        pad_d = (pf - (D % pf)) % pf;
        if (pad_h > 0 || pad_w > 0 || pad_d > 0) {
            const int64_t Hp = H + pad_h;
            const int64_t Wp = W + pad_w;
            const int64_t Dp = D + pad_d;
            padded = Tensor::Zeros({bsz, w, Hp, Wp, Dp});
            for (int64_t bi = 0; bi < bsz; ++bi) {
                for (int wi = 0; wi < w; ++wi) {
                    for (int64_t hi = 0; hi < H; ++hi) {
                        for (int64_t wj = 0; wj < W; ++wj) {
                            std::memcpy(
                                padded.data() + (((bi * w + wi) * Hp + hi) * Wp + wj) * Dp,
                                h_perm.data() + (((bi * w + wi) * H + hi) * W + wj) * D,
                                D * sizeof(float));
                        }
                    }
                }
            }
            h_perm = padded;
            H += pad_h;
            W += pad_w;
            D += pad_d;
        }
    }

    Tensor x1 = Tensor::Zeros({bsz, w, H, W, D});
    Tensor x2 = Tensor::Zeros({bsz, w, H, W, D});
    Tensor x2_perm_in = Tensor::Zeros({bsz, H, W, D, w});
    Tensor x2_perm_out = Tensor::Zeros({bsz, H, W, D, w});
    Tensor add = Tensor::Zeros({bsz, w, H, W, D});

    for (int i = 0; i < L; ++i) {
        SpectralConv3d(
            h_perm.data(), static_cast<int>(bsz), w,
            static_cast<int>(H), static_cast<int>(W), static_cast<int>(D),
            mh, mw, md,
            weights_.spec_w_real[i].data(),
            weights_.spec_w_imag[i].data(), x1.data());

        // x2 = permute(h_perm, (0, 2, 3, 4, 1)) @ loc_w^T + loc_b
        //     -> (b, H, W, D, w) -> permute back to (b, w, H, W, D)
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int64_t hi = 0; hi < H; ++hi) {
                for (int64_t wj = 0; wj < W; ++wj) {
                    for (int64_t dk = 0; dk < D; ++dk) {
                        for (int wi = 0; wi < w; ++wi) {
                            x2_perm_in.data()[(((bi * H + hi) * W + wj) * D + dk) * w + wi] =
                                h_perm.data()[(((bi * w + wi) * H + hi) * W + wj) * D + dk];
                        }
                    }
                }
            }
        }
        Linear(
            x2_perm_in.data(), weights_.loc_w[i].data(), weights_.loc_b[i].data(),
            static_cast<int>(bsz) * static_cast<int>(H) * static_cast<int>(W) * static_cast<int>(D),
            w, w, x2_perm_out.data());
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int wi = 0; wi < w; ++wi) {
                for (int64_t hi = 0; hi < H; ++hi) {
                    for (int64_t wj = 0; wj < W; ++wj) {
                        for (int64_t dk = 0; dk < D; ++dk) {
                            x2.data()[(((bi * w + wi) * H + hi) * W + wj) * D + dk] =
                                x2_perm_out.data()[(((bi * H + hi) * W + wj) * D + dk) * w + wi];
                        }
                    }
                }
            }
        }

        // h = act(x1 + x2)
        const int64_t total = bsz * w * H * W * D;
        for (int64_t k = 0; k < total; ++k) {
            add.data()[k] = x1.data()[k] + x2.data()[k];
        }
        ApplyActivation(add.data(), total, act);
        std::memcpy(h_perm.data(), add.data(), total * sizeof(float));
    }

    if (pad_h > 0 || pad_w > 0 || pad_d > 0) {
        const int64_t orig_h = H - pad_h;
        const int64_t orig_w = W - pad_w;
        const int64_t orig_d = D - pad_d;
        Tensor trimmed = Tensor::Zeros({bsz, w, orig_h, orig_w, orig_d});
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int wi = 0; wi < w; ++wi) {
                for (int64_t hi = 0; hi < orig_h; ++hi) {
                    for (int64_t wj = 0; wj < orig_w; ++wj) {
                        std::memcpy(
                            trimmed.data() + (((bi * w + wi) * orig_h + hi) * orig_w + wj) * orig_d,
                            h_perm.data() + (((bi * w + wi) * H + hi) * W + wj) * D,
                            orig_d * sizeof(float));
                    }
                }
            }
        }
        h_perm = trimmed;
        H = orig_h;
        W = orig_w;
        D = orig_d;
    }

    // h (b, w, H, W, D) -> permute (b, H, W, D, w) -> proj_q -> act -> proj_out -> y
    Tensor h_back = Tensor::Zeros({bsz, H, W, D, w});
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int64_t hi = 0; hi < H; ++hi) {
            for (int64_t wj = 0; wj < W; ++wj) {
                for (int64_t dk = 0; dk < D; ++dk) {
                    for (int wi = 0; wi < w; ++wi) {
                        h_back.data()[(((bi * H + hi) * W + wj) * D + dk) * w + wi] =
                            h_perm.data()[(((bi * w + wi) * H + hi) * W + wj) * D + dk];
                    }
                }
            }
        }
    }

    Tensor q = Tensor::Zeros({bsz, H, W, D, w});
    Linear(
        h_back.data(), weights_.proj_q_w.data(), weights_.proj_q_b.data(),
        static_cast<int>(bsz) * static_cast<int>(H) * static_cast<int>(W) * static_cast<int>(D),
        w, w, q.data());
    ApplyActivation(q.data(), bsz * H * W * D * w, act);

    Linear(
        q.data(), weights_.proj_out_w.data(), weights_.proj_out_b.data(),
        static_cast<int>(bsz) * static_cast<int>(H) * static_cast<int>(W) * static_cast<int>(D),
        w, static_cast<int>(cfg_.out_channels), y.data());

    return Status::Ok();
}

// =============================================================================
// DeepONet (Stage 2 Sprint 2)
// =============================================================================

DeepONet::DeepONet(DeepONetConfig cfg, DeepONetWeights w) : cfg_(cfg), weights_(std::move(w)) {}

Status DeepONet::Forward(const Tensor& u, const Tensor& y, Tensor& out) const {
    if (u.shape().size() != 3) {
        return Status::ShapeMismatch("DeepONet: expected 3D u (batch, n_sensor, in_branch)");
    }
    if (y.shape().size() != 3) {
        return Status::ShapeMismatch("DeepONet: expected 3D y (batch, n_query, in_trunk)");
    }
    int64_t bsz = u.shape()[0];
    int64_t n_sensor = u.shape()[1];
    int64_t in_branch = u.shape()[2];
    int64_t n_query = y.shape()[1];
    int64_t in_trunk = y.shape()[2];
    if (in_branch != cfg_.in_branch) {
        return Status::ShapeMismatch("DeepONet: input branch feature dim mismatch");
    }
    if (in_trunk != cfg_.in_trunk) {
        return Status::ShapeMismatch("DeepONet: input trunk feature dim mismatch");
    }
    if (out.shape().size() != 3 || out.shape()[0] != bsz || out.shape()[1] != n_query ||
        out.shape()[2] != cfg_.out_channels) {
        return Status::ShapeMismatch("DeepONet: output buffer shape mismatch");
    }
    const int out_ch = cfg_.out_channels;
    const int latent_dim = cfg_.latent_dim;
    const std::string& act = cfg_.activation;

    // Branch: u (b, n_sensor, in_branch) -> (b, n_sensor, out_ch * latent_dim)
    //        -> mean over n_sensor -> (b, out_ch, latent_dim)
    //
    // The branch MLP has the layer structure:
    //   layers[0]    : (hidden_branch, in_branch)
    //   layers[1..L-2]: (hidden_branch, hidden_branch)
    //   layers[L-1]  : (out_ch * latent_dim, hidden_branch)
    // where L = n_layers_branch. Activations are applied after every layer
    // *except* the final one.
    Tensor b_flat;
    for (int i = 0; i < cfg_.n_layers_branch; ++i) {
        const int in_dim = (i == 0) ? static_cast<int>(in_branch) : cfg_.hidden_branch;
        const int out_dim = (i == cfg_.n_layers_branch - 1)
                                ? (out_ch * latent_dim)
                                : cfg_.hidden_branch;
        Tensor next = Tensor::Zeros({bsz, n_sensor, static_cast<int64_t>(out_dim)});
        const float* in_data = (i == 0) ? u.data() : b_flat.data();
        Linear(
            in_data,
            weights_.branch.weight[i].data(),
            weights_.branch.bias[i].data(),
            static_cast<int>(bsz) * static_cast<int>(n_sensor),
            in_dim, out_dim, next.data());
        if (i < cfg_.n_layers_branch - 1) {
            ApplyActivation(
                next.data(),
                static_cast<int64_t>(bsz) * n_sensor * out_dim,
                act);
        }
        b_flat = next;
    }

    // b_flat is (b, n_sensor, out_ch * latent_dim). Reshape to
    // (b, n_sensor, out_ch, latent_dim) and mean over n_sensor to get
    // (b, out_ch, latent_dim).
    Tensor branch_out = Tensor::Zeros({bsz, static_cast<int64_t>(out_ch), latent_dim});
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int ci = 0; ci < out_ch; ++ci) {
            for (int k = 0; k < latent_dim; ++k) {
                float acc = 0.0f;
                for (int64_t si = 0; si < n_sensor; ++si) {
                    acc += b_flat.data()[((bi * n_sensor + si) * out_ch + ci) * latent_dim + k];
                }
                branch_out.data()[(bi * out_ch + ci) * latent_dim + k] = acc / static_cast<float>(n_sensor);
            }
        }
    }

    // Trunk: y (b, n_query, in_trunk) -> (b, n_query, latent_dim) (with
    // activations on hidden layers, no activation on the output layer).
    Tensor trunk_out_buf = Tensor::Zeros({bsz, n_query, static_cast<int64_t>(in_trunk)});
    std::memcpy(trunk_out_buf.data(), y.data(),
                static_cast<size_t>(bsz * n_query * in_trunk) * sizeof(float));
    for (int i = 0; i < cfg_.n_layers_trunk; ++i) {
        const int in_dim = (i == 0) ? static_cast<int>(in_trunk) : cfg_.hidden_trunk;
        const int out_dim = (i == cfg_.n_layers_trunk - 1)
                                ? latent_dim
                                : cfg_.hidden_trunk;
        Tensor next = Tensor::Zeros({bsz, n_query, static_cast<int64_t>(out_dim)});
        Linear(
            trunk_out_buf.data(),
            weights_.trunk.weight[i].data(),
            weights_.trunk.bias[i].data(),
            static_cast<int>(bsz) * static_cast<int>(n_query),
            in_dim, out_dim, next.data());
        if (i < cfg_.n_layers_trunk - 1) {
            ApplyActivation(
                next.data(),
                static_cast<int64_t>(bsz) * n_query * out_dim,
                act);
        }
        trunk_out_buf = next;
    }
    // Compute the dot product + bias, writing into `out`.
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int64_t i = 0; i < n_query; ++i) {
            for (int ci = 0; ci < out_ch; ++ci) {
                float acc = 0.0f;
                for (int k = 0; k < latent_dim; ++k) {
                    acc += branch_out.data()[(bi * out_ch + ci) * latent_dim + k] *
                           trunk_out_buf.data()[(bi * n_query + i) * latent_dim + k];
                }
                out.data()[(bi * n_query + i) * out_ch + ci] =
                    acc + weights_.bias.data()[ci];
            }
        }
    }

    // (Dot product + bias already computed inside the trunk block above.)

    return Status::Ok();
}

// =============================================================================
// TokenMixer (Stage 2 Sprint 3) — Transolver-style operator learner.
// =============================================================================

TokenMixer::TokenMixer(TokenMixerConfig cfg, TokenMixerWeights w)
    : cfg_(cfg), weights_(std::move(w)) {}

namespace {

/// LayerNorm on the last dim. x: (rows, d). Writes into y (may alias x).
inline void LayerNormLastDim(const float* x, int rows, int d, const float* gamma,
                             const float* beta, float* y) {
    const float eps = 1e-5f;
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * d;
        float* yr = y + static_cast<int64_t>(r) * d;
        float mean = 0.0f;
        for (int i = 0; i < d; ++i) mean += xr[i];
        mean /= static_cast<float>(d);
        float var = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = xr[i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(d);
        float inv = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < d; ++i) {
            yr[i] = (xr[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

/// Softmax along the last dim, in-place. z: (rows, d).
inline void SoftmaxLastDim(float* z, int rows, int d) {
    for (int r = 0; r < rows; ++r) {
        float* zr = z + static_cast<int64_t>(r) * d;
        float m = zr[0];
        for (int i = 1; i < d; ++i) {
            if (zr[i] > m) m = zr[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            zr[i] = std::exp(zr[i] - m);
            sum += zr[i];
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < d; ++i) zr[i] *= inv;
    }
}

}  // namespace

Status TokenMixer::Forward(const Tensor& x, Tensor& y) const {
    if (x.shape().size() != 3) {
        return Status::ShapeMismatch("TokenMixer: expected 3D x (batch, n_points, in_dim)");
    }
    int64_t bsz = x.shape()[0];
    int64_t n = x.shape()[1];
    int64_t in_dim = x.shape()[2];
    if (in_dim != cfg_.in_dim) {
        return Status::ShapeMismatch("TokenMixer: input feature dim mismatch");
    }
    if (n != cfg_.n_points) {
        return Status::ShapeMismatch("TokenMixer: n_points mismatch");
    }
    if (y.shape().size() != 3 || y.shape()[0] != bsz || y.shape()[1] != n ||
        y.shape()[2] != cfg_.out_dim) {
        return Status::ShapeMismatch("TokenMixer: output buffer shape mismatch");
    }
    const int in_d = cfg_.in_dim;
    const int out_d = cfg_.out_dim;
    const int n_p = cfg_.n_points;
    const int n_patches = cfg_.n_patches;
    const int latent = cfg_.latent_dim;
    const int n_heads = cfg_.n_heads;
    const int n_layers = cfg_.n_layers;
    const int head_dim = latent / n_heads;
    if (n_patches <= 0 || n_p % n_patches != 0) {
        return Status::InvalidArg("TokenMixer: n_points must be a multiple of n_patches");
    }
    if (latent % n_heads != 0) {
        return Status::InvalidArg("TokenMixer: latent_dim must be a multiple of n_heads");
    }
    if (n_layers < 1) {
        return Status::InvalidArg("TokenMixer: n_layers must be >= 1");
    }
    if (static_cast<int>(weights_.blocks.size()) != n_layers) {
        return Status::Internal("TokenMixer: weights_.blocks.size() != n_layers");
    }
    const int pp = n_p / n_patches;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::string& act = cfg_.activation;

    // ----- SliceEmbed: (b, n_p, in_d) -> (b, n_patches, latent) -----
    // Step 1: mean-pool over `pp` points per patch.
    // Step 2: Linear to `latent`.
    const int bnp = static_cast<int>(bsz) * n_patches;
    std::vector<float> pooled(static_cast<size_t>(bnp) * in_d, 0.0f);
    std::vector<float> tokens(static_cast<size_t>(bnp) * latent, 0.0f);
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int pi = 0; pi < n_patches; ++pi) {
            for (int ppi = 0; ppi < pp; ++ppi) {
                int64_t x_off = ((bi * n_p + pi * pp + ppi) * in_d);
                float* dst = pooled.data() + (bi * n_patches + pi) * in_d;
                for (int di = 0; di < in_d; ++di) {
                    dst[di] += x.data()[x_off + di] / static_cast<float>(pp);
                }
            }
        }
    }
    Linear(pooled.data(), weights_.slice_embed_proj_w.data(),
           weights_.slice_embed_proj_b.data(), bnp, in_d, latent, tokens.data());

    // ----- Transformer blocks (Stage 2: n_layers = 1 in practice) -----
    // Per-block buffers.
    std::vector<float> ln_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> q_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> k_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> v_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> attn_out(static_cast<size_t>(bnp) * latent);
    std::vector<float> ffn_mid(static_cast<size_t>(bnp) * 2 * latent);
    std::vector<float> ffn_out(static_cast<size_t>(bnp) * latent);

    for (int li = 0; li < n_layers; ++li) {
        const auto& W = weights_.blocks[li];

        // Pre-LN 1.
        LayerNormLastDim(tokens.data(), bnp, latent, W.ln1_w.data(), W.ln1_b.data(),
                         ln_buf.data());

        // Q, K, V projections (3 separate Linears on the same input).
        Linear(ln_buf.data(), W.q_proj_w.data(), W.q_proj_b.data(), bnp, latent, latent,
               q_buf.data());
        Linear(ln_buf.data(), W.k_proj_w.data(), W.k_proj_b.data(), bnp, latent, latent,
               k_buf.data());
        Linear(ln_buf.data(), W.v_proj_w.data(), W.v_proj_b.data(), bnp, latent, latent,
               v_buf.data());

        // Multi-head attention. We loop over batch then heads. Within each
        // (batch, head) we have (n_patches, head_dim) Q, K, V. We compute
        // (n_patches, n_patches) attention scores then multiply by V.
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);
        const int np2 = n_patches * n_patches;
        std::vector<float> attn(np2);
        const int hd = head_dim;
        for (int64_t bi = 0; bi < bsz; ++bi) {
            for (int hi = 0; hi < n_heads; ++hi) {
                // Slice Q, K, V for this (bi, hi).
                // q_buf is (bsz, n_patches, n_heads, head_dim) in row-major:
                // index = ((bi * n_patches + pi) * n_heads + hi) * head_dim + dh
                // Gather into (n_patches, head_dim) views by computing offsets.
                for (int i = 0; i < n_patches; ++i) {
                    for (int j = 0; j < n_patches; ++j) {
                        float acc = 0.0f;
                        for (int dh = 0; dh < hd; ++dh) {
                            int64_t q_off = ((static_cast<int64_t>(bi) * n_patches + i) * n_heads + hi) * hd + dh;
                            int64_t k_off = ((static_cast<int64_t>(bi) * n_patches + j) * n_heads + hi) * hd + dh;
                            acc += q_buf[q_off] * k_buf[k_off];
                        }
                        attn[i * n_patches + j] = acc * scale;
                    }
                }
                SoftmaxLastDim(attn.data(), n_patches, n_patches);
                // out[i, dh] = sum_j attn[i, j] * V[j, dh]
                for (int i = 0; i < n_patches; ++i) {
                    for (int dh = 0; dh < hd; ++dh) {
                        float acc = 0.0f;
                        for (int j = 0; j < n_patches; ++j) {
                            int64_t v_off = ((static_cast<int64_t>(bi) * n_patches + j) * n_heads + hi) * hd + dh;
                            acc += attn[i * n_patches + j] * v_buf[v_off];
                        }
                        int64_t out_off = ((static_cast<int64_t>(bi) * n_patches + i) * n_heads + hi) * hd + dh;
                        attn_out[out_off] = acc;
                    }
                }
            }
        }
        // Output projection + residual.
        std::vector<float> proj_out(static_cast<size_t>(bnp) * latent);
        Linear(attn_out.data(), W.o_proj_w.data(), W.o_proj_b.data(), bnp, latent, latent,
               proj_out.data());
        for (int i = 0; i < bnp * latent; ++i) {
            tokens[i] += proj_out[i];
        }

        // Pre-LN 2 + FFN.
        LayerNormLastDim(tokens.data(), bnp, latent, W.ln2_w.data(), W.ln2_b.data(),
                         ln_buf.data());
        Linear(ln_buf.data(), W.ffn0_w.data(), W.ffn0_b.data(), bnp, latent,
               2 * latent, ffn_mid.data());
        ApplyActivation(ffn_mid.data(), static_cast<int64_t>(bnp) * 2 * latent, act);
        Linear(ffn_mid.data(), W.ffn1_w.data(), W.ffn1_b.data(), bnp, 2 * latent,
               latent, ffn_out.data());
        for (int i = 0; i < bnp * latent; ++i) {
            tokens[i] += ffn_out[i];
        }
    }

    // ----- UnsliceDecode: broadcast tokens to per-point, concat with x features -----
    // tokens: (b, n_patches, latent) -> tokens_rep: (b, n_p, latent) by repeat pp times
    std::vector<float> tokens_rep(static_cast<size_t>(bsz) * n_p * latent);
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int pi = 0; pi < n_patches; ++pi) {
            for (int ppi = 0; ppi < pp; ++ppi) {
                std::memcpy(
                    tokens_rep.data() + ((bi * n_p + pi * pp + ppi) * latent),
                    tokens.data() + ((bi * n_patches + pi) * latent),
                    static_cast<size_t>(latent) * sizeof(float));
            }
        }
    }
    // Concat: (b, n_p, in_d + latent). Apply Linear to latent.
    const int in_plus_lat = in_d + latent;
    std::vector<float> concat_buf(static_cast<size_t>(bsz) * n_p * in_plus_lat);
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int64_t ni = 0; ni < n_p; ++ni) {
            int64_t x_off = (bi * n_p + ni) * in_d;
            int64_t t_off = (bi * n_p + ni) * latent;
            int64_t c_off = (bi * n_p + ni) * in_plus_lat;
            std::memcpy(concat_buf.data() + c_off, x.data() + x_off,
                        static_cast<size_t>(in_d) * sizeof(float));
            std::memcpy(concat_buf.data() + c_off + in_d,
                        tokens_rep.data() + t_off,
                        static_cast<size_t>(latent) * sizeof(float));
        }
    }
    std::vector<float> features(static_cast<size_t>(bsz) * n_p * latent);
    const int bnp_full = static_cast<int>(bsz) * static_cast<int>(n_p);
    Linear(concat_buf.data(), weights_.unslice_proj_w.data(), weights_.unslice_proj_b.data(),
           bnp_full, in_plus_lat, latent, features.data());

    // ----- Head: (b, n_p, latent) -> (b, n_p, out_d) -----
    Linear(features.data(), weights_.head_w.data(), weights_.head_b.data(),
           bnp_full, latent, out_d, y.data());

    return Status::Ok();
}

// =============================================================================
// GraphOp (Stage 2 Sprint 3.2) — GCN-style operator learner.
// =============================================================================

GraphOp::GraphOp(GraphOpConfig cfg, GraphOpWeights w)
    : cfg_(cfg), weights_(std::move(w)) {}

Status GraphOp::Forward(const Tensor& x, Tensor& y) const {
    if (x.shape().size() != 3) {
        return Status::ShapeMismatch("GraphOp: expected 3D x (batch, n_nodes, in_dim)");
    }
    int64_t bsz = x.shape()[0];
    int64_t n = x.shape()[1];
    int64_t in_dim = x.shape()[2];
    if (in_dim != cfg_.in_dim) {
        return Status::ShapeMismatch("GraphOp: input feature dim mismatch");
    }
    if (n != cfg_.n_nodes) {
        return Status::ShapeMismatch("GraphOp: n_nodes mismatch");
    }
    if (y.shape().size() != 3 || y.shape()[0] != bsz || y.shape()[1] != n ||
        y.shape()[2] != cfg_.out_dim) {
        return Status::ShapeMismatch("GraphOp: output buffer shape mismatch");
    }
    const int in_d = cfg_.in_dim;
    const int out_d = cfg_.out_dim;
    const int n_nodes = cfg_.n_nodes;
    const int hidden = cfg_.hidden_dim;
    const int n_layers = cfg_.n_layers;
    if (n_layers < 1) {
        return Status::InvalidArg("GraphOp: n_layers must be >= 1");
    }
    if (static_cast<int>(weights_.blocks.size()) != n_layers) {
        return Status::Internal("GraphOp: weights_.blocks.size() != n_layers");
    }
    if (static_cast<int>(weights_.adj_offsets.size()) != n_nodes + 1) {
        return Status::Internal("GraphOp: adj_offsets size mismatch");
    }
    if (static_cast<int>(weights_.deg_inv.size()) != n_nodes) {
        return Status::Internal("GraphOp: deg_inv size mismatch");
    }
    const std::string& act = cfg_.activation;

    // Lift: x (b, n_nodes, in_d) -> h (b, n_nodes, hidden)
    const int bnh = static_cast<int>(bsz) * n_nodes;
    std::vector<float> h(static_cast<size_t>(bnh) * hidden);
    Linear(x.data(), weights_.lift_w.data(), weights_.lift_b.data(), bnh, in_d,
           hidden, h.data());

    std::vector<float> agg(static_cast<size_t>(bnh) * hidden);
    std::vector<float> h_self(static_cast<size_t>(bnh) * hidden);
    std::vector<float> h_neigh(static_cast<size_t>(bnh) * hidden);

    for (int li = 0; li < n_layers; ++li) {
        const auto& W = weights_.blocks[li];

        // Aggregate neighbours: agg[b, i, d] = (1/deg[i]) * sum_{j in adj[i]} h[b, j, d]
        std::fill(agg.begin(), agg.end(), 0.0f);
        for (int i = 0; i < n_nodes; ++i) {
            const int lo = weights_.adj_offsets[i];
            const int hi = weights_.adj_offsets[i + 1];
            const float di = weights_.deg_inv[i];
            for (int64_t bi = 0; bi < bsz; ++bi) {
                for (int k = lo; k < hi; ++k) {
                    const int j = weights_.adj_indices[k];
                    const int dst_off = static_cast<int>(bi) * n_nodes * hidden + i * hidden;
                    const int src_off = static_cast<int>(bi) * n_nodes * hidden + j * hidden;
                    for (int d = 0; d < hidden; ++d) {
                        agg[dst_off + d] += h[src_off + d];
                    }
                }
                for (int d = 0; d < hidden; ++d) {
                    agg[static_cast<int>(bi) * n_nodes * hidden + i * hidden + d] *= di;
                }
            }
        }

        // W_self @ h + W_neigh @ agg -> act + h
        Linear(h.data(), W.lin_self_w.data(), W.lin_self_b.data(), bnh, hidden,
               hidden, h_self.data());
        Linear(agg.data(), W.lin_neigh_w.data(), W.lin_neigh_b.data(), bnh, hidden,
               hidden, h_neigh.data());
        for (int64_t k = 0; k < bnh * hidden; ++k) {
            h_self[k] = h_self[k] + h_neigh[k];
        }
        ApplyActivation(h_self.data(), bnh * hidden, act);
        for (int64_t k = 0; k < bnh * hidden; ++k) {
            h[k] += h_self[k];
        }
    }

    // Head: (b, n_nodes, hidden) -> (b, n_nodes, out_d)
    Linear(h.data(), weights_.head_w.data(), weights_.head_b.data(),
           bnh, hidden, out_d, y.data());
    return Status::Ok();
}

// =============================================================================
// TokenMixer2D (Stage 2 Sprint 3.6) — 2D Transolver-style operator.
// =============================================================================
//
// Forward: x (b, h, w, in_dim) -> y (b, h, w, out_dim).
// Internally flattens (h, w) to a sequence of length n_points = h*w
// and reuses the 1D TokenMixer mechanism.  Implementation is
// deliberately a thin wrapper around the same per-block code paths
// used by the 1D forward, with explicit flatten / reshape steps.

TokenMixer2D::TokenMixer2D(TokenMixer2DConfig cfg, TokenMixerWeights w)
    : cfg_(cfg), weights_(std::move(w)) {}

Status TokenMixer2D::Forward(const Tensor& x, Tensor& y) const {
    if (x.shape().size() != 4) {
        return Status::ShapeMismatch(
            "TokenMixer2D: expected 4D x (batch, h, w, in_dim)");
    }
    int64_t bsz = x.shape()[0];
    int64_t h = x.shape()[1];
    int64_t w = x.shape()[2];
    int64_t in_dim = x.shape()[3];
    if (in_dim != cfg_.in_dim) {
        return Status::ShapeMismatch("TokenMixer2D: input feature dim mismatch");
    }
    if (h != cfg_.h || w != cfg_.w) {
        return Status::ShapeMismatch("TokenMixer2D: h / w mismatch");
    }
    if (y.shape().size() != 4 || y.shape()[0] != bsz ||
        y.shape()[1] != h || y.shape()[2] != w ||
        y.shape()[3] != cfg_.out_dim) {
        return Status::ShapeMismatch("TokenMixer2D: output buffer shape mismatch");
    }
    const int in_d = cfg_.in_dim;
    const int out_d = cfg_.out_dim;
    const int h_d = cfg_.h;
    const int w_d = cfg_.w;
    const int n_patches = cfg_.n_patches;
    const int latent = cfg_.latent_dim;
    const int n_heads = cfg_.n_heads;
    const int n_layers = cfg_.n_layers;
    const int head_dim = latent / n_heads;
    if (n_patches <= 0 || (h_d * w_d) % n_patches != 0) {
        return Status::InvalidArg("TokenMixer2D: h*w must be a multiple of n_patches");
    }
    if (latent % n_heads != 0) {
        return Status::InvalidArg("TokenMixer2D: latent_dim must be a multiple of n_heads");
    }
    if (n_layers < 1) {
        return Status::InvalidArg("TokenMixer2D: n_layers must be >= 1");
    }
    if (static_cast<int>(weights_.blocks.size()) != n_layers) {
        return Status::Internal("TokenMixer2D: weights_.blocks.size() != n_layers");
    }
    const int n_points = h_d * w_d;
    const int pp = n_points / n_patches;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::string& act = cfg_.activation;

    // Step 1: mean-pool per (patch, point) into (b, n_patches, in_d).
    //         x is (b, h, w, in_d) row-major; for a given batch and
    //         patch, the pp input points are
    //           x[b, patch/pp along h, (patch%pp) along w, :]
    //         with a non-trivial reshape.  We use a temporary
    //         buffer x_patches.
    std::vector<float> x_patched(static_cast<size_t>(bsz) * n_patches * in_d, 0.0f);
    for (int64_t bi = 0; bi < bsz; ++bi) {
        for (int pi = 0; pi < n_patches; ++pi) {
            for (int ppi = 0; ppi < pp; ++ppi) {
                int64_t x_off = ((bi * h_d * w_d) + pi * pp + ppi) * in_d;
                float* dst = x_patched.data() +
                             (bi * n_patches + pi) * in_d;
                for (int di = 0; di < in_d; ++di) {
                    dst[di] += x.data()[x_off + di] / static_cast<float>(pp);
                }
            }
        }
    }

    // Step 2: lift x_patched (b, n_patches, in_d) to tokens
    //         (b, n_patches, latent).
    const int bnp = static_cast<int>(bsz) * n_patches;
    std::vector<float> tokens(static_cast<size_t>(bnp) * latent);
    Linear(x_patched.data(),
           weights_.slice_embed_proj_w.data(),
           weights_.slice_embed_proj_b.data(),
           bnp, in_d, latent, tokens.data());

    // Per-block scratch buffers.
    std::vector<float> ln_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> Q_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> K_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> V_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> head_out_buf(static_cast<size_t>(bnp) * latent);
    std::vector<float> ffn_buf(static_cast<size_t>(bnp) * 2 * latent);
    std::vector<float> ffn_out(static_cast<size_t>(bnp) * latent);

    for (int li = 0; li < n_layers; ++li) {
        const auto& W = weights_.blocks[li];

        // Pre-LN 1.
        LayerNormLastDim(tokens.data(), bnp, latent,
                         W.ln1_w.data(), W.ln1_b.data(), ln_buf.data());

        // Q, K, V projections.
        Linear(ln_buf.data(), W.q_proj_w.data(), W.q_proj_b.data(),
               bnp, latent, latent, Q_buf.data());
        Linear(ln_buf.data(), W.k_proj_w.data(), W.k_proj_b.data(),
               bnp, latent, latent, K_buf.data());
        Linear(ln_buf.data(), W.v_proj_w.data(), W.v_proj_b.data(),
               bnp, latent, latent, V_buf.data());

        // Multi-head attention.
        const int hd = head_dim;
        for (int64_t bi2 = 0; bi2 < bsz; ++bi2) {
            for (int hi = 0; hi < n_heads; ++hi) {
                // Per-(bi, hi) scratch for the n_patches x n_patches
                // attention matrix.  Declared inside the head loop so
                // it is auto-zero-initialised every iteration (the
                // 1D TokenMixer also does this).
                const int np2 = n_patches * n_patches;
                std::vector<float> attn(np2, 0.0f);
                for (int i = 0; i < n_patches; ++i) {
                    for (int j = 0; j < n_patches; ++j) {
                        float acc = 0.0f;
                        for (int dh = 0; dh < hd; ++dh) {
                            int64_t q_off = ((bi2 * n_patches + i) * n_heads + hi) * hd + dh;
                            int64_t k_off = ((bi2 * n_patches + j) * n_heads + hi) * hd + dh;
                            acc += Q_buf[q_off] * K_buf[k_off];
                        }
                        attn[i * n_patches + j] = acc * scale;
                    }
                }
                SoftmaxLastDim(attn.data(), n_patches, n_patches);
                for (int i = 0; i < n_patches; ++i) {
                    for (int dh = 0; dh < hd; ++dh) {
                        float acc = 0.0f;
                        for (int j = 0; j < n_patches; ++j) {
                            int64_t v_off = ((bi2 * n_patches + j) * n_heads + hi) * hd + dh;
                            acc += attn[i * n_patches + j] * V_buf[v_off];
                        }
                        int64_t out_off = ((bi2 * n_patches + i) * n_heads + hi) * hd + dh;
                        head_out_buf[out_off] = acc;
                    }
                }
            }
        }
        // O Linear + residual.
        std::vector<float> proj_out(static_cast<size_t>(bnp) * latent);
        Linear(head_out_buf.data(), W.o_proj_w.data(), W.o_proj_b.data(),
               bnp, latent, latent, proj_out.data());
        for (int i = 0; i < bnp * latent; ++i) tokens[i] += proj_out[i];

        // Pre-LN 2 + FFN.
        LayerNormLastDim(tokens.data(), bnp, latent,
                         W.ln2_w.data(), W.ln2_b.data(), ln_buf.data());
        Linear(ln_buf.data(), W.ffn0_w.data(), W.ffn0_b.data(),
               bnp, latent, 2 * latent, ffn_buf.data());
        ApplyActivation(ffn_buf.data(),
                        static_cast<int64_t>(bnp) * 2 * latent, act);
        Linear(ffn_buf.data(), W.ffn1_w.data(), W.ffn1_b.data(),
               bnp, 2 * latent, latent, ffn_out.data());
        for (int i = 0; i < bnp * latent; ++i) tokens[i] += ffn_out[i];
    }

    // Step 3: UnsliceDecode.
    // Broadcast tokens to per-point: (b, n_points, latent) by
    // repeating each token pp times.
    std::vector<float> tokens_rep(static_cast<size_t>(bsz) * n_points * latent);
    for (int64_t bi2 = 0; bi2 < bsz; ++bi2) {
        for (int pi = 0; pi < n_patches; ++pi) {
            for (int ppi = 0; ppi < pp; ++ppi) {
                std::memcpy(
                    tokens_rep.data() +
                        ((bi2 * n_points + pi * pp + ppi) * latent),
                    tokens.data() + ((bi2 * n_patches + pi) * latent),
                    static_cast<size_t>(latent) * sizeof(float));
            }
        }
    }
    // Concat (in_d, latent) per point: copy x and tokens_rep into
    // a single buffer.
    const int in_plus_lat = in_d + latent;
    std::vector<float> concat_buf(static_cast<size_t>(bsz) * n_points * in_plus_lat);
    for (int64_t bi2 = 0; bi2 < bsz; ++bi2) {
        for (int64_t ni = 0; ni < n_points; ++ni) {
            int64_t x_off = (bi2 * n_points + ni) * in_d;
            int64_t t_off = (bi2 * n_points + ni) * latent;
            int64_t c_off = (bi2 * n_points + ni) * in_plus_lat;
            std::memcpy(concat_buf.data() + c_off, x.data() + x_off,
                        static_cast<size_t>(in_d) * sizeof(float));
            std::memcpy(concat_buf.data() + c_off + in_d,
                        tokens_rep.data() + t_off,
                        static_cast<size_t>(latent) * sizeof(float));
        }
    }
    std::vector<float> features(static_cast<size_t>(bsz) * n_points * latent);
    const int bnp_full = static_cast<int>(bsz) * n_points;
    Linear(concat_buf.data(),
           weights_.unslice_proj_w.data(),
           weights_.unslice_proj_b.data(),
           bnp_full, in_plus_lat, latent, features.data());

    // Head: (b, n_points, latent) -> (b, n_points, out_d).
    std::vector<float> y_flat(static_cast<size_t>(bsz) * n_points * out_d);
    Linear(features.data(),
           weights_.head_w.data(), weights_.head_b.data(),
           bnp_full, latent, out_d, y_flat.data());
    // Reshape (b, n_points, out_d) -> (b, h, w, out_d) — the
    // row-major layouts are identical because n_points == h*w.
    std::memcpy(y.data(), y_flat.data(),
                static_cast<size_t>(bsz) * n_points * out_d * sizeof(float));
    return Status::Ok();
}

// =============================================================================
// GraphOp2D (Stage 2 Sprint 3.6) — 2D GCN-style operator.
// =============================================================================
//
// Forward: x (b, h, w, in_dim) -> y (b, h, w, out_dim).
// Internally flattens (h, w) to a sequence of n_nodes = h*w and
// reuses the 1D GCN per-node aggregation.

GraphOp2D::GraphOp2D(GraphOp2DConfig cfg, GraphOpWeights w)
    : cfg_(cfg), weights_(std::move(w)) {}

Status GraphOp2D::Forward(const Tensor& x, Tensor& y) const {
    if (x.shape().size() != 4) {
        return Status::ShapeMismatch(
            "GraphOp2D: expected 4D x (batch, h, w, in_dim)");
    }
    int64_t bsz = x.shape()[0];
    int64_t h = x.shape()[1];
    int64_t w = x.shape()[2];
    int64_t in_dim = x.shape()[3];
    if (in_dim != cfg_.in_dim) {
        return Status::ShapeMismatch("GraphOp2D: input feature dim mismatch");
    }
    if (h != cfg_.h || w != cfg_.w) {
        return Status::ShapeMismatch("GraphOp2D: h / w mismatch");
    }
    if (y.shape().size() != 4 || y.shape()[0] != bsz ||
        y.shape()[1] != h || y.shape()[2] != w ||
        y.shape()[3] != cfg_.out_dim) {
        return Status::ShapeMismatch("GraphOp2D: output buffer shape mismatch");
    }
    const int in_d = cfg_.in_dim;
    const int out_d = cfg_.out_dim;
    const int h_d = cfg_.h;
    const int w_d = cfg_.w;
    const int hidden = cfg_.hidden_dim;
    const int n_layers = cfg_.n_layers;
    if (n_layers < 1) {
        return Status::InvalidArg("GraphOp2D: n_layers must be >= 1");
    }
    if (static_cast<int>(weights_.blocks.size()) != n_layers) {
        return Status::Internal("GraphOp2D: weights_.blocks.size() != n_layers");
    }
    if (static_cast<int>(weights_.adj_offsets.size()) != h_d * w_d + 1) {
        return Status::Internal("GraphOp2D: adj_offsets size mismatch");
    }
    if (static_cast<int>(weights_.deg_inv.size()) != h_d * w_d) {
        return Status::Internal("GraphOp2D: deg_inv size mismatch");
    }
    const int n_nodes = h_d * w_d;
    const std::string& act = cfg_.activation;

    // Flatten: x is (b, h, w, in_d) row-major == (b, n_nodes, in_d).
    // Lift: (b, n_nodes, in_d) -> (b, n_nodes, hidden).
    const int bnh = static_cast<int>(bsz) * n_nodes;
    std::vector<float> h_feat(static_cast<size_t>(bnh) * hidden);
    Linear(x.data(), weights_.lift_w.data(), weights_.lift_b.data(),
           bnh, in_d, hidden, h_feat.data());

    std::vector<float> agg(static_cast<size_t>(bnh) * hidden);
    std::vector<float> h_self(static_cast<size_t>(bnh) * hidden);
    std::vector<float> h_neigh(static_cast<size_t>(bnh) * hidden);

    for (int li = 0; li < n_layers; ++li) {
        const auto& W = weights_.blocks[li];

        // Aggregate neighbours: agg[b, i, d] = (1/deg[i]) * sum_{j in adj[i]} h_feat[b, j, d].
        std::fill(agg.begin(), agg.end(), 0.0f);
        for (int i = 0; i < n_nodes; ++i) {
            const int lo = weights_.adj_offsets[i];
            const int hi = weights_.adj_offsets[i + 1];
            const float di = weights_.deg_inv[i];
            for (int64_t bi2 = 0; bi2 < bsz; ++bi2) {
                for (int k = lo; k < hi; ++k) {
                    const int j = weights_.adj_indices[k];
                    const int dst_off = static_cast<int>(bi2) * n_nodes * hidden + i * hidden;
                    const int src_off = static_cast<int>(bi2) * n_nodes * hidden + j * hidden;
                    for (int d = 0; d < hidden; ++d) {
                        agg[dst_off + d] += h_feat[src_off + d];
                    }
                }
                for (int d = 0; d < hidden; ++d) {
                    agg[static_cast<int>(bi2) * n_nodes * hidden + i * hidden + d] *= di;
                }
            }
        }

        Linear(h_feat.data(), W.lin_self_w.data(), W.lin_self_b.data(),
               bnh, hidden, hidden, h_self.data());
        Linear(agg.data(), W.lin_neigh_w.data(), W.lin_neigh_b.data(),
               bnh, hidden, hidden, h_neigh.data());
        for (int64_t k = 0; k < bnh * hidden; ++k) h_self[k] += h_neigh[k];
        ApplyActivation(h_self.data(), bnh * hidden, act);
        for (int64_t k = 0; k < bnh * hidden; ++k) h_feat[k] += h_self[k];
    }

    // Head: (b, n_nodes, hidden) -> (b, n_nodes, out_d).
    // Reshape (b, n_nodes, out_d) -> (b, h, w, out_d) — identical
    // row-major layout.
    Linear(h_feat.data(), weights_.head_w.data(), weights_.head_b.data(),
           bnh, hidden, out_d, y.data());
    return Status::Ok();
}

}  // namespace nflow::fno
