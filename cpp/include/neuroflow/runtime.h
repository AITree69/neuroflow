// =============================================================================
// NeuroFlow C++ Runtime — high-level InferenceRuntime
// =============================================================================

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuroflow/fno.h"
#include "neuroflow/quant_types.h"
#include "neuroflow/tensor.h"

namespace nflow {

/// In-memory representation of a loaded NeuroIR v0 model.
struct LoadedModel {
    std::string op;
    fno::FNO1dConfig fno1d_cfg;
    fno::FNO1dWeights fno1d_weights;
    fno::FNO2dConfig fno2d_cfg;
    fno::FNO2dWeights fno2d_weights;
    fno::FNO3dConfig fno3d_cfg;
    fno::FNO3dWeights fno3d_weights;
    fno::DeepONetConfig deeponet_cfg;
    fno::DeepONetWeights deeponet_weights;
    fno::TokenMixerConfig tokenmixer_cfg;
    fno::TokenMixerWeights tokenmixer_weights;
    fno::GraphOpConfig graphop_cfg;
    fno::GraphOpWeights graphop_weights;
    fno::TokenMixer2DConfig tokenmixer2d_cfg;
    fno::TokenMixerWeights tokenmixer2d_weights;
    fno::GraphOp2DConfig graphop2d_cfg;
    fno::GraphOpWeights graphop2d_weights;
    // Optional INT8 W8A8 fake-quant block (v0.15.0).
    bool quant_enabled = false;
    // Per-tensor scale / zero_point for every weight name
    // in the model.  Activation qparams live in
    // `activation_qparams` keyed by the layer's output
    // name (e.g. "lift.output", "locs.0.output").
    std::unordered_map<std::string, QuantParams> weight_qparams;
    std::unordered_map<std::string, QuantParams> activation_qparams;
    // Per-channel weight qparams (v0.16.0).  When a
    // weight name is present here, the per-channel
    // scheme is used for that weight; otherwise the
    // per-tensor `weight_qparams` is used.
    std::unordered_map<std::string, PerChannelQuantParams>
        weight_per_channel_qparams;
    // Per-token activation qparams (v0.17.0).  Keyed
    // by the activation name (e.g. "locs.0.output").
    // When present, the per-token scheme is used for
    // the post-activation fake-quant round-trip; the
    // per-tensor `activation_qparams` is ignored.
    std::unordered_map<std::string, PerTokenQuantParams>
        activation_per_token_qparams;
    // FP8 E4M3 per-tensor activation qparams (v0.21.0).
    // Keyed by the activation name.  When present, FP8
    // fake-quant is applied instead of INT8.
    std::unordered_map<std::string, FP8E4M3Params>
        activation_fp8_qparams;
};

/// Loads a NeuroIR v0 JSON spec from disk.
Status LoadNeuroIr(const std::string& path, LoadedModel& out);

/// High-level runtime: load once, run many inferences.
class InferenceRuntime {
public:
    /// Construct from a path. Returns a Status.
    static Status Create(const std::string& ir_path,
                         std::unique_ptr<InferenceRuntime>& out);

    /// Run inference. x and y are raw pointers to float32 buffers.
    /// Shapes must be (1, n, in_channels) and (1, n, out_channels).
    Status Run(const float* x, const std::vector<int64_t>& x_shape,
               float* y, const std::vector<int64_t>& y_shape);

    /// Run a DeepONet: u (n_sensor, in_branch) + y (n_query, in_trunk) -> out
    /// (n_query, out_channels). Only valid when op() == "DeepONet".
    Status RunDeepONet(const float* u, const std::vector<int64_t>& u_shape,
                        const float* y, const std::vector<int64_t>& y_shape,
                        float* out, const std::vector<int64_t>& out_shape);

    /// Run a TokenMixer: x (batch, n_points, in_dim) -> y (batch, n_points, out_dim).
    /// Only valid when op() == "TokenMixer".
    Status RunTokenMixer(const float* x, const std::vector<int64_t>& x_shape,
                          float* y, const std::vector<int64_t>& y_shape);

    /// Run a GraphOp: x (batch, n_nodes, in_dim) -> y (batch, n_nodes, out_dim).
    /// Only valid when op() == "GraphOp".
    Status RunGraphOp(const float* x, const std::vector<int64_t>& x_shape,
                       float* y, const std::vector<int64_t>& y_shape);

    /// Run a TokenMixer2D: x (batch, h, w, in_dim) -> y (batch, h, w, out_dim).
    /// Only valid when op() == "TokenMixer2D".
    Status RunTokenMixer2D(const float* x, const std::vector<int64_t>& x_shape,
                            float* y, const std::vector<int64_t>& y_shape);

    /// Run a GraphOp2D: x (batch, h, w, in_dim) -> y (batch, h, w, out_dim).
    /// Only valid when op() == "GraphOp2D".
    Status RunGraphOp2D(const float* x, const std::vector<int64_t>& x_shape,
                         float* y, const std::vector<int64_t>& y_shape);

    const std::string& op() const { return model_.op; }

    /// Enable inline INT8 GEMM (Sprint 3.28) for the
    /// underlying FNO1d.  When enabled, the per-layer
    /// `locs.{i}.weight` Linear is routed through
    /// `int8_gemm::LinearForward` (VNNI / AVX2 / scalar)
    /// instead of the FP32 path.  Per-tensor activation
    /// qparam is required; per-token / FP8 activation
    /// paths fall back to FP32 automatically.  No-op if
    /// `op()` is not "FNO1d" or if the per-channel
    /// qparam map is empty.
    void EnableInt8Gemm();

    /// True iff `EnableInt8Gemm` is in effect and at
    /// least one Linear layer has an INT8 cache entry.
    bool IsInt8GemmEnabled() const;

private:
    InferenceRuntime() = default;
    Status Init(const std::string& ir_path);

    LoadedModel model_;
    std::unique_ptr<fno::FNO1d> fno_;
    std::unique_ptr<fno::FNO2d> fno2d_;
    std::unique_ptr<fno::FNO3d> fno3d_;
    std::unique_ptr<fno::DeepONet> deeponet_;
    std::unique_ptr<fno::TokenMixer> tokenmixer_;
    std::unique_ptr<fno::GraphOp> graphop_;
    std::unique_ptr<fno::TokenMixer2D> tokenmixer2d_;
    std::unique_ptr<fno::GraphOp2D> graphop2d_;
};

}  // namespace nflow
