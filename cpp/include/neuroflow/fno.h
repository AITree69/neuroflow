// =============================================================================
// NeuroFlow C++ Runtime — FNO1d inference (Stage 1)
// =============================================================================
//
// Mirrors neuroflow/nn/fno.py:FNO1d exactly. Single-threaded, float32.
// Activation: GELU (tanh approx) or ReLU, selected at construction.
// =============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuroflow/quant_types.h"
#include "neuroflow/tensor.h"

namespace nflow::fno {

struct FNO1dConfig {
    int in_channels = 0;
    int out_channels = 0;
    int width = 0;
    int modes = 0;
    int n_layers = 0;
    std::string activation = "gelu";  // "gelu" or "relu"
    int pad_factor = 1;
};

struct FNO1dWeights {
    // Lifting
    Tensor lift_w;   // (w, in_ch)
    Tensor lift_b;   // (w,)

    // Per-layer
    // Spectral weights: shape (in_channels, out_channels, modes) = (w, w, modes)
    // in row-major. Matches the PyTorch reference (see SpectralConv1d).
    std::vector<Tensor> spec_w_real;  // (in=w, out=w, modes)
    std::vector<Tensor> spec_w_imag;  // (in=w, out=w, modes)
    std::vector<Tensor> loc_w;        // (out=w, in=w) — PyTorch nn.Linear layout
    std::vector<Tensor> loc_b;        // (out=w,)

    // Output projection
    Tensor proj_q_w;  // (w, w)
    Tensor proj_q_b;  // (w,)
    Tensor proj_out_w;  // (out_ch, w)
    Tensor proj_out_b;  // (out_ch,)
};

// =============================================================================
// Sprint 3.28: inline INT8 GEMM layer cache for FNO1d
// =============================================================================
// One entry per Linear in FNO1d.  The `W_int8` buffer
// is the FP32 weight quantised to INT8 (one-shot at
// `EnableInt8Gemm` time) using the supplied per-channel
// qparams.  `scale_W` and `zp_W` are the per-channel
// scales / zero_points.  `bias` is the FP32 bias (NOT
// quantised).  Activation quantisation (scale_A, zp_A)
// is looked up at call time from the FNO1d's
// `activation_qparams_` map, because the input
// distribution changes per-call (the previous layer's
// output).
// =============================================================================

struct Int8GemmLayer {
    // (out_features, in_features) row-major.
    std::vector<int8_t> W_int8;
    std::vector<float> scale_W;   // (out_features,)
    std::vector<int32_t> zp_W;    // (out_features,)
    // Sprint 3.28: pre-computed per-row signed sum
    // (W_int8[o, :] - zp_W[o]).  Cached at
    // `EnableInt8Gemm` time so the per-row
    // `LinearForward` call skips the O(out*in)
    // pre-sum pre-pass (essential for the
    // bsz*n-calls-per-forward FNO1d use case).
    // Owned by this layer; freed when the
    // Int8GemmLayer is destroyed.
    std::vector<int32_t> sum_W_per_row;
    int in_features = 0;
    int out_features = 0;
};

// Keyed by layer name (matches the per-channel qparam
// map keys): "lift.weight", "locs.0.weight", ...,
// "proj_q.weight", "proj_out.weight".
using Int8GemmCache = std::unordered_map<std::string, Int8GemmLayer>;

class FNO1d {
public:
    FNO1d() = default;
    FNO1d(FNO1dConfig cfg, FNO1dWeights w);

    /// Run inference. x: (batch, n, in_channels), y: (batch, n, out_channels).
    /// y must already be allocated with the correct shape.
    Status Forward(const Tensor& x, Tensor& y) const;

    const FNO1dConfig& config() const { return cfg_; }
    const FNO1dWeights& weights() const { return weights_; }

    /// Enable INT8 (W8A8 fake-quant) inference.  When
    /// enabled, `Forward` applies a quantise → dequantise
    /// round-trip at every Linear layer's output (key
    /// `"<layer>.output"`) using the supplied per-tensor
    /// scale / zero_point.
    void EnableFakeQuant(
        const std::unordered_map<std::string, QuantParams>&
            activation_qparams) {
        quant_enabled_ = true;
        activation_qparams_ = activation_qparams;
    }

    /// Enable per-token (per-spatial-point) activation
    /// fake-quant.  When an entry is present for a key
    /// in `activation_per_token_qparams`, the per-token
    /// scheme is used for that layer's output
    /// (overriding the per-tensor scheme).  The C{++}
    /// runtime looks up the right `(scale, zero_point)`
    /// per (n_idx, w_idx) at fake-quant time.
    void EnablePerTokenActivation(
        const std::unordered_map<std::string, PerTokenQuantParams>&
            per_token_qparams) {
        activation_per_token_qparams_ = per_token_qparams;
    }

    /// Enable FP8 E4M3 per-tensor activation fake-quant.
    /// When an entry is present for a key in
    /// `activation_fp8_qparams`, FP8 is used for that
    /// layer's output (overriding INT8 schemes).
    void EnableFP8Activation(
        const std::unordered_map<std::string, FP8E4M3Params>&
            fp8_qparams) {
        activation_fp8_qparams_ = fp8_qparams;
    }

    /// Dequantise the (INT8-on-disk) weight tensors in
    /// place using the supplied per-tensor and per-
    /// channel quantisation parameters.  Per-channel
    /// parameters take precedence when present.
    /// This is a one-shot operation at construction
    /// time — afterwards, the weights are in FP32 and
    /// `Forward` runs as usual.
    void EnablePerChannelWeightDequant(
        const std::unordered_map<std::string, QuantParams>&
            per_tensor_qparams,
        const std::unordered_map<std::string, PerChannelQuantParams>&
            per_channel_qparams);

    /// Enable inline INT8 GEMM (per-channel W + per-tensor A).
    /// Quantises the FP32 weights to INT8 (one-shot,
    /// per-channel) and routes the Linear GEMM in
    /// `Forward` through `int8_gemm::LinearForward` (which
    /// selects VNNI / AVX2 / scalar at compile time).
    /// The activation qparam is looked up per-Linear
    /// from `activation_qparams_` (the same map used
    /// by `EnableFakeQuant`) under the layer's INPUT
    /// key (e.g. `lift.output` for the lift Linear,
    /// `locs.{i-1}.output` for `locs.{i}.weight`).
    /// Per-token and FP8 activation paths fall back to
    /// the existing FP32 Linear (no speedup there).
    void EnableInt8Gemm(
        const std::unordered_map<std::string, PerChannelQuantParams>&
            per_channel_qparams);

    bool IsFakeQuantEnabled() const { return quant_enabled_; }
    bool IsInt8GemmEnabled() const { return int8_gemm_enabled_; }

    // Sprint 3.28: the int8_gemm dispatch helper
    // (LinearDispatchTryInt8 in fno.cpp) needs to read
    // the per-token / FP8 qparam maps and the
    // int8_gemm_cache_ for a given key.  Expose the
    // bare minimum as public accessors (no setters, no
    // mutation).
    bool HasPerTokenQparam(const std::string& k) const {
        return activation_per_token_qparams_.count(k) > 0;
    }
    bool HasFP8Qparam(const std::string& k) const {
        return activation_fp8_qparams_.count(k) > 0;
    }
    const QuantParams* GetActivationQparam(const std::string& k) const {
        auto it = activation_qparams_.find(k);
        return it == activation_qparams_.end() ? nullptr : &it->second;
    }
    const Int8GemmLayer* GetInt8GemmLayer(const std::string& k) const {
        auto it = int8_gemm_cache_.find(k);
        return it == int8_gemm_cache_.end() ? nullptr : &it->second;
    }

private:
    FNO1dConfig cfg_;
    FNO1dWeights weights_;
    // INT8 fake-quant state (mutable so it can be set
    // post-construction without breaking const-correctness
    // of Forward).
    mutable bool quant_enabled_ = false;
    mutable std::unordered_map<std::string, QuantParams>
        activation_qparams_;
    mutable std::unordered_map<std::string, PerTokenQuantParams>
        activation_per_token_qparams_;
    mutable std::unordered_map<std::string, FP8E4M3Params>
        activation_fp8_qparams_;
    // Sprint 3.28: inline INT8 GEMM cache.  When
    // `int8_gemm_enabled_` is true and the per-tensor
    // activation qparam is available (i.e. NOT per-token
    // and NOT FP8), the Linear GEMM in `Forward` is
    // routed through `int8_gemm::LinearForward` (which
    // selects VNNI / AVX2 / scalar at compile time).
    mutable bool int8_gemm_enabled_ = false;
    mutable Int8GemmCache int8_gemm_cache_;
};

// =============================================================================
// FNO2d (Stage 2 Sprint 1 Phase 2).
// =============================================================================
//
// Mirrors neuroflow/nn/fno2d.py:FNO2d. Spectral weights are stored as
// (in_channels, out_channels, modes_h, modes_w) in row-major order — same
// axis convention as SpectralConv1d (in, out, modes) and the FNO paper.
// =============================================================================

struct FNO2dConfig {
    int in_channels = 0;
    int out_channels = 0;
    int width = 0;
    int modes_h = 0;
    int modes_w = 0;
    int n_layers = 0;
    std::string activation = "gelu";  // "gelu" or "relu"
    int pad_factor = 1;
};

struct FNO2dWeights {
    // Lifting
    Tensor lift_w;   // (w, in_ch)
    Tensor lift_b;   // (w,)

    // Per-layer
    // Spectral weights: shape (in=w, out=w, modes_h, modes_w) row-major.
    std::vector<Tensor> spec_w_real;
    std::vector<Tensor> spec_w_imag;
    std::vector<Tensor> loc_w;   // (out=w, in=w) — PyTorch nn.Linear layout
    std::vector<Tensor> loc_b;   // (out=w,)

    // Output projection
    Tensor proj_q_w;   // (w, w)
    Tensor proj_q_b;   // (w,)
    Tensor proj_out_w; // (out_ch, w)
    Tensor proj_out_b; // (out_ch,)
};

class FNO2d {
public:
    FNO2d() = default;
    FNO2d(FNO2dConfig cfg, FNO2dWeights w);

    /// Run inference. x: (batch, h, w, in_channels), y: (batch, h, w, out_channels).
    /// y must already be allocated with the correct shape.
    Status Forward(const Tensor& x, Tensor& y) const;

    const FNO2dConfig& config() const { return cfg_; }
    const FNO2dWeights& weights() const { return weights_; }

    /// Enable FP8 E4M3 per-tensor activation fake-quant
    /// (Sprint 3.19 — mirrors FNO1d's path).  When an
    /// entry is present for a key in
    /// `activation_fp8_qparams`, FP8 is used for that
    /// layer's output (overriding INT8 schemes).
    void EnableFP8Activation(
        const std::unordered_map<std::string, FP8E4M3Params>&
            fp8_qparams) {
        quant_enabled_ = true;
        activation_fp8_qparams_ = fp8_qparams;
    }

    /// Enable INT8 (W8A8 fake-quant) inference — per-tensor.
    void EnableFakeQuant(
        const std::unordered_map<std::string, QuantParams>&
            activation_qparams) {
        quant_enabled_ = true;
        activation_qparams_ = activation_qparams;
    }

    /// Dequantise the (INT8-on-disk) weight tensors in
    /// place using the supplied per-tensor and per-
    /// channel quantisation parameters (Sprint 3.21 —
    /// mirrors FNO1d).  Per-channel parameters take
    /// precedence when present.  This is a one-shot
    /// operation at load time; afterwards, the weights
    /// are in FP32 and `Forward` runs as usual.
    void EnablePerChannelWeightDequant(
        const std::unordered_map<std::string, QuantParams>&
            per_tensor,
        const std::unordered_map<std::string, PerChannelQuantParams>&
            per_channel);

    bool IsFakeQuantEnabled() const { return quant_enabled_; }

private:
    FNO2dConfig cfg_;
    FNO2dWeights weights_;
    // INT8 / FP8 fake-quant state (mutable so it can be
    // set post-construction without breaking const-
    // correctness of Forward).
    mutable bool quant_enabled_ = false;
    mutable std::unordered_map<std::string, QuantParams>
        activation_qparams_;
    mutable std::unordered_map<std::string, FP8E4M3Params>
        activation_fp8_qparams_;
};

// =============================================================================
// FNO3d (Stage 2 Sprint 2).
// =============================================================================
//
// Mirrors neuroflow/nn/fno3d.py:FNO3d. Spectral weights are stored as
// (in_channels, out_channels, modes_h, modes_w, modes_d) in row-major
// order — same axis convention as SpectralConv1d (in, out, modes) and
// SpectralConv2d (in, out, mh, mw).
// =============================================================================

struct FNO3dConfig {
    int in_channels = 0;
    int out_channels = 0;
    int width = 0;
    int modes_h = 0;
    int modes_w = 0;
    int modes_d = 0;
    int n_layers = 0;
    std::string activation = "gelu";  // "gelu" or "relu"
    int pad_factor = 1;
};

struct FNO3dWeights {
    // Lifting
    Tensor lift_w;   // (w, in_ch)
    Tensor lift_b;   // (w,)

    // Per-layer
    // Spectral weights: shape (in=w, out=w, modes_h, modes_w, modes_d) row-major.
    std::vector<Tensor> spec_w_real;
    std::vector<Tensor> spec_w_imag;
    std::vector<Tensor> loc_w;   // (out=w, in=w) — PyTorch nn.Linear layout
    std::vector<Tensor> loc_b;   // (out=w,)

    // Output projection
    Tensor proj_q_w;   // (w, w)
    Tensor proj_q_b;   // (w,)
    Tensor proj_out_w; // (out_ch, w)
    Tensor proj_out_b; // (out_ch,)
};

class FNO3d {
public:
    FNO3d() = default;
    FNO3d(FNO3dConfig cfg, FNO3dWeights w);

    /// Run inference. x: (batch, h, w, d, in_channels), y: (batch, h, w, d, out_channels).
    /// y must already be allocated with the correct shape.
    Status Forward(const Tensor& x, Tensor& y) const;

    const FNO3dConfig& config() const { return cfg_; }
    const FNO3dWeights& weights() const { return weights_; }

private:
    FNO3dConfig cfg_;
    FNO3dWeights weights_;
};

// =============================================================================
// DeepONet (Stage 2 Sprint 2).
// =============================================================================
//
// Mirrors neuroflow/nn/deeponet.py:DeepONet. Two MLPs:
//   - branch: (b, n_sensor, in_branch) -> (b, out_ch, latent_dim) (mean over n_sensor)
//   - trunk:  (b, n_query,  in_trunk)  -> (b, n_query,  latent_dim)
// Output:  out[b, i, c] = sum_k branch[b, c, k] * trunk[b, i, k] + bias[c]
// Weights stored as PyTorch nn.Linear convention (out, in) per layer, plus
// a per-output-channel bias.
// =============================================================================

struct DeepONetConfig {
    int in_branch = 0;
    int in_trunk = 0;
    int latent_dim = 0;
    int out_channels = 0;
    int hidden_branch = 0;
    int hidden_trunk = 0;
    int n_layers_branch = 0;
    int n_layers_trunk = 0;
    std::string activation = "gelu";  // "gelu" or "relu"
};

struct DeepONetLayerWeights {
    std::vector<Tensor> weight;  // (out, in) per layer (PyTorch nn.Linear layout)
    std::vector<Tensor> bias;    // (out,) per layer
};

struct DeepONetWeights {
    DeepONetLayerWeights branch;
    DeepONetLayerWeights trunk;
    Tensor bias;  // (out_channels,)
};

class DeepONet {
public:
    DeepONet() = default;
    DeepONet(DeepONetConfig cfg, DeepONetWeights w);

    /// Run inference. u: (batch, n_sensor, in_branch), y: (batch, n_query, in_trunk),
    /// out: (batch, n_query, out_channels). `out` must be allocated.
    Status Forward(const Tensor& u, const Tensor& y, Tensor& out) const;

    const DeepONetConfig& config() const { return cfg_; }
    const DeepONetWeights& weights() const { return weights_; }

private:
    DeepONetConfig cfg_;
    DeepONetWeights weights_;
};

// =============================================================================
// TokenMixer (Stage 2 Sprint 3) — Transolver-style operator learner.
// =============================================================================
//
// Mirrors neuroflow/nn/tokenmixer.py:TokenMixer exactly. Architecture:
//
//   1. SliceEmbed:  (b, n_points, in_dim)  -> (b, n_patches, latent_dim)
//                  via mean pool over `points_per_patch` + Linear.
//   2. For each block (n_layers, currently 1 for Stage 2):
//        Pre-LN -> Q,K,V Linear -> multi-head self-attn over (n_patches, head_dim)
//        -> O Linear -> residual
//        Pre-LN -> FFN (Linear, act, Linear) -> residual
//   3. UnsliceDecode: broadcast patch embedding back to per-point,
//        concat with per-point in_dim features, Linear -> latent_dim.
//   4. Head: Linear(latent_dim -> out_dim).
//
// Forward: x (batch, n_points, in_dim) -> y (batch, n_points, out_dim).
//
// Constraints (Stage 2):
//   - n_layers == 1 (multi-block is a Stage 3 extension; the IR weight
//     layout and forward already iterate over n_layers for future use).
//   - n_points must be a multiple of n_patches.
//   - latent_dim must be a multiple of n_heads.
// =============================================================================

struct TokenMixerConfig {
    int in_dim = 0;
    int out_dim = 0;
    int n_points = 0;
    int n_patches = 0;
    int latent_dim = 0;
    int n_heads = 0;
    int n_layers = 0;
    std::string activation = "gelu";  // "gelu" or "relu"
};

struct TokenMixerBlockWeights {
    Tensor ln1_w;       // (latent_dim,)
    Tensor ln1_b;       // (latent_dim,)
    Tensor q_proj_w;    // (latent_dim, latent_dim)
    Tensor q_proj_b;    // (latent_dim,)
    Tensor k_proj_w;    // (latent_dim, latent_dim)
    Tensor k_proj_b;    // (latent_dim,)
    Tensor v_proj_w;    // (latent_dim, latent_dim)
    Tensor v_proj_b;    // (latent_dim,)
    Tensor o_proj_w;    // (latent_dim, latent_dim)
    Tensor o_proj_b;    // (latent_dim,)
    Tensor ln2_w;       // (latent_dim,)
    Tensor ln2_b;       // (latent_dim,)
    Tensor ffn0_w;      // (2 * latent_dim, latent_dim)
    Tensor ffn0_b;      // (2 * latent_dim,)
    Tensor ffn1_w;      // (latent_dim, 2 * latent_dim)
    Tensor ffn1_b;      // (latent_dim,)
};

struct TokenMixerWeights {
    Tensor slice_embed_proj_w;  // (latent_dim, in_dim)
    Tensor slice_embed_proj_b;  // (latent_dim,)
    std::vector<TokenMixerBlockWeights> blocks;
    Tensor unslice_proj_w;      // (latent_dim, in_dim + latent_dim)
    Tensor unslice_proj_b;      // (latent_dim,)
    Tensor head_w;              // (out_dim, latent_dim)
    Tensor head_b;              // (out_dim,)
};

class TokenMixer {
public:
    TokenMixer() = default;
    TokenMixer(TokenMixerConfig cfg, TokenMixerWeights w);

    /// Run inference. x: (batch, n_points, in_dim), y: (batch, n_points, out_dim).
    /// `y` must be allocated with the correct shape.
    Status Forward(const Tensor& x, Tensor& y) const;

    const TokenMixerConfig& config() const { return cfg_; }
    const TokenMixerWeights& weights() const { return weights_; }

private:
    TokenMixerConfig cfg_;
    TokenMixerWeights weights_;
};

// =============================================================================
// GraphOp (Stage 2 Sprint 3.2) — GCN-style operator learner.
// =============================================================================
//
// Mirrors neuroflow/nn/graph_op.py:GraphOp exactly. Architecture:
//
//   1. lift:    (b, n_nodes, in_dim) -> (b, n_nodes, hidden_dim)
//   2. For each block (n_layers, currently 1 for Stage 2):
//        agg[i] = sum_{j in adj[i]} h[j] * deg_inv[i]
//        h' = act(W_self @ h + W_neigh @ agg) + h
//   3. head:    (b, n_nodes, hidden_dim) -> (b, n_nodes, out_dim)
//
// Forward: x (batch, n_nodes, in_dim) -> y (batch, n_nodes, out_dim).
//
// Constraints (Stage 2):
//   - n_layers == 1.
//   - Graph topology is encoded as three float32 weight arrays
//     (`graph.adj_offsets`, `graph.adj_indices`, `graph.deg_inv`)
//     that the IR loader casts back to int32 / float32. The CSR
//     representation (offsets, indices) matches a 1D line graph
//     with self-loops; arbitrary CSR graphs are supported as long
//     as `deg_inv` was precomputed for them.
// =============================================================================

struct GraphOpConfig {
    int in_dim = 0;
    int out_dim = 0;
    int n_nodes = 0;
    int hidden_dim = 0;
    int n_layers = 0;
    std::string activation = "gelu";  // "gelu" or "relu"
};

struct GraphOpBlockWeights {
    Tensor lin_self_w;  // (hidden_dim, hidden_dim)
    Tensor lin_self_b;  // (hidden_dim,)
    Tensor lin_neigh_w; // (hidden_dim, hidden_dim)
    Tensor lin_neigh_b; // (hidden_dim,)
};

struct GraphOpWeights {
    Tensor lift_w;          // (hidden_dim, in_dim)
    Tensor lift_b;          // (hidden_dim,)
    std::vector<GraphOpBlockWeights> blocks;
    Tensor head_w;          // (out_dim, hidden_dim)
    Tensor head_b;          // (out_dim,)

    // Graph topology (cast back from float32 storage in the IR loader).
    std::vector<int32_t> adj_offsets;  // length n_nodes + 1
    std::vector<int32_t> adj_indices;  // length sum(deg)
    std::vector<float> deg_inv;        // length n_nodes
};

class GraphOp {
public:
    GraphOp() = default;
    GraphOp(GraphOpConfig cfg, GraphOpWeights w);

    /// Run inference. x: (batch, n_nodes, in_dim), y: (batch, n_nodes, out_dim).
    /// `y` must be allocated with the correct shape.
    Status Forward(const Tensor& x, Tensor& y) const;

    const GraphOpConfig& config() const { return cfg_; }
    const GraphOpWeights& weights() const { return weights_; }

private:
    GraphOpConfig cfg_;
    GraphOpWeights weights_;
};

// =============================================================================
// TokenMixer2D (Stage 2 Sprint 3.6) — 2D Transolver-style operator.
// =============================================================================
//
// Mirrors neuroflow/nn/tokenmixer2d.py:TokenMixer2D exactly. The C++
// forward accepts 4D input (b, h, w, in_dim) and produces 4D output
// (b, h, w, out_dim); internally it flattens the (h, w) grid into a
// sequence of length n_points = h*w and reuses the same SliceEmbed
// / multi-head self-attention / UnsliceDecode / head pipeline as
// the 1D `TokenMixer` (see the 1D version's documentation above).
//
// Weight layout mirrors the 1D TokenMixer (22 weight names per
// block: slice_embed.proj.{weight,bias}, blocks.{i}.{ln1,q_proj,
// k_proj,v_proj,o_proj,ln2,ffn0,ffn1}.{weight,bias}, unslice.proj
// .{weight,bias}, head.{weight,bias}).  No additional graph-topology
// entries are needed (the 2D version is a regular grid, not an
// arbitrary graph).
// =============================================================================

struct TokenMixer2DConfig {
    int in_dim = 0;
    int out_dim = 0;
    int h = 0;
    int w = 0;
    int n_patches = 0;
    int latent_dim = 0;
    int n_heads = 0;
    int n_layers = 0;
    std::string activation = "gelu";  // "gelu" or "relu"
};

// TokenMixer2D uses the same per-block weight layout as the 1D
// TokenMixer, so we alias the existing struct.
using TokenMixer2DWeights = TokenMixerWeights;

class TokenMixer2D {
public:
    TokenMixer2D() = default;
    TokenMixer2D(TokenMixer2DConfig cfg, TokenMixerWeights w);

    /// Run inference. x: (batch, h, w, in_dim), y: (batch, h, w, out_dim).
    /// `y` must be allocated with the correct shape.
    Status Forward(const Tensor& x, Tensor& y) const;

    const TokenMixer2DConfig& config() const { return cfg_; }
    const TokenMixerWeights& weights() const { return weights_; }

private:
    TokenMixer2DConfig cfg_;
    TokenMixerWeights weights_;
};

// =============================================================================
// GraphOp2D (Stage 2 Sprint 3.6) — 2D GCN-style operator.
// =============================================================================
//
// Mirrors neuroflow/nn/graph_op2d.py:GraphOp2D exactly. The C++ forward
// accepts 4D input (b, h, w, in_dim) and produces 4D output
// (b, h, w, out_dim); internally it flattens the (h, w) grid into a
// sequence of length n_nodes = h*w and reuses the same per-node
// degree-normalised neighbour-aggregation mechanism as the 1D
// `GraphOp`.  The graph topology (`graph.adj_offsets`,
// `graph.adj_indices`, `graph.deg_inv`) is supplied as three
// float32 weight entries (cast back to int32 / float32 at use
// time), matching the 1D GraphOp convention.
// =============================================================================

struct GraphOp2DConfig {
    int in_dim = 0;
    int out_dim = 0;
    int h = 0;
    int w = 0;
    int hidden_dim = 0;
    int n_layers = 0;
    std::string activation = "gelu";  // "gelu" or "relu"
};

// GraphOp2D uses the same weight layout as the 1D GraphOp, so
// we alias the existing struct (which already carries
// adj_offsets / adj_indices / deg_inv).
using GraphOp2DWeights = GraphOpWeights;

class GraphOp2D {
public:
    GraphOp2D() = default;
    GraphOp2D(GraphOp2DConfig cfg, GraphOpWeights w);

    /// Run inference. x: (batch, h, w, in_dim), y: (batch, h, w, out_dim).
    /// `y` must be allocated with the correct shape.
    Status Forward(const Tensor& x, Tensor& y) const;

    const GraphOp2DConfig& config() const { return cfg_; }
    const GraphOpWeights& weights() const { return weights_; }

private:
    GraphOp2DConfig cfg_;
    GraphOp2DWeights weights_;
};

}  // namespace nflow::fno
