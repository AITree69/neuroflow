// =============================================================================
// NeuroFlow C++ Runtime — high-level runtime (load + run)
// =============================================================================

#include "neuroflow/runtime.h"

#include <cstring>
#include <utility>

#include "neuroflow/fno.h"
#include "neuroflow/ir_loader.h"
#include "neuroflow/tensor.h"

namespace nflow {

Status LoadNeuroIr(const std::string& path, LoadedModel& out) {
    // Stage 1 supports only the native binary format (.nneuroir).
    return ir_native::LoadBinary(path, out);
}

Status InferenceRuntime::Init(const std::string& ir_path) {
    Status s = LoadNeuroIr(ir_path, model_);
    if (!s.ok()) return s;
    if (model_.op == "FNO1d") {
        fno_ = std::make_unique<fno::FNO1d>(model_.fno1d_cfg, model_.fno1d_weights);
        if (model_.quant_enabled) {
            fno_->EnableFakeQuant(model_.activation_qparams);
            fno_->EnablePerTokenActivation(
                model_.activation_per_token_qparams);
            fno_->EnableFP8Activation(
                model_.activation_fp8_qparams);
            fno_->EnablePerChannelWeightDequant(
                model_.weight_qparams,
                model_.weight_per_channel_qparams);
        }
        return Status::Ok();
    }
    if (model_.op == "FNO2d") {
        fno2d_ = std::make_unique<fno::FNO2d>(model_.fno2d_cfg, model_.fno2d_weights);
        if (model_.quant_enabled) {
            fno2d_->EnablePerChannelWeightDequant(
                model_.weight_qparams,
                model_.weight_per_channel_qparams);
            fno2d_->EnableFakeQuant(model_.activation_qparams);
            fno2d_->EnableFP8Activation(
                model_.activation_fp8_qparams);
        }
        return Status::Ok();
    }
    if (model_.op == "FNO3d") {
        fno3d_ = std::make_unique<fno::FNO3d>(model_.fno3d_cfg, model_.fno3d_weights);
        return Status::Ok();
    }
    if (model_.op == "DeepONet") {
        deeponet_ = std::make_unique<fno::DeepONet>(model_.deeponet_cfg, model_.deeponet_weights);
        return Status::Ok();
    }
    if (model_.op == "TokenMixer") {
        tokenmixer_ = std::make_unique<fno::TokenMixer>(model_.tokenmixer_cfg, model_.tokenmixer_weights);
        return Status::Ok();
    }
    if (model_.op == "GraphOp") {
        graphop_ = std::make_unique<fno::GraphOp>(model_.graphop_cfg, model_.graphop_weights);
        return Status::Ok();
    }
    if (model_.op == "TokenMixer2D") {
        tokenmixer2d_ = std::make_unique<fno::TokenMixer2D>(
            model_.tokenmixer2d_cfg, model_.tokenmixer2d_weights);
        return Status::Ok();
    }
    if (model_.op == "GraphOp2D") {
        graphop2d_ = std::make_unique<fno::GraphOp2D>(
            model_.graphop2d_cfg, model_.graphop2d_weights);
        return Status::Ok();
    }
    return Status::UnsupportedOp("runtime does not support op: " + model_.op);
}

Status InferenceRuntime::Create(const std::string& ir_path,
                                std::unique_ptr<InferenceRuntime>& out) {
    out.reset(new InferenceRuntime());
    return out->Init(ir_path);
}

Status InferenceRuntime::Run(const float* x, const std::vector<int64_t>& x_shape,
                             float* y, const std::vector<int64_t>& y_shape) {
    // x is read-only, so Wrap (no copy) is safe. y is the output buffer; we
    // wrap it without copying so writes propagate to the caller's buffer.
    Tensor x_t = Tensor::Wrap(x_shape, x);
    Tensor y_t = Tensor::WrapMutable(y_shape, y);
    if (fno_) {
        return fno_->Forward(x_t, y_t);
    }
    if (fno2d_) {
        return fno2d_->Forward(x_t, y_t);
    }
    if (fno3d_) {
        return fno3d_->Forward(x_t, y_t);
    }
    return Status::Internal("runtime does not support op: " + model_.op);
}

Status InferenceRuntime::RunDeepONet(const float* u, const std::vector<int64_t>& u_shape,
                                     const float* y, const std::vector<int64_t>& y_shape,
                                     float* out, const std::vector<int64_t>& out_shape) {
    if (!deeponet_) {
        return Status::Internal("DeepONet not initialized");
    }
    Tensor u_t = Tensor::Wrap(u_shape, u);
    Tensor y_t = Tensor::Wrap(y_shape, y);
    Tensor out_t = Tensor::WrapMutable(out_shape, out);
    return deeponet_->Forward(u_t, y_t, out_t);
}

Status InferenceRuntime::RunTokenMixer(const float* x, const std::vector<int64_t>& x_shape,
                                       float* y, const std::vector<int64_t>& y_shape) {
    if (!tokenmixer_) {
        return Status::Internal("TokenMixer not initialized");
    }
    Tensor x_t = Tensor::Wrap(x_shape, x);
    Tensor y_t = Tensor::WrapMutable(y_shape, y);
    return tokenmixer_->Forward(x_t, y_t);
}

Status InferenceRuntime::RunGraphOp(const float* x, const std::vector<int64_t>& x_shape,
                                    float* y, const std::vector<int64_t>& y_shape) {
    if (!graphop_) {
        return Status::Internal("GraphOp not initialized");
    }
    Tensor x_t = Tensor::Wrap(x_shape, x);
    Tensor y_t = Tensor::WrapMutable(y_shape, y);
    return graphop_->Forward(x_t, y_t);
}

Status InferenceRuntime::RunTokenMixer2D(const float* x, const std::vector<int64_t>& x_shape,
                                       float* y, const std::vector<int64_t>& y_shape) {
    if (!tokenmixer2d_) {
        return Status::Internal("TokenMixer2D not initialized");
    }
    Tensor x_t = Tensor::Wrap(x_shape, x);
    Tensor y_t = Tensor::WrapMutable(y_shape, y);
    return tokenmixer2d_->Forward(x_t, y_t);
}

Status InferenceRuntime::RunGraphOp2D(const float* x, const std::vector<int64_t>& x_shape,
                                     float* y, const std::vector<int64_t>& y_shape) {
    if (!graphop2d_) {
        return Status::Internal("GraphOp2D not initialized");
    }
    Tensor x_t = Tensor::Wrap(x_shape, x);
    Tensor y_t = Tensor::WrapMutable(y_shape, y);
    return graphop2d_->Forward(x_t, y_t);
}

void InferenceRuntime::EnableInt8Gemm() {
    // Sprint 3.28: opt-in INT8 GEMM dispatch for FNO1d.
    // Only effective when the model is FNO1d AND the
    // per-channel weight qparams are present (Sprint
    // 3.10+ calibration) AND the per-tensor activation
    // qparams are present (Sprint 3.11+ calibration).
    // For the per-token / FP8 activation paths, the
    // Linear GEMM falls back to FP32 automatically
    // (the dispatch helper checks for those cases).
    if (!fno_) return;
    fno_->EnableInt8Gemm(model_.weight_per_channel_qparams);
}

bool InferenceRuntime::IsInt8GemmEnabled() const {
    if (!fno_) return false;
    return fno_->IsInt8GemmEnabled();
}

}  // namespace nflow
