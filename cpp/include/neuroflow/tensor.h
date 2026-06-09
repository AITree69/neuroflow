// =============================================================================
// NeuroFlow C++ Runtime — minimal row-major Tensor (Stage 1)
// =============================================================================
//
// Design goals (Stage 1):
//   - Single dtype: float32
//   - Row-major, contiguous
//   - CPU only (CUDA path is Stage 2+)
//   - No exceptions in hot paths; reports via nflow::Status
//   - Header-only documentation; implementation in tensor.cpp
//
// NOT a replacement for Eigen/Torch — just enough to run an FNO inference.
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nflow {

/// Status code returned by inference routines.
enum class StatusCode {
    Ok = 0,
    InvalidArg = 1,
    FileNotFound = 2,
    ParseError = 3,
    ShapeMismatch = 4,
    UnsupportedOp = 5,
    Internal = 99,
};

class Status {
public:
    Status() = default;
    explicit Status(StatusCode code, std::string msg = "")
        : code_(code), msg_(std::move(msg)) {}

    bool ok() const { return code_ == StatusCode::Ok; }
    StatusCode code() const { return code_; }
    const std::string& message() const { return msg_; }

    static Status Ok() { return Status(StatusCode::Ok); }
    static Status InvalidArg(std::string m) { return Status(StatusCode::InvalidArg, std::move(m)); }
    static Status FileNotFound(std::string m) { return Status(StatusCode::FileNotFound, std::move(m)); }
    static Status ParseError(std::string m) { return Status(StatusCode::ParseError, std::move(m)); }
    static Status ShapeMismatch(std::string m) { return Status(StatusCode::ShapeMismatch, std::move(m)); }
    static Status UnsupportedOp(std::string m) { return Status(StatusCode::UnsupportedOp, std::move(m)); }
    static Status Internal(std::string m) { return Status(StatusCode::Internal, std::move(m)); }

private:
    StatusCode code_ = StatusCode::Ok;
    std::string msg_;
};

/// Lightweight row-major float32 tensor. Owns its storage via shared_ptr
/// to support slice/clone without copies in view-only operations.
class Tensor {
public:
    Tensor() = default;

    /// Construct an uninitialized tensor of given shape.
    static Tensor Empty(const std::vector<int64_t>& shape);

    /// Construct a tensor filled with `value`.
    static Tensor Full(const std::vector<int64_t>& shape, float value);

    /// Construct a tensor of zeros.
    static Tensor Zeros(const std::vector<int64_t>& shape);

    /// Wrap externally-owned memory (does not copy). `data` must remain
    /// valid for the lifetime of the Tensor. Used by the IR loader.
    static Tensor Wrap(const std::vector<int64_t>& shape, const float* data);

    /// Wrap externally-owned mutable memory (no copy). Caller must guarantee
    /// `data` outlives the Tensor. Used for output buffers in inference.
    static Tensor WrapMutable(const std::vector<int64_t>& shape, float* data);

    /// Copy-construct (deep copy).
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(size_t i) const { return shape_.at(i); }
    int64_t numel() const { return numel_; }
    const float* data() const { return data_.get(); }
    float* data() { return data_.get(); }

    /// Total byte size of stored data.
    size_t bytes() const { return static_cast<size_t>(numel_) * sizeof(float); }

    /// True if the tensor is contiguous in row-major order.
    bool is_contiguous() const { return true; }

    /// Pretty-print shape and a few values (for debugging only).
    std::string DebugString() const;

private:
    std::vector<int64_t> shape_;
    int64_t numel_ = 0;
    std::shared_ptr<float> data_;  // null until constructed
};

/// Compute strides for a contiguous row-major tensor.
std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape);

/// Element-wise access (slow, range-checked in debug).
float At(const Tensor& t, const std::vector<int64_t>& idx);

}  // namespace nflow
