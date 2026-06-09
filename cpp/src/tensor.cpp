// =============================================================================
// NeuroFlow C++ Runtime — Tensor implementation
// =============================================================================

#include "neuroflow/tensor.h"

#include <cstring>
#include <sstream>

namespace nflow {

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t s = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = s;
        s *= shape[i - 1];
    }
    return strides;
}

Tensor Tensor::Empty(const std::vector<int64_t>& shape) {
    Tensor t;
    t.shape_ = shape;
    t.numel_ = 1;
    for (auto d : shape) t.numel_ *= d;
    if (t.numel_ > 0) {
        t.data_ = std::shared_ptr<float>(new float[static_cast<size_t>(t.numel_)](),
                                         [](float* p) { delete[] p; });
    }
    return t;
}

Tensor Tensor::Zeros(const std::vector<int64_t>& shape) {
    Tensor t = Empty(shape);
    if (t.numel_ > 0) {
        std::memset(t.data_.get(), 0, static_cast<size_t>(t.numel_) * sizeof(float));
    }
    return t;
}

Tensor Tensor::Full(const std::vector<int64_t>& shape, float value) {
    Tensor t = Empty(shape);
    if (t.numel_ > 0) {
        for (int64_t i = 0; i < t.numel_; ++i) t.data_.get()[i] = value;
    }
    return t;
}

Tensor Tensor::Wrap(const std::vector<int64_t>& shape, const float* data) {
    Tensor t;
    t.shape_ = shape;
    t.numel_ = 1;
    for (auto d : shape) t.numel_ *= d;
    // We still take a copy here so that the Tensor owns its memory.
    // In Stage 1 simplicity > efficiency. Stage 2: add a no-copy Wrap variant.
    if (t.numel_ > 0 && data != nullptr) {
        t.data_ = std::shared_ptr<float>(new float[static_cast<size_t>(t.numel_)](),
                                         [](float* p) { delete[] p; });
        std::memcpy(t.data_.get(), data, static_cast<size_t>(t.numel_) * sizeof(float));
    }
    return t;
}

Tensor Tensor::WrapMutable(const std::vector<int64_t>& shape, float* data) {
    Tensor t;
    t.shape_ = shape;
    t.numel_ = 1;
    for (auto d : shape) t.numel_ *= d;
    // No-copy wrap. The shared_ptr has a no-op deleter; the external buffer
    // must outlive the Tensor. Used for output buffers in inference.
    if (t.numel_ > 0 && data != nullptr) {
        t.data_ = std::shared_ptr<float>(data, [](float*) { /* no-op, caller owns */ });
    }
    return t;
}

std::string Tensor::DebugString() const {
    std::ostringstream os;
    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        os << shape_[i] << (i + 1 == shape_.size() ? "" : ",");
    }
    os << "], numel=" << numel_ << ")";
    return os.str();
}

float At(const Tensor& t, const std::vector<int64_t>& idx) {
    if (idx.size() != t.shape().size()) {
        throw std::invalid_argument("At: index rank mismatch");
    }
    auto strides = ContiguousStrides(t.shape());
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= t.shape()[i]) {
            throw std::out_of_range("At: index out of range");
        }
        off += idx[i] * strides[i];
    }
    return t.data()[off];
}

}  // namespace nflow
