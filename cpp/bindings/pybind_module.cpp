// =============================================================================
// NeuroFlow C++ Runtime — pybind11 bindings
// =============================================================================
//
// Build with: cmake -DNFLOW_BUILD_PYBIND=ON
// Produces: neuroflow_cpp (Python importable module)
//
// Entry points (Stage 2):
//   - infer(model_path, input_path, output_path)
//   - infer_arrays(model_path, x) -> y
// Both dispatch to FNO1d (3D input) or FNO2d (4D input) by op_code in the IR.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "neuroflow/npy_io.h"
#include "neuroflow/runtime.h"
#include "neuroflow/tensor.h"
#include "neuroflow/fp8_e4m3.h"
#define NFLOW_HAS_PYBIND11
#define NFLOW_PYBIND_FORWARD_FP8
#include "neuroflow/fp8_e4m3_pybind.h"

namespace py = pybind11;

namespace {

nflow::Tensor NumpyToTensor(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    std::vector<int64_t> shape(buf.ndim);
    for (int i = 0; i < buf.ndim; ++i) shape[i] = buf.shape[i];
    return nflow::Tensor::Wrap(shape, static_cast<const float*>(buf.ptr));
}

py::array_t<float> TensorToNumpy(const nflow::Tensor& t) {
    std::vector<ssize_t> shape(t.shape().begin(), t.shape().end());
    std::vector<ssize_t> strides(shape.size());
    ssize_t s = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = s * static_cast<ssize_t>(sizeof(float));
        s *= shape[i - 1];
    }
    py::array_t<float> out(shape, strides);
    std::memcpy(out.mutable_data(), t.data(), t.bytes());
    return out;
}

}  // namespace

PYBIND11_MODULE(neuroflow_cpp, m) {
    m.doc() = "NeuroFlow C++ runtime - FNO1d / FNO2d / FNO3d inference";

    m.def("infer",
          [](const std::string& model_path, const std::string& input_path,
             const std::string& output_path) {
              nflow::LoadedModel lm;
              auto s = nflow::LoadNeuroIr(model_path, lm);
              if (!s.ok()) throw std::runtime_error("load IR failed: " + s.message());
              nflow::Tensor x;
              s = nflow::npy::Read(input_path, x);
              if (!s.ok()) throw std::runtime_error("read input failed: " + s.message());
              size_t expected_rank = (lm.op == "FNO1d") ? 3u
                                    : (lm.op == "FNO2d") ? 4u
                                    : (lm.op == "FNO3d") ? 5u
                                    : (lm.op == "TokenMixer") ? 3u
                                    : (lm.op == "GraphOp") ? 3u
                                    : (lm.op == "TokenMixer2D") ? 4u
                                    : (lm.op == "GraphOp2D") ? 4u
                                    : 0u;
              if (expected_rank == 0 || x.shape().size() != expected_rank) {
                  throw std::runtime_error("input rank mismatch for op " + lm.op);
              }
              int64_t out_ch = (lm.op == "FNO1d")    ? lm.fno1d_cfg.out_channels
                              : (lm.op == "FNO2d") ? lm.fno2d_cfg.out_channels
                              : (lm.op == "FNO3d") ? lm.fno3d_cfg.out_channels
                              : (lm.op == "TokenMixer") ? lm.tokenmixer_cfg.out_dim
                              : (lm.op == "GraphOp")    ? lm.graphop_cfg.out_dim
                              : (lm.op == "TokenMixer2D") ? lm.tokenmixer2d_cfg.out_dim
                                                          : lm.graphop2d_cfg.out_dim;
              std::vector<int64_t> y_shape = x.shape();
              if (!y_shape.empty()) y_shape.back() = out_ch;
              std::unique_ptr<nflow::InferenceRuntime> rt;
              s = nflow::InferenceRuntime::Create(model_path, rt);
              if (!s.ok()) throw std::runtime_error("create runtime failed: " + s.message());
              nflow::Tensor y = nflow::Tensor::Zeros(y_shape);
              if (lm.op == "TokenMixer") {
                  s = rt->RunTokenMixer(x.data(), x.shape(), y.data(), y.shape());
              } else if (lm.op == "GraphOp") {
                  s = rt->RunGraphOp(x.data(), x.shape(), y.data(), y.shape());
              } else if (lm.op == "TokenMixer2D") {
                  s = rt->RunTokenMixer2D(x.data(), x.shape(), y.data(), y.shape());
              } else if (lm.op == "GraphOp2D") {
                  s = rt->RunGraphOp2D(x.data(), x.shape(), y.data(), y.shape());
              } else {
                  s = rt->Run(x.data(), x.shape(), y.data(), y.shape());
              }
              if (!s.ok()) throw std::runtime_error("infer failed: " + s.message());
              s = nflow::npy::Write(output_path, y);
              if (!s.ok()) throw std::runtime_error("write output failed: " + s.message());
              return std::string("ok");
          },
          py::arg("model_path"), py::arg("input_path"), py::arg("output_path"),
          "Run inference: load .nneuroir, read .npy, write .npy");

    m.def("infer_arrays",
          [](const std::string& model_path,
             py::array_t<float, py::array::c_style | py::array::forcecast> x) -> py::array_t<float> {
              nflow::LoadedModel lm;
              auto s = nflow::LoadNeuroIr(model_path, lm);
              if (!s.ok()) throw std::runtime_error("load IR failed: " + s.message());
              nflow::Tensor x_t = NumpyToTensor(x);
              size_t expected_rank = (lm.op == "FNO1d") ? 3u
                                    : (lm.op == "FNO2d") ? 4u
                                    : (lm.op == "FNO3d") ? 5u
                                    : (lm.op == "TokenMixer") ? 3u
                                    : (lm.op == "GraphOp") ? 3u
                                    : (lm.op == "TokenMixer2D") ? 4u
                                    : (lm.op == "GraphOp2D") ? 4u
                                    : 0u;
              if (expected_rank == 0 || x_t.shape().size() != expected_rank) {
                  throw std::runtime_error("input rank mismatch for op " + lm.op);
              }
              int64_t out_ch = (lm.op == "FNO1d")    ? lm.fno1d_cfg.out_channels
                              : (lm.op == "FNO2d") ? lm.fno2d_cfg.out_channels
                              : (lm.op == "FNO3d") ? lm.fno3d_cfg.out_channels
                              : (lm.op == "TokenMixer") ? lm.tokenmixer_cfg.out_dim
                              : (lm.op == "GraphOp")    ? lm.graphop_cfg.out_dim
                              : (lm.op == "TokenMixer2D") ? lm.tokenmixer2d_cfg.out_dim
                                                          : lm.graphop2d_cfg.out_dim;
              std::vector<int64_t> y_shape = x_t.shape();
              if (!y_shape.empty()) y_shape.back() = out_ch;
              std::unique_ptr<nflow::InferenceRuntime> rt;
              s = nflow::InferenceRuntime::Create(model_path, rt);
              if (!s.ok()) throw std::runtime_error("create runtime failed: " + s.message());
              nflow::Tensor y = nflow::Tensor::Zeros(y_shape);
              if (lm.op == "TokenMixer") {
                  s = rt->RunTokenMixer(x_t.data(), x_t.shape(), y.data(), y.shape());
              } else if (lm.op == "GraphOp") {
                  s = rt->RunGraphOp(x_t.data(), x_t.shape(), y.data(), y.shape());
              } else if (lm.op == "TokenMixer2D") {
                  s = rt->RunTokenMixer2D(x_t.data(), x_t.shape(), y.data(), y.shape());
              } else if (lm.op == "GraphOp2D") {
                  s = rt->RunGraphOp2D(x_t.data(), x_t.shape(), y.data(), y.shape());
              } else {
                  s = rt->Run(x_t.data(), x_t.shape(), y.data(), y.shape());
              }
              if (!s.ok()) throw std::runtime_error("infer failed: " + s.message());
              return TensorToNumpy(y);
          },
          py::arg("model_path"), py::arg("x"),
          "Run inference in-memory: load .nneuroir, take numpy array, return numpy array");

    py::class_<nflow::InferenceRuntime, std::unique_ptr<nflow::InferenceRuntime>>(
        m, "InferenceRuntime", "Persistent runtime (load once, run many)")
        .def(py::init([](const std::string& ir_path) {
            std::unique_ptr<nflow::InferenceRuntime> rt;
            auto s = nflow::InferenceRuntime::Create(ir_path, rt);
            if (!s.ok()) throw std::runtime_error("create runtime failed: " + s.message());
            return rt.release();  // pybind11 takes ownership
        }))
        .def("op", &nflow::InferenceRuntime::op)
        .def("enable_int8_gemm", &nflow::InferenceRuntime::EnableInt8Gemm,
             "Sprint 3.28: route FNO1d's per-layer Linear through "
             "int8_gemm::LinearForward (VNNI/AVX2/scalar). "
             "No-op for non-FNO1d models.")
        .def("is_int8_gemm_enabled", &nflow::InferenceRuntime::IsInt8GemmEnabled,
             "True iff enable_int8_gemm was effective.")
        .def("run",
             [](nflow::InferenceRuntime& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> x,
                py::array_t<float, py::array::c_style | py::array::forcecast> y) {
                 py::buffer_info x_buf = x.request();
                 py::buffer_info y_buf = y.request();
                 std::vector<int64_t> x_shape(
                     x_buf.shape.begin(), x_buf.shape.end());
                 std::vector<int64_t> y_shape(
                     y_buf.shape.begin(), y_buf.shape.end());
                 auto s = self.Run(
                     static_cast<const float*>(x_buf.ptr), x_shape,
                     static_cast<float*>(y_buf.ptr), y_shape);
                 if (!s.ok()) {
                     throw std::runtime_error("Run failed: " + s.message());
                 }
             },
             py::arg("x"), py::arg("y"),
             "Run FNO1d/2d/3d inference.  x and y are "
             "pre-allocated float32 numpy arrays with the "
             "expected input/output shapes.");

    // DeepONet has two inputs (u, y) and one output. Provide a dedicated
    // entry point so the single-input `infer_arrays` above stays simple.
    m.def("infer_deeponet_arrays",
          [](const std::string& model_path,
             py::array_t<float, py::array::c_style | py::array::forcecast> u,
             py::array_t<float, py::array::c_style | py::array::forcecast> y) -> py::array_t<float> {
              nflow::LoadedModel lm;
              auto s = nflow::LoadNeuroIr(model_path, lm);
              if (!s.ok()) throw std::runtime_error("load IR failed: " + s.message());
              if (lm.op != "DeepONet") {
                  throw std::runtime_error("infer_deeponet_arrays: model op is " + lm.op
                                           + ", not DeepONet");
              }
              nflow::Tensor u_t = NumpyToTensor(u);
              nflow::Tensor y_t = NumpyToTensor(y);
              if (u_t.shape().size() != 3) {
                  throw std::runtime_error("DeepONet u must be 3D (batch, n_sensor, in_branch)");
              }
              if (y_t.shape().size() != 3) {
                  throw std::runtime_error("DeepONet y must be 3D (batch, n_query, in_trunk)");
              }
              int64_t bsz = y_t.shape()[0];
              int64_t n_query = y_t.shape()[1];
              std::vector<int64_t> out_shape = {bsz, n_query, lm.deeponet_cfg.out_channels};
              std::unique_ptr<nflow::InferenceRuntime> rt;
              s = nflow::InferenceRuntime::Create(model_path, rt);
              if (!s.ok()) throw std::runtime_error("create runtime failed: " + s.message());
              nflow::Tensor out_t = nflow::Tensor::Zeros(out_shape);
              s = rt->RunDeepONet(u_t.data(), u_t.shape(), y_t.data(), y_t.shape(),
                                   out_t.data(), out_t.shape());
              if (!s.ok()) throw std::runtime_error("infer failed: " + s.message());
              return TensorToNumpy(out_t);
          },
          py::arg("model_path"), py::arg("u"), py::arg("y"),
          "Run DeepONet inference in-memory: load .nneuroir, take u + y numpy arrays, return output array");

    // Sprint 3.30: FP8 E4M3 IEEE-754 bit-level conversions exposed
    // to Python so the cross-language parity test can verify C++
    // vs Python agreement on FP8 quantise/dequantise.
    nflow::RegisterFp8E4M3Bindings(m);
}
