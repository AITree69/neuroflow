// =============================================================================
// NeuroFlow C++ Runtime — CLI: nflow_infer
// =============================================================================
//
// Usage:
//   nflow_infer --model foo.nneuroir --input x.npy --output y.npy
//
// I/O convention:
//   - FNO1d: input (1, n, in_channels), output (1, n, out_channels)
//   - FNO2d: input (1, h, w, in_channels), output (1, h, w, out_channels)
//   - FNO3d: input (1, h, w, d, in_channels), output (1, h, w, d, out_channels)
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "neuroflow/npy_io.h"
#include "neuroflow/runtime.h"
#include "neuroflow/tensor.h"

namespace {

void PrintUsage() {
    std::cout << "Usage: nflow_infer --model <file.nneuroir> "
                 "--input <x.npy> --output <y.npy>\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, input_path, output_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--help" || a == "-h") {
            PrintUsage();
            return 0;
        }
    }
    if (model_path.empty() || input_path.empty() || output_path.empty()) {
        PrintUsage();
        return 1;
    }

    // Read the model up front so we know the op and out_channels.
    nflow::LoadedModel lm;
    auto s = nflow::LoadNeuroIr(model_path, lm);
    if (!s.ok()) {
        std::cerr << "model load failed: " << s.message() << std::endl;
        return 1;
    }
    int64_t expected_rank = 0;
    int64_t out_ch = 0;
    if (lm.op == "FNO1d") {
        expected_rank = 3;
        out_ch = lm.fno1d_cfg.out_channels;
    } else if (lm.op == "FNO2d") {
        expected_rank = 4;
        out_ch = lm.fno2d_cfg.out_channels;
    } else if (lm.op == "FNO3d") {
        expected_rank = 5;
        out_ch = lm.fno3d_cfg.out_channels;
    } else {
        std::cerr << "unsupported op in IR: " << lm.op << std::endl;
        return 1;
    }

    nflow::Tensor x;
    s = nflow::npy::Read(input_path, x);
    if (!s.ok()) {
        std::cerr << "input read failed: " << s.message() << std::endl;
        return 1;
    }
    if (static_cast<int64_t>(x.shape().size()) != expected_rank) {
        std::cerr << "expected " << expected_rank << "D input for " << lm.op
                  << ", got rank " << x.shape().size() << std::endl;
        return 1;
    }

    std::unique_ptr<nflow::InferenceRuntime> rt;
    s = nflow::InferenceRuntime::Create(model_path, rt);
    if (!s.ok()) {
        std::cerr << "runtime create failed: " << s.message() << std::endl;
        return 1;
    }

    nflow::Tensor y = nflow::Tensor::Zeros(x.shape());
    // Replace the last dim with out_channels.
    std::vector<int64_t> y_shape = x.shape();
    if (!y_shape.empty()) y_shape.back() = out_ch;
    y = nflow::Tensor::Zeros(y_shape);
    s = rt->Run(x.data(), x.shape(), y.data(), y.shape());
    if (!s.ok()) {
        std::cerr << "inference failed: " << s.message() << std::endl;
        return 1;
    }
    s = nflow::npy::Write(output_path, y);
    if (!s.ok()) {
        std::cerr << "output write failed: " << s.message() << std::endl;
        return 1;
    }
    std::cout << "ok: wrote " << output_path << " op=" << lm.op << " shape=(";
    for (size_t i = 0; i < y_shape.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << y_shape[i];
    }
    std::cout << ")" << std::endl;
    return 0;
}
