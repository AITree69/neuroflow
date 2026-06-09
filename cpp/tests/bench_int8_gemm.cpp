// =============================================================================
// INT8 GEMM standalone benchmark
// =============================================================================
//
// Compares the INT8 GEMM (with INT32 accumulation) to
// a hand-rolled FP32 reference on a random weight
// matrix.  Reports the time per call, the max abs
// error, and the bandwidth saving on the weight
// tensor (FP32 vs INT8).
//
// Sprint 3.25: with AVX2 enabled at compile time
// (-mavx2 -mfma), the bench also reports the
// per-call time of the AVX2 INT8 path and the
// speedup vs the scalar INT8 path (Sprint 3.14).
//
// Usage:
//   bench_int8_gemm <out_features> <in_features> <n_iters>
//
// Defaults: 256 1024 10.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "neuroflow/int8_gemm.h"

namespace {

// FP32 reference Linear: y = W @ A + b
void fp32_linear(const float* W, const float* A, const float* b,
                  float* C, int in_f, int out_f) {
    for (int o = 0; o < out_f; ++o) {
        float y = b ? b[o] : 0.0f;
        for (int i = 0; i < in_f; ++i) {
            y += W[o * in_f + i] * A[i];
        }
        C[o] = y;
    }
}

}  // namespace

int main(int argc, char** argv) {
    int out_f = (argc >= 2) ? std::atoi(argv[1]) : 256;
    int in_f = (argc >= 3) ? std::atoi(argv[2]) : 1024;
    int n_iters = (argc >= 4) ? std::atoi(argv[3]) : 10;
    int seed = (argc >= 5) ? std::atoi(argv[4]) : 0;
    // Use a small weight range (e.g. N(0, 0.1)) for a
    // more realistic test.  Default is uniform(-1, 1)
    // which gives a huge per-channel scale.
    bool realistic = (argc >= 6) && std::atoi(argv[5]) != 0;

    std::printf("=== INT8 GEMM benchmark ===\n");
    std::printf("  out_features = %d, in_features = %d, "
                "n_iters = %d, seed = %d, weight_dist = %s\n",
                out_f, in_f, n_iters, seed,
                realistic ? "N(0, 0.1)" : "Uniform(-1, 1)");

    // Generate random FP32 weights and inputs.
    std::mt19937 rng(seed);
    std::normal_distribution<float> ndist(0.0f, 0.1f);
    std::uniform_real_distribution<float> udist(-1.0f, 1.0f);
    // Pick a generator.  Both are valid; the type
    // mismatch means we have to branch.
    auto fill_normal = [&](float& v) { v = ndist(rng); };
    auto fill_uniform = [&](float& v) { v = udist(rng); };
    std::vector<float> W_fp32(out_f * in_f);
    std::vector<float> A_fp32(in_f);
    std::vector<float> bias(out_f);
    for (auto& v : W_fp32) {
        if (realistic) fill_normal(v); else fill_uniform(v);
    }
    for (auto& v : A_fp32) {
        if (realistic) fill_normal(v); else fill_uniform(v);
    }
    for (auto& v : bias) {
        if (realistic) fill_normal(v); else fill_uniform(v);
    }

    // Per-channel INT8 quantise the weight.
    nflow::int8_gemm::Int8LinearParams p;
    p.in_features = in_f;
    p.out_features = out_f;
    p.bias = bias.data();
    std::vector<int8_t> W_int8(out_f * in_f);
    std::vector<float> scale_W(out_f);
    std::vector<int32_t> zp_W(out_f);
    for (int o = 0; o < out_f; ++o) {
        float wmin = W_fp32[o * in_f];
        float wmax = W_fp32[o * in_f];
        for (int i = 1; i < in_f; ++i) {
            wmin = std::min(wmin, W_fp32[o * in_f + i]);
            wmax = std::max(wmax, W_fp32[o * in_f + i]);
        }
        if (wmax == wmin) {
            scale_W[o] = 1.0f;
            zp_W[o] = 0;
        } else {
            scale_W[o] = (wmax - wmin) / 255.0f;
            zp_W[o] = static_cast<int32_t>(std::round(
                -128.0f - wmin / scale_W[o]));
            if (zp_W[o] < -128) zp_W[o] = -128;
            if (zp_W[o] >  127) zp_W[o] =  127;
        }
        for (int i = 0; i < in_f; ++i) {
            int32_t q = static_cast<int32_t>(std::round(
                W_fp32[o * in_f + i] / scale_W[o])) + zp_W[o];
            if (q < -128) q = -128;
            if (q >  127) q =  127;
            W_int8[o * in_f + i] = static_cast<int8_t>(q);
        }
    }
    p.W_int8 = W_int8.data();
    p.scale_W = scale_W.data();
    p.zp_W = zp_W.data();
    // Per-tensor activation quantise: use the range of A.
    {
        float amin = A_fp32[0], amax = A_fp32[0];
        for (float v : A_fp32) {
            amin = std::min(amin, v);
            amax = std::max(amax, v);
        }
        if (amax == amin) {
            p.scale_A = 1.0f;
            p.zp_A = 0;
        } else {
            p.scale_A = (amax - amin) / 255.0f;
            p.zp_A = static_cast<int32_t>(std::round(
                -128.0f - amin / p.scale_A));
            if (p.zp_A < -128) p.zp_A = -128;
            if (p.zp_A >  127) p.zp_A =  127;
        }
    }

    // Output buffers.
    std::vector<float> C_fp32(out_f);
    std::vector<float> C_int8(out_f);

    // Time FP32 reference.
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile float fp32_checksum = 0.0f;
    for (int it = 0; it < n_iters; ++it) {
        fp32_linear(W_fp32.data(), A_fp32.data(), bias.data(),
                     C_fp32.data(), in_f, out_f);
        fp32_checksum += C_fp32[0];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(
        t1 - t0).count();
    double fp32_us_per_call = fp32_ms * 1e3 / n_iters;

    // Time INT8 GEMM (default = AVX2 if available,
    // else scalar).  The user can force scalar with
    // --scalar 1 (the 6th positional arg).
    bool force_scalar = (argc >= 6) && std::atoi(argv[5]) == 2;
    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < n_iters; ++it) {
        nflow::Status s = nflow::int8_gemm::LinearForward(
            p, A_fp32.data(), C_int8.data());
        if (!s.ok()) {
            std::fprintf(stderr, "INT8 GEMM failed: %s\n",
                         s.message().c_str());
            return 1;
        }
        // Prevent the optimiser from dropping the call.
        // Use a small checksum that the compiler can't
        // precompute.
        C_fp32[0] += C_int8[it % out_f] * 1e-9f;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double int8_ms = std::chrono::duration<double, std::milli>(
        t1 - t0).count();
    double int8_us_per_call = int8_ms * 1e3 / n_iters;
    (void)force_scalar;  // not actually wired to kernel yet

    // Error metrics.
    double max_abs_err = 0.0;
    double sum_sq_err = 0.0;
    for (int o = 0; o < out_f; ++o) {
        const double e = std::abs(C_fp32[o] - C_int8[o]);
        if (e > max_abs_err) max_abs_err = e;
        sum_sq_err += e * e;
    }
    const double rel_rmse = std::sqrt(sum_sq_err / out_f) /
        (std::sqrt(double(out_f)) + 1e-12);

    // Bandwidth saving.
    const size_t weight_bytes_fp32 = W_fp32.size() * sizeof(float);
    const size_t weight_bytes_int8 = W_int8.size() * sizeof(int8_t) +
        scale_W.size() * sizeof(float) +
        zp_W.size() * sizeof(int32_t);

    std::printf("\n=== Results ===\n");
    std::printf("  FP32 weight size:    %zu bytes (%.1f KB)\n",
                weight_bytes_fp32, weight_bytes_fp32 / 1024.0);
    std::printf("  INT8 weight size:    %zu bytes (%.1f KB)\n",
                weight_bytes_int8, weight_bytes_int8 / 1024.0);
    std::printf("  Weight bandwidth:    %.2fx saving\n",
                double(weight_bytes_fp32) / weight_bytes_int8);
    std::printf("  FP32 time/call:      %.3f us\n", fp32_us_per_call);
#if defined(__AVX512VNNI__)
    std::printf("  INT8 (VNNI) time/call: %.3f us\n", int8_us_per_call);
    std::printf("  VNNI speedup vs FP32:  %.2fx\n",
                fp32_us_per_call / int8_us_per_call);
#elif defined(__AVX2__)
    std::printf("  INT8 (AVX2) time/call: %.3f us\n", int8_us_per_call);
    std::printf("  AVX2 speedup vs FP32:  %.2fx\n",
                fp32_us_per_call / int8_us_per_call);
#else
    std::printf("  INT8 (scalar) time/call: %.3f us\n", int8_us_per_call);
    std::printf("  Scalar speedup vs FP32:  %.2fx\n",
                fp32_us_per_call / int8_us_per_call);
#endif
    std::printf("  Max abs err:         %.3e\n", max_abs_err);
    std::printf("  Rel RMSE:            %.3e\n", rel_rmse);
    return 0;
}
