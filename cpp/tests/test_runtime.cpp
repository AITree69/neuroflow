// =============================================================================
// NeuroFlow C++ Runtime — unit tests (Stage 1)
// =============================================================================
//
// Compile and run via:
//   cmake -DNFLOW_BUILD_TESTS=ON .. && cmake --build . && ctest --output-on-failure
//
// Tests:
//   1. Tensor construction (Empty / Zeros / Full / Wrap / DebugString)
//   2. FFT roundtrip (Rfft -> Irfft recovers original signal)
//   3. ContiguousStrides correctness
// =============================================================================

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "neuroflow/fft.h"
#include "neuroflow/tensor.h"

#define EXPECT(cond)                                                                  \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "FAIL: %s @ %s:%d\n", #cond, __FILE__, __LINE__);     \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

#define EXPECT_NEAR(a, b, eps)                                                        \
    do {                                                                              \
        double _aa = static_cast<double>(a);                                          \
        double _bb = static_cast<double>(b);                                          \
        if (std::fabs(_aa - _bb) > (eps)) {                                           \
            std::fprintf(stderr, "FAIL: |%g - %g| > %g @ %s:%d\n", _aa, _bb,          \
                         double(eps), __FILE__, __LINE__);                            \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

int TestTensor() {
    auto t = nflow::Tensor::Zeros({2, 3, 4});
    EXPECT(t.shape().size() == 3);
    EXPECT(t.shape()[0] == 2);
    EXPECT(t.shape()[1] == 3);
    EXPECT(t.shape()[2] == 4);
    EXPECT(t.numel() == 24);
    EXPECT(t.bytes() == 24 * sizeof(float));
    for (int64_t i = 0; i < t.numel(); ++i) EXPECT(t.data()[i] == 0.0f);

    auto f = nflow::Tensor::Full({2, 2}, 3.5f);
    EXPECT(f.shape()[0] == 2);
    EXPECT(f.data()[0] == 3.5f);
    EXPECT(f.data()[3] == 3.5f);

    return 0;
}

int TestContiguousStrides() {
    auto s = nflow::ContiguousStrides({2, 3, 4});
    EXPECT(s.size() == 3);
    EXPECT(s[0] == 12);
    EXPECT(s[1] == 4);
    EXPECT(s[2] == 1);
    return 0;
}

int TestFftRoundtrip() {
    // Use a synthetic signal: sum of two sinusoids.
    const int n = 64;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(n);
        x[i] = std::sin(2.0f * 3.14159265f * 3.0f * t) + 0.5f * std::cos(2.0f * 3.14159265f * 5.0f * t);
    }
    std::vector<float> H((n / 2 + 1) * 2);
    nflow::fft::Rfft(x.data(), n, H.data());
    std::vector<float> y(n);
    nflow::fft::Irfft(H.data(), n, y.data());
    // Real FFT/IFFT has ~1e-5 roundtrip error.
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], y[i], 1e-4);
    }
    return 0;
}

int TestFftIsPow2() {
    EXPECT(nflow::fft::IsPow2(1));
    EXPECT(nflow::fft::IsPow2(2));
    EXPECT(nflow::fft::IsPow2(4));
    EXPECT(nflow::fft::IsPow2(64));
    EXPECT(!nflow::fft::IsPow2(3));
    EXPECT(!nflow::fft::IsPow2(0));
    EXPECT(!nflow::fft::IsPow2(63));
    EXPECT(nflow::fft::NextPow2(3) == 4);
    EXPECT(nflow::fft::NextPow2(64) == 64);
    EXPECT(nflow::fft::NextPow2(65) == 128);
    return 0;
}

int TestFft2dRoundtrip() {
    // 2D signal: a separable sum of two sinusoids along each axis.
    const int H = 16;
    const int W = 32;
    const int half_w = W / 2 + 1;
    std::vector<float> x(static_cast<size_t>(H) * W);
    for (int i = 0; i < H; ++i) {
        float yi = static_cast<float>(i) / static_cast<float>(H);
        for (int j = 0; j < W; ++j) {
            float xj = static_cast<float>(j) / static_cast<float>(W);
            x[static_cast<size_t>(i) * W + j] =
                std::sin(2.0f * 3.14159265f * 3.0f * yi) +
                0.5f * std::cos(2.0f * 3.14159265f * 5.0f * xj);
        }
    }
    std::vector<float> H_spec(static_cast<size_t>(H) * half_w * 2);
    nflow::fft::Rfft2(x.data(), H, W, H_spec.data());
    std::vector<float> y(static_cast<size_t>(H) * W);
    nflow::fft::Irfft2(H_spec.data(), H, W, y.data());
    for (int i = 0; i < H * W; ++i) {
        EXPECT_NEAR(x[i], y[i], 1e-4);
    }
    return 0;
}

int TestFft3dRoundtrip() {
    // 3D signal: separable sinusoids on each axis. Roundtrip should
    // recover x within float32 noise.
    const int H = 8;
    const int W = 8;
    const int D = 16;
    const int half_d = D / 2 + 1;
    std::vector<float> x(static_cast<size_t>(H) * W * D);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < D; ++k) {
                float yi = static_cast<float>(i) / static_cast<float>(H);
                float yj = static_cast<float>(j) / static_cast<float>(W);
                float yk = static_cast<float>(k) / static_cast<float>(D);
                x[(static_cast<size_t>(i) * W + j) * D + k] =
                    std::sin(2.0f * 3.14159265f * 2.0f * yi) +
                    0.5f * std::cos(2.0f * 3.14159265f * 3.0f * yj) +
                    0.25f * std::sin(2.0f * 3.14159265f * 4.0f * yk);
            }
        }
    }
    std::vector<float> H_spec(static_cast<size_t>(H) * W * half_d * 2);
    nflow::fft::Rfft3(x.data(), H, W, D, H_spec.data());
    std::vector<float> y(static_cast<size_t>(H) * W * D);
    nflow::fft::Irfft3(H_spec.data(), H, W, D, y.data());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x[i], y[i], 1e-4);
    }
    return 0;
}

int main() {
    int rc = 0;
    rc |= TestTensor();
    rc |= TestContiguousStrides();
    rc |= TestFftIsPow2();
    rc |= TestFftRoundtrip();
    rc |= TestFft2dRoundtrip();
    rc |= TestFft3dRoundtrip();
    if (rc == 0) {
        std::printf("All tests passed.\n");
    } else {
        std::printf("Some tests failed.\n");
    }
    return rc;
}
