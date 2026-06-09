// =============================================================================
// NeuroFlow C++ Runtime — FP8 E4M3 unit tests
// =============================================================================
//
// Compile and run via the existing cmake test target:
//   cmake -DNFLOW_BUILD_TESTS=ON .. && cmake --build . && ctest --output-on-failure
//
// Tests cover:
//   1. Bit-level round-trip for hand-picked values (0, ±1, ±448, denormals, NaN)
//   2. Round-to-nearest-even on the half-ULP boundary
//   3. Saturation behaviour: ±Inf -> ±448 (NOT NaN, since E4M3 has no Inf)
//   4. NaN propagation: FP32 NaN -> 0x7F (E4M3 NaN byte)
//   5. Bulk fake-quant matches per-element fake-quant
//   6. Round-trip identity: fake-quant(fake-quant(x)) == fake-quant(x)
//      (idempotence is the key sanity check for any quantise-dequantise pair)
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "neuroflow/fp8_e4m3.h"

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


// ---------------------------------------------------------------------------
// Test 1: bit-level round-trip for hand-picked values.
// ---------------------------------------------------------------------------
int TestBitRoundTrip() {
    // +0, -0
    EXPECT(nflow::fp32_to_e4m3_bits(0.0f) == 0x00);
    EXPECT(nflow::e4m3_bits_to_fp32(0x00) == 0.0f);
    {
        float neg_zero = nflow::e4m3_bits_to_fp32(0x80);
        EXPECT(std::signbit(neg_zero));
        EXPECT(neg_zero == 0.0f);
    }
    EXPECT(nflow::fp32_to_e4m3_bits(-0.0f) == 0x80);

    // ±1.0  (1.000_bin × 2^0 -> E=7 bias, mantissa 0)
    EXPECT(nflow::fp32_to_e4m3_bits(1.0f) == 0x38);   // 0_0111_000
    EXPECT(nflow::e4m3_bits_to_fp32(0x38) == 1.0f);
    EXPECT(nflow::fp32_to_e4m3_bits(-1.0f) == 0xB8);

    // ±448  (max finite: 1.110_bin × 2^8 -> E=15, mantissa 0b110)
    EXPECT(nflow::fp32_to_e4m3_bits(448.0f) == 0x7E);
    EXPECT(nflow::e4m3_bits_to_fp32(0x7E) == 448.0f);
    EXPECT(nflow::fp32_to_e4m3_bits(-448.0f) == 0xFE);

    // The byte 0x7E is the largest FINITE, NOT Inf (E4M3 has no Inf)
    EXPECT(std::isfinite(nflow::e4m3_bits_to_fp32(0x7E)));
    EXPECT(std::isfinite(nflow::e4m3_bits_to_fp32(0xFE)));

    // 0x7F (E=15, M=0b111) and 0xFF are the two NaN bytes
    EXPECT(std::isnan(nflow::e4m3_bits_to_fp32(0x7F)));
    EXPECT(std::isnan(nflow::e4m3_bits_to_fp32(0xFF)));

    // Smallest positive normal: 2^-6
    EXPECT(nflow::e4m3_bits_to_fp32(0x08) == nflow::kE4M3MinNormal);

    // Subnormal range: 0b000_?_??? (E=0).  Round-trip is exact.
    for (uint8_t b = 0x01; b < 0x08; ++b) {
        const float v = nflow::e4m3_bits_to_fp32(b);
        EXPECT(v > 0.0f);
        if (v >= nflow::kE4M3MinNormal) {
            std::fprintf(stderr, "subnormal byte 0x%02X: v=%g, expected < %g\n",
                         b, double(v), double(nflow::kE4M3MinNormal));
            return 1;
        }
        // Round-trip
        const uint8_t b2 = nflow::fp32_to_e4m3_bits(v);
        if (b2 != b) {
            std::fprintf(stderr, "subnormal byte 0x%02X: v=%g, rt=0x%02X\n",
                         b, double(v), b2);
            return 1;
        }
    }
    return 0;
}


// ---------------------------------------------------------------------------
// Test 2: round-to-nearest-even on the half-ULP boundary.
// ---------------------------------------------------------------------------
int TestRNE() {
    // Halfway between 1.0 (0x38) and 1.125 (1.001_bin = 0x39) is 1.0625.
    // The mantissa LSB of 0x38 is 0 (even), so 1.0625 must round DOWN to 1.0.
    EXPECT(nflow::fp32_to_e4m3_bits(1.0625f) == 0x38);

    // Halfway between 1.125 (0x39) and 1.25 (1.010_bin = 0x3A) is 1.1875.
    // The mantissa LSB of 0x39 is 1 (odd), so 1.1875 must round UP to 1.25.
    EXPECT(nflow::fp32_to_e4m3_bits(1.1875f) == 0x3A);

    // Sanity: values strictly closer to 1.125 than 1.0 round to 0x39.
    EXPECT(nflow::fp32_to_e4m3_bits(1.10f) == 0x39);
    return 0;
}


// ---------------------------------------------------------------------------
// Test 3: saturation: ±Inf -> ±448 (NOT NaN, since E4M3 has no Inf).
// ---------------------------------------------------------------------------
int TestSaturation() {
    const uint8_t pos_inf_bytes = nflow::fp32_to_e4m3_bits(
        std::numeric_limits<float>::infinity());
    EXPECT(pos_inf_bytes == 0x7E);  // +448

    const uint8_t neg_inf_bytes = nflow::fp32_to_e4m3_bits(
        -std::numeric_limits<float>::infinity());
    EXPECT(neg_inf_bytes == 0xFE);  // -448

    // Values above 448 also saturate to 0x7E / 0xFE.
    EXPECT(nflow::fp32_to_e4m3_bits(1000.0f) == 0x7E);
    EXPECT(nflow::fp32_to_e4m3_bits(-1000.0f) == 0xFE);
    return 0;
}


// ---------------------------------------------------------------------------
// Test 4: NaN propagation.
// ---------------------------------------------------------------------------
int TestNaN() {
    const uint8_t nan_byte = nflow::fp32_to_e4m3_bits(
        std::numeric_limits<float>::quiet_NaN());
    EXPECT(nan_byte == nflow::kE4M3NaNByte);  // 0x7F

    // FP32 signalling NaN -> E4M3 NaN
    const uint32_t snan_bits = 0x7F800001u;
    float snan = 0.0f;
    std::memcpy(&snan, &snan_bits, sizeof(snan));
    EXPECT(std::isnan(nflow::e4m3_bits_to_fp32(nflow::fp32_to_e4m3_bits(snan))));
    return 0;
}


// ---------------------------------------------------------------------------
// Test 5: bulk path matches per-element.
// ---------------------------------------------------------------------------
int TestBulkPath() {
    const size_t n = 1024;
    std::vector<float> in(n);
    std::vector<uint8_t> bulk(n);
    std::vector<uint8_t> scalar(n);
    for (size_t i = 0; i < n; ++i) {
        // Mix of small, large, denormal, zero
        const float v = static_cast<float>((i % 7) - 3) * 0.137f
                        + (i == 0 ? 0.0f : 100.0f / static_cast<float>(i));
        in[i] = v;
    }
    nflow::fp32_array_to_e4m3_bits(in.data(), bulk.data(), n);
    for (size_t i = 0; i < n; ++i) {
        scalar[i] = nflow::fp32_to_e4m3_bits(in[i]);
    }
    for (size_t i = 0; i < n; ++i) {
        if (bulk[i] != scalar[i]) {
            std::fprintf(stderr, "mismatch at i=%zu: bulk=0x%02X scalar=0x%02X\n",
                         i, bulk[i], scalar[i]);
            return 1;
        }
    }
    return 0;
}


// ---------------------------------------------------------------------------
// Test 6: fake-quant idempotence.  fake_quant(fake_quant(x)) == fake_quant(x).
// ---------------------------------------------------------------------------
int TestFakeQuantIdempotence() {
    const size_t n = 4096;
    std::vector<float> in(n);
    std::vector<float> out1(n);
    std::vector<float> out2(n);
    for (size_t i = 0; i < n; ++i) {
        // Spread across the full E4M3 range
        in[i] = static_cast<float>((i % 32) - 16) * 30.0f + (i & 1 ? 0.01f : -0.01f);
    }
    const float scale = 1.0f;
    nflow::fp8_e4m3_per_tensor_fake_quant(in.data(), scale, out1.data(), n);
    nflow::fp8_e4m3_per_tensor_fake_quant(out1.data(), scale, out2.data(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out1[i], out2[i], 1e-7);
    }
    return 0;
}


// ---------------------------------------------------------------------------
// Test 7: fake-quant noise floor: with scale=1 and inputs in [-1, 1],
// the absolute error must be bounded by 2^-4 = 0.0625 (the ULP of the
// mantissa around 1.0 in E4M3).  This is the headline test — it
// documents the FP8 quantisation noise floor.
// ---------------------------------------------------------------------------
int TestFakeQuantNoiseFloor() {
    const size_t n = 1024;
    std::vector<float> in(n);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        in[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(n - 1);
    }
    nflow::fp8_e4m3_per_tensor_fake_quant(in.data(), 1.0f, out.data(), n);
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float err = std::fabs(out[i] - in[i]);
        if (err > max_abs_err) max_abs_err = err;
    }
    // ULP around 1.0 in E4M3 is 1/8 = 0.125.  Max abs error is half of
    // that for round-to-nearest: 0.0625.  Allow a 1e-6 slack for the
    // bound itself.
    EXPECT(max_abs_err <= 0.0625f + 1e-6f);
    return 0;
}


int main() {
    int rc = 0;
    rc |= TestBitRoundTrip();
    if (rc) return rc;
    std::printf("Test 1 (bit round-trip) ......... OK\n");

    rc |= TestRNE();
    if (rc) return rc;
    std::printf("Test 2 (round-to-nearest-even) .. OK\n");

    rc |= TestSaturation();
    if (rc) return rc;
    std::printf("Test 3 (saturation) ............. OK\n");

    rc |= TestNaN();
    if (rc) return rc;
    std::printf("Test 4 (NaN propagation) ........ OK\n");

    rc |= TestBulkPath();
    if (rc) return rc;
    std::printf("Test 5 (bulk path) .............. OK\n");

    rc |= TestFakeQuantIdempotence();
    if (rc) return rc;
    std::printf("Test 6 (fake-quant idempotence) .. OK\n");

    rc |= TestFakeQuantNoiseFloor();
    if (rc) return rc;
    std::printf("Test 7 (noise floor) ............ OK\n");

    std::printf("\nAll FP8 E4M3 tests passed.\n");
    return 0;
}
