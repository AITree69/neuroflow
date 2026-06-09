// =============================================================================
// NeuroFlow C++ Runtime — minimal radix-2 Cooley-Tukey FFT
// =============================================================================
//
// Real and complex FFT/IFFT. In-place, single-threaded, float32.
// Length must be a power of two (asserted).
// =============================================================================

#include "neuroflow/fft.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

// MinGW (and strict-mode MSVC) does not expose M_PI from <cmath> by default.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nflow::fft {

bool IsPow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int NextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

namespace {

// Bit-reversal permutation (in-place). Length n must be power of two.
void BitReverseInPlace(std::complex<float>* a, int n) {
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

// In-place complex FFT (forward). Length n must be power of two.
void ComplexFftInPlace(std::complex<float>* a, int n, bool inverse) {
    BitReverseInPlace(a, n);
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = (inverse ? 2.0f : -2.0f) * static_cast<float>(M_PI) / static_cast<float>(len);
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int k = 0; k < len / 2; ++k) {
                std::complex<float> u = a[i + k];
                std::complex<float> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        const float inv = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) a[i] *= inv;
    }
}

}  // namespace

void Rfft(const float* x, int n, float* out_complex) {
    assert(IsPow2(n));
    std::vector<std::complex<float>> buf(n);
    for (int i = 0; i < n; ++i) buf[i] = std::complex<float>(x[i], 0.0f);
    ComplexFftInPlace(buf.data(), n, /*inverse=*/false);
    // out is (n/2+1) complex values, interleaved [re0, im0, re1, im1, ...]
    int half = n / 2 + 1;
    for (int i = 0; i < half; ++i) {
        out_complex[2 * i] = buf[i].real();
        out_complex[2 * i + 1] = buf[i].imag();
    }
}

void Irfft(const float* x_complex, int n, float* out) {
    assert(IsPow2(n));
    std::vector<std::complex<float>> buf(n);
    // x_complex is (n/2+1) interleaved. Mirror the upper half (conjugate).
    int half = n / 2 + 1;
    for (int i = 0; i < half; ++i) {
        buf[i] = std::complex<float>(x_complex[2 * i], x_complex[2 * i + 1]);
    }
    for (int i = 1; i < n - half + 1; ++i) {
        buf[n - i] = std::conj(buf[i]);
    }
    ComplexFftInPlace(buf.data(), n, /*inverse=*/true);
    for (int i = 0; i < n; ++i) out[i] = buf[i].real();
}

void Rfft2(const float* x, int H, int W, float* out_complex) {
    assert(IsPow2(H));
    assert(IsPow2(W));
    const int half_w = W / 2 + 1;
    // Step 1: row-wise Rfft. Scratch holds the post-row-FFT complex spectrum
    // in (H, half_w) interleaved layout.
    std::vector<std::complex<float>> scratch(static_cast<size_t>(H) * static_cast<size_t>(half_w));
    for (int i = 0; i < H; ++i) {
        Rfft(x + static_cast<int64_t>(i) * W, W,
             reinterpret_cast<float*>(scratch.data() + static_cast<int64_t>(i) * half_w));
    }
    // Step 2: column-wise complex FFT (forward) on each of the `half_w`
    // columns. The column length is H (a power of two, so
    // ComplexFftInPlace applies directly).
    std::vector<std::complex<float>> col(H);
    for (int j = 0; j < half_w; ++j) {
        for (int i = 0; i < H; ++i) col[i] = scratch[static_cast<int64_t>(i) * half_w + j];
        ComplexFftInPlace(col.data(), H, /*inverse=*/false);
        for (int i = 0; i < H; ++i) {
            out_complex[(static_cast<int64_t>(i) * half_w + j) * 2] = col[i].real();
            out_complex[(static_cast<int64_t>(i) * half_w + j) * 2 + 1] = col[i].imag();
        }
    }
}

void Irfft2(const float* x_complex, int H, int W, float* out) {
    assert(IsPow2(H));
    assert(IsPow2(W));
    const int half_w = W / 2 + 1;
    // Step 1: column-wise inverse complex FFT on each of the `half_w` columns.
    std::vector<std::complex<float>> col(H);
    std::vector<std::complex<float>> scratch(static_cast<size_t>(H) * static_cast<size_t>(half_w));
    for (int j = 0; j < half_w; ++j) {
        for (int i = 0; i < H; ++i) {
            col[i] = std::complex<float>(
                x_complex[(static_cast<int64_t>(i) * half_w + j) * 2],
                x_complex[(static_cast<int64_t>(i) * half_w + j) * 2 + 1]);
        }
        ComplexFftInPlace(col.data(), H, /*inverse=*/true);
        for (int i = 0; i < H; ++i) {
            scratch[static_cast<int64_t>(i) * half_w + j] = col[i];
        }
    }
    // Step 2: row-wise Irfft.
    for (int i = 0; i < H; ++i) {
        Irfft(reinterpret_cast<float*>(scratch.data() + static_cast<int64_t>(i) * half_w), W,
              out + static_cast<int64_t>(i) * W);
    }
}

void Rfft3(const float* x, int H, int W, int D, float* out_complex) {
    assert(IsPow2(H));
    assert(IsPow2(W));
    assert(IsPow2(D));
    const int half_d = D / 2 + 1;
    // Step 1: along the D axis, rfft each (h, w) line. The scratch holds
    // the post-D-rfft complex spectrum in (H, W, half_d) interleaved.
    std::vector<std::complex<float>> scratch(
        static_cast<size_t>(H) * W * half_d);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            Rfft(x + (static_cast<int64_t>(i) * W + j) * D, D,
                 reinterpret_cast<float*>(scratch.data()
                     + (static_cast<int64_t>(i) * W + j) * half_d));
        }
    }
    // Step 2: along the W axis, cfft each (h, half_d) column. The column
    // length is W (a power of two, so ComplexFftInPlace applies directly).
    std::vector<std::complex<float>> col_w(W);
    for (int i = 0; i < H; ++i) {
        for (int k = 0; k < half_d; ++k) {
            for (int j = 0; j < W; ++j) {
                col_w[j] = scratch[(static_cast<int64_t>(i) * W + j) * half_d + k];
            }
            ComplexFftInPlace(col_w.data(), W, /*inverse=*/false);
            for (int j = 0; j < W; ++j) {
                scratch[(static_cast<int64_t>(i) * W + j) * half_d + k] = col_w[j];
            }
        }
    }
    // Step 3: along the H axis, cfft each (w, half_d) column. The column
    // length is H (a power of two).
    std::vector<std::complex<float>> col_h(H);
    for (int j = 0; j < W; ++j) {
        for (int k = 0; k < half_d; ++k) {
            for (int i = 0; i < H; ++i) {
                col_h[i] = scratch[(static_cast<int64_t>(i) * W + j) * half_d + k];
            }
            ComplexFftInPlace(col_h.data(), H, /*inverse=*/false);
            for (int i = 0; i < H; ++i) {
                out_complex[((static_cast<int64_t>(i) * W + j) * half_d + k) * 2] =
                    col_h[i].real();
                out_complex[((static_cast<int64_t>(i) * W + j) * half_d + k) * 2 + 1] =
                    col_h[i].imag();
            }
        }
    }
}

void Irfft3(const float* x_complex, int H, int W, int D, float* out) {
    assert(IsPow2(H));
    assert(IsPow2(W));
    assert(IsPow2(D));
    const int half_d = D / 2 + 1;
    // Step 1: along the H axis, inverse cfft each (w, half_d) column.
    std::vector<std::complex<float>> col_h(H);
    std::vector<std::complex<float>> scratch(
        static_cast<size_t>(H) * W * half_d);
    for (int j = 0; j < W; ++j) {
        for (int k = 0; k < half_d; ++k) {
            for (int i = 0; i < H; ++i) {
                col_h[i] = std::complex<float>(
                    x_complex[((static_cast<int64_t>(i) * W + j) * half_d + k) * 2],
                    x_complex[((static_cast<int64_t>(i) * W + j) * half_d + k) * 2 + 1]);
            }
            ComplexFftInPlace(col_h.data(), H, /*inverse=*/true);
            for (int i = 0; i < H; ++i) {
                scratch[(static_cast<int64_t>(i) * W + j) * half_d + k] = col_h[i];
            }
        }
    }
    // Step 2: along the W axis, inverse cfft each (h, half_d) column.
    std::vector<std::complex<float>> col_w(W);
    for (int i = 0; i < H; ++i) {
        for (int k = 0; k < half_d; ++k) {
            for (int j = 0; j < W; ++j) {
                col_w[j] = scratch[(static_cast<int64_t>(i) * W + j) * half_d + k];
            }
            ComplexFftInPlace(col_w.data(), W, /*inverse=*/true);
            for (int j = 0; j < W; ++j) {
                scratch[(static_cast<int64_t>(i) * W + j) * half_d + k] = col_w[j];
            }
        }
    }
    // Step 3: along the D axis, irfft each (h, w) line.
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            Irfft(reinterpret_cast<float*>(scratch.data()
                       + (static_cast<int64_t>(i) * W + j) * half_d),
                  D, out + (static_cast<int64_t>(i) * W + j) * D);
        }
    }
}

}  // namespace nflow::fft
