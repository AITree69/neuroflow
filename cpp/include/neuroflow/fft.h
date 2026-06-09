// =============================================================================
// NeuroFlow C++ Runtime — minimal FFT (Stage 1)
// =============================================================================
//
// radix-2 Cooley-Tukey, in-place, real/complex forward, complex inverse.
// Asserts that input length is a power of two (Stage 1).
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>

namespace nflow::fft {

/// Real FFT. Length must be a power of two.
/// In: `x` of size `n` (real). Out: `out` of size n/2+1 (complex, interleaved).
void Rfft(const float* x, int n, float* out_complex);

/// Inverse real FFT.
/// In: `x_complex` of size n/2+1 (interleaved). Out: `out` of size n (real).
void Irfft(const float* x_complex, int n, float* out);

/// 2D real FFT (row-wise rfft + column-wise cfft). Both H and W must be
/// powers of two (Stage 1 limitation; matches the 1D path).
/// In: `x` of size H*W (real), row-major.
/// Out: `out_complex` of size H * (W/2+1) * 2 (complex, interleaved),
///       row-major. The DC mode is at (0, 0) and the Nyquist mode is
///       at (0, W/2).
void Rfft2(const float* x, int H, int W, float* out_complex);

/// Inverse 2D real FFT (column-wise inverse cfft + row-wise irfft).
/// In: `x_complex` of size H * (W/2+1) * 2 (interleaved, row-major).
/// Out: `out` of size H * W (real), row-major.
void Irfft2(const float* x_complex, int H, int W, float* out);

/// 3D real FFT (D-axis rfft + W-axis cfft + H-axis cfft). H, W, and D
/// must all be powers of two (Stage 2 Sprint 2 limitation; matches the
/// 1D / 2D paths).
/// In: `x` of size H*W*D (real), row-major.
/// Out: `out_complex` of size H * W * (D/2+1) * 2 (complex, interleaved),
///       row-major. The DC mode is at (0, 0, 0) and the Nyquist mode is
///       at (0, 0, D/2).
void Rfft3(const float* x, int H, int W, int D, float* out_complex);

/// Inverse 3D real FFT (H-axis inverse cfft + W-axis inverse cfft +
/// D-axis irfft).
/// In: `x_complex` of size H * W * (D/2+1) * 2 (interleaved, row-major).
/// Out: `out` of size H * W * D (real), row-major.
void Irfft3(const float* x_complex, int H, int W, int D, float* out);

/// True if `n` is a power of two.
bool IsPow2(int n);

/// Smallest power of two >= n.
int NextPow2(int n);

}  // namespace nflow::fft
