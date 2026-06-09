// =============================================================================
// NeuroFlow C++ Runtime — minimal NPY I/O
// =============================================================================
//
// Read/write NumPy .npy files (single array, float32, C-contiguous).
// This is enough to interoperate with numpy.save / numpy.load.
// =============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "neuroflow/tensor.h"

namespace nflow::npy {

/// Read a .npy file (float32, C-contig) into a Tensor.
Status Read(const std::string& path, Tensor& out);

/// Write a Tensor to a .npy file.
Status Write(const std::string& path, const Tensor& t);

}  // namespace nflow::npy
