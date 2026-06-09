// =============================================================================
// NeuroFlow Python <-> C++ bit-level FP8 E4M3 parity shim (header only)
// =============================================================================
//
// This is a thin header that exposes the C++ bit-level FP8 E4M3
// routines through pybind11.  The C++ implementation in
// `cpp/include/neuroflow/fp8_e4m3.h` is the single source of truth
// for FP8 E4M3 quantise / dequantise; the Python side uses
// `bitround_e4m3` (see `neuroflow/quant/fp8_e4m3.py`) which mirrors
// the same bit-level code path for tests.  This file makes the C++
// implementation available to Python test code so that we can
// verify cross-language parity without going through the IR.
// =============================================================================

#pragma once

#ifdef NFLOW_HAS_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "neuroflow/fp8_e4m3.h"

namespace py = pybind11;

inline void RegisterFp8E4M3Bindings(py::module_& m) {
    m.def("fp8_e4m3_fake_quant",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> in,
           float scale) -> py::array_t<float> {
            auto in_unchecked = in.unchecked<1>();
            py::array_t<float> out(in_unchecked.shape(0));
            auto out_mut = out.mutable_unchecked<1>();
            nflow::fp8_e4m3_per_tensor_fake_quant(
                in_unchecked.data(0), scale,
                out_mut.mutable_data(0),
                static_cast<size_t>(in_unchecked.shape(0)));
            return out;
        },
        py::arg("in"), py::arg("scale"),
        "Per-tensor FP8 E4M3 fake-quant (in: float32, out: float32).");

    m.def("fp8_e4m3_to_bits",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> in)
            -> py::array_t<uint8_t> {
            auto in_unchecked = in.unchecked<1>();
            py::array_t<uint8_t> out(in_unchecked.shape(0));
            auto out_mut = out.mutable_unchecked<1>();
            nflow::fp32_array_to_e4m3_bits(
                in_unchecked.data(0), out_mut.mutable_data(0),
                static_cast<size_t>(in_unchecked.shape(0)));
            return out;
        },
        py::arg("in"),
        "Convert FP32 -> FP8 E4M3 byte (IEEE-754 binary8 RNE).");

    m.def("fp8_e4m3_from_bits",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> in)
            -> py::array_t<float> {
            auto in_unchecked = in.unchecked<1>();
            py::array_t<float> out(in_unchecked.shape(0));
            auto out_mut = out.mutable_unchecked<1>();
            nflow::e4m3_bits_to_fp32_array(
                in_unchecked.data(0), out_mut.mutable_data(0),
                static_cast<size_t>(in_unchecked.shape(0)));
            return out;
        },
        py::arg("in"),
        "Convert FP8 E4M3 byte -> FP32.");
}

// Make a forwarder symbol in `nflow` namespace so callers can use
// `nflow::RegisterFp8E4M3Bindings` cleanly.  We have to do this in
// the global namespace because `inline` functions in headers don't
// take kindly to being referenced as `nflow::Name` from another
// translation unit (the inline-ness keeps the function in *every*
// TU, but the symbol resolution is still per-TU).
#ifdef NFLOW_PYBIND_FORWARD_FP8
namespace nflow {
    inline void RegisterFp8E4M3Bindings(py::module_& m) {
        ::RegisterFp8E4M3Bindings(m);
    }
}
#endif

#endif  // NFLOW_HAS_PYBIND11
