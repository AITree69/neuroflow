# =============================================================================
# NeuroFlow Stage 1 C++ Runtime
# =============================================================================
# Builds:
#   - nflow_infer       : command-line inference tool
#   - neuroflow_cpp     : Python extension (pybind11) — pip-installable wheel
#   - test_runtime      : unit tests (optional)
#
# Tested on: Linux (gcc 11), Windows (MSVC 19.30+), macOS (clang 14+)
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(neuroflow_cpp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NFLOW_BUILD_CLI "Build command-line nflow_infer" ON)
option(NFLOW_BUILD_PYBIND "Build Python bindings (pybind11)" OFF)
option(NFLOW_BUILD_TESTS "Build C++ unit tests" OFF)
option(NFLOW_BUILD_LAMMPS_SHIM "Build the LAMMPS `fix nflow` shim + standalone harness" OFF)
option(NFLOW_BUILD_INT8_GEMM_BENCH "Build the INT8 GEMM benchmark" OFF)
option(NFLOW_USE_EIGEN "Use Eigen for the Linear and spectral-convolution GEMM" ON)

# ----------------------------------------------------------------------------
# Core library: nflow_core
# ----------------------------------------------------------------------------
set(NFLOW_CORE_SOURCES
  src/tensor.cpp
  src/fft.cpp
  src/fno.cpp
  src/runtime.cpp
  src/npy_io.cpp
  src/int8_gemm.cpp
)
add_library(nflow_core STATIC ${NFLOW_CORE_SOURCES})
target_include_directories(nflow_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(nflow_core PUBLIC cxx_std_20)

# Optional Eigen backend. Eigen 3.4.0 is vendored under
# `third_party/eigen-3.4.0/` (header-only, ~3.5 MB of headers).
# Set NFLOW_USE_EIGEN=OFF to fall back to the hand-rolled loops.
if(NFLOW_USE_EIGEN)
  set(_EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0")
  if(NOT EXISTS "${_EIGEN_ROOT}/Eigen/Core")
    message(FATAL_ERROR
      "NFLOW_USE_EIGEN=ON but Eigen 3.4.0 not found at ${_EIGEN_ROOT}. "
      "Either vendor Eigen under cpp/third_party/eigen-3.4.0/ or set "
      "NFLOW_USE_EIGEN=OFF.")
  endif()
  target_include_directories(nflow_core PUBLIC
    $<BUILD_INTERFACE:${_EIGEN_ROOT}>
  )
  target_compile_definitions(nflow_core PUBLIC NFLOW_HAS_EIGEN=1)
endif()

if(MSVC)
  target_compile_options(nflow_core PRIVATE /W4 /permissive-)
  target_compile_definitions(nflow_core PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(nflow_core PRIVATE -Wall -Wextra -Wpedantic -O3)
  # Sprint 3.25: enable AVX2 INT8 dot product in
  # int8_gemm.cpp.  We use -mavx2 (not -march=native)
  # so the binary runs on the widest set of x86_64
  # CPUs that support AVX2 (Intel Haswell 2013+,
  # AMD Zen 1 2017+).  For VNNI (Intel Cascade Lake+,
  # AMD Zen4+), use -mavx512vnni -mavx512f
  # -mavx512bw -mavx512dq.  We use the manual
  # flag list (not -march=native) so the binary
  # is portable across VNNI-capable CPUs.
  target_compile_options(nflow_core PRIVATE -mavx2 -mfma -mavx512vnni -mavx512f -mavx512bw -mavx512dq)
endif()

# ----------------------------------------------------------------------------
# CLI binary: nflow_infer
# ----------------------------------------------------------------------------
if(NFLOW_BUILD_CLI)
  add_executable(nflow_infer src/main.cpp)
  target_link_libraries(nflow_infer PRIVATE nflow_core)
endif()

# ----------------------------------------------------------------------------
# Python bindings
# ----------------------------------------------------------------------------
if(NFLOW_BUILD_PYBIND)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(neuroflow_cpp bindings/pybind_module.cpp)
  target_link_libraries(neuroflow_cpp PRIVATE nflow_core)
  if(MSVC)
    target_compile_options(neuroflow_cpp PRIVATE /W4)
  endif()
endif()

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(NFLOW_BUILD_TESTS)
  enable_testing()
  add_executable(test_runtime tests/test_runtime.cpp)
  target_link_libraries(test_runtime PRIVATE nflow_core)
  add_test(NAME test_runtime COMMAND test_runtime)
endif()

# ----------------------------------------------------------------------------
# LAMMPS `fix nflow` shim — Stage 3 Sprint 3.13
# ----------------------------------------------------------------------------
# Implements the LAMMPS `fix` interface contract via a
# LAMMPS-agnostic FixNflow class.  The standalone harness
# `test_fix_nflow_standalone` mimics what LAMMPS would do
# so the shim can be tested without LAMMPS installed.
# To plug into LAMMPS, drop `src/lammps_shim/fix_nflow.cpp`
# into LAMMPS's EXTRA-FIX and write a ~50-line
# `LAMMPS_NS::FixNflow` that delegates to `FixNflow`.
if(NFLOW_BUILD_LAMMPS_SHIM)
  add_library(nflow_lammps_shim STATIC
    src/lammps_shim/fix_nflow.cpp
  )
  target_include_directories(nflow_lammps_shim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(nflow_lammps_shim PUBLIC nflow_core)
  add_executable(test_fix_nflow_standalone
    tests/test_fix_nflow_standalone.cpp
  )
  target_link_libraries(test_fix_nflow_standalone PRIVATE
    nflow_lammps_shim nflow_core)
  if(MSVC)
    target_compile_options(test_fix_nflow_standalone PRIVATE /W4)
  endif()
endif()

# ----------------------------------------------------------------------------
# INT8 GEMM benchmark — Stage 3 Sprint 3.14
# ----------------------------------------------------------------------------
if(NFLOW_BUILD_INT8_GEMM_BENCH)
  add_executable(bench_int8_gemm tests/bench_int8_gemm.cpp)
  target_link_libraries(bench_int8_gemm PRIVATE nflow_core)
  if(MSVC)
    target_compile_options(bench_int8_gemm PRIVATE /W4)
  else()
    # Sprint 3.25: enable AVX2 in the bench too.
    # Sprint 3.26: also enable VNNI.
    target_compile_options(bench_int8_gemm PRIVATE -O3 -mavx2 -mfma -mavx512vnni -mavx512f -mavx512bw -mavx512dq)
  endif()
endif()
