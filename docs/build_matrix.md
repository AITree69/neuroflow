# Build Matrix

The C++ runtime is zero-dependency (standard library only). The pybind11
binding needs a C++17 compiler and a Python 3.10+ development headers.

| Platform | Compiler | CMake | Python | pybind11 | Status |
|---|---|---|---|---|---|
| Windows 11 / 10 | MinGW g++ 13.1 (CLion-bundled) | 4.2.2 | 3.12 | 2.11+ | ✅ verified |
| Ubuntu 22.04 | g++ 11 / 12 | ≥ 3.20 | 3.10 / 3.11 / 3.12 | 2.11+ | ✅ CI (no pybind yet) |
| macOS 13+ | Apple Clang 14 | ≥ 3.20 | 3.10 / 3.11 / 3.12 | 2.11+ | ✅ CI (no pybind yet) |

## Windows + MinGW (verified locally)

```powershell
$env:PATH = "C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin;$env:PATH"
cmake -S cpp -B cpp/build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target nflow_core test_runtime
./cpp/build/test_runtime

# pybind11 binding
python -m pip install pybind11
cmake -S cpp -B cpp/build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DNFLOW_BUILD_PYBIND=ON
cmake --build cpp/build --target neuroflow_cpp
# Copy the .pyd and the three MinGW DLLs next to it, or prepend MinGW bin to PATH.
```

## Ubuntu 22.04

```bash
sudo apt-get install -y cmake g++ python3-dev
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target nflow_core test_runtime
./cpp/build/test_runtime
```

## macOS 13+

```bash
brew install cmake
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target nflow_core test_runtime
./cpp/build/test_runtime
```

## Eigen backend (Stage 2 Sprint 2)

`nflow_core` uses [Eigen 3.4.0](https://gitlab.com/libeigen/eigen/-/releases)
for the `Linear` GEMM (and the spectral-convolution frequency-domain
multiplies when the path is enabled). Eigen is **header-only**, so no
link step is required — only an include path.

The Eigen tree is vendored at
`cpp/third_party/eigen-3.4.0/` (about 3.5 MB of headers). To refresh
the vendored copy:

```bash
# from the repo root
mkdir -p cpp/third_party
curl -L -o cpp/third_party/eigen-3.4.0.zip \
    https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip -q cpp/third_party/eigen-3.4.0.zip -d cpp/third_party/
rm cpp/third_party/eigen-3.4.0.zip
```

CMake picks Eigen up automatically when
`cpp/third_party/eigen-3.4.0/Eigen/Core` exists. To opt out (e.g. for
debugging the hand-rolled loop or for an exotic target without
Eigen SIMD detection), pass `-DNFLOW_USE_EIGEN=OFF`:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DNFLOW_USE_EIGEN=OFF
```

On MinGW / GCC / Clang the build is also pinned to `-O3` so the
non-Matmul loops in `fno.cpp` (the FFT inner loops, the spectral
window copies, the per-channel multiplies) get auto-vectorized.


## Known cross-platform gotchas

- **`#ifndef M_PI` guard.** MinGW's `<cmath>` does not define `M_PI` by
  default; guard the include in any `.cpp` that uses it.
- **MinGW pybind11 runtime DLLs.** A MinGW-built `.pyd` depends on
  `libgcc_s_seh-1.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll`. Either
  copy them next to the `.pyd`, or prepend the MinGW `bin/` directory
  to `PATH` before launching Python.
- **CMake minimum.** 3.20 is required for `target_link_libraries` with
  generator expressions on Windows + MinGW. Older versions fail silently.
- **Apple Clang and `<numbers>`.** macOS's libstdc++ is too old for
  `std::numbers::pi`; use `M_PI` and the same `#ifndef` guard.
