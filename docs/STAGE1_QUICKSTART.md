# NeuroFlow — Stage 1 Quick Start

This document walks you through the Stage 1 MVP end-to-end: train an FNO1d
in Python, export it to the NeuroFlow IR (JSON + binary), and run inference
in C++ via the pybind11 module.

> Time budget: 5-10 minutes from `git clone` to a working inference benchmark.

---

## 0. Prerequisites

| Tool    | Min version | Notes |
|---------|-------------|-------|
| Python  | 3.9+        | Anaconda or system Python |
| PyTorch | 2.0+        | `pip install torch` |
| CMake   | 3.18+       | For the C++ runtime |
| A C++20 compiler | gcc 11 / clang 14 / MSVC 19.30+ | Required for C++ runtime |
| pybind11 | 2.11+     | `pip install pybind11` |

> The C++ runtime has **zero external C++ dependencies** (no FFTW, no Eigen, no JSON lib).
> All FFT/JSON/NPY I/O is implemented in Stage 1 in-house.

---

## 1. Install the Python package

```bash
cd neuroflow
pip install -e .
```

(Use `pip install -e .[dev]` for tests/lint.)

Verify:

```bash
python -c "import neuroflow; print(neuroflow.__version__)"
# 0.1.0
```

---

## 2. Train an FNO1d on Burgers 1D

```bash
python examples/01_train_burgers1d.py --epochs 20 --n-train 200
```

This will produce, under `./artifacts/`:

| File | Purpose |
|---|---|
| `fno1d_burgers.pt`        | PyTorch checkpoint |
| `fno1d_burgers.neuroir`   | NeuroIR v0 (JSON) — human-readable, single source of truth |
| `fno1d_burgers.nneuroir`  | NeuroIR v0 (binary) — for C++ runtime, zero-deps parse |
| `fno1d_burgers_train.png` | Loss curve |
| `fno1d_burgers_pred.png`  | True vs predicted trajectory |

Expected terminal output (approx., 1-3 min on a laptop):

```
==> Building datasets
   train samples = ?, val samples = ?, t_in = 10, t_out = 20, n = 256
==> Building model
   parameters = ?
==> Training
[epoch   0] train=... val=... lr=1.00e-03
...
[epoch  19] train=... val=... lr=...
   elapsed = 80-150 s
==> Exporting NeuroIR v0 (JSON + binary)
   JSON: artifacts/fno1d_burgers.neuroir (~30-50 KB)
   BIN : artifacts/fno1d_burgers.nneuroir (~30-50 KB)
```

---

## 3. Build the C++ runtime

### 3a. Linux / macOS

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3b. Windows (MSVC)

```bat
cd cpp
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j
```

This produces:
- `build/nflow_infer` (or `build/Release/nflow_infer.exe`) — CLI tool
- `build/libnflow_core.a` (or `build/Release/nflow_core.lib`) — static library

Optional flags:
- `-DNFLOW_BUILD_PYBIND=ON` — also build the Python extension
- `-DNFLOW_BUILD_TESTS=ON`  — also build C++ unit tests

### 3c. Run C++ unit tests (optional)

```bash
cd build
ctest --output-on-failure
```

Expected: `All tests passed.` (covers Tensor, FFT roundtrip, IsPow2).

---

## 4. Wire the C++ runtime into Python (pybind11)

```bash
cd cpp
cmake -S . -B build -DNFLOW_BUILD_PYBIND=ON \
      -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build -j
```

This produces a Python extension module. To make it importable, copy or symlink
the built module into the project root (or set `PYTHONPATH`):

```bash
# Linux/macOS
cp build/neuroflow_cpp*.so ../neuroflow_cpp.so
# Windows
copy build\Release\neuroflow_cpp.pyd ..\neuroflow_cpp.pyd
```

---

## 5. Run the C++ inference + benchmark

```bash
cd ..  # back to project root
python examples/02_export_and_infer.py
```

Expected output:

```
==> Loading NeuroIR (JSON): artifacts/fno1d_burgers.neuroir
   op = FNO1d, version = 0.1.0
==> PyTorch reference inference
   output shape = (1, 256, 20)
==> Pure-Python IR inference (roundtrip)
   max abs diff vs PyTorch = ~1e-5
==> C++ runtime inference
   C++ output shape = (1, 256, 20)
   max abs diff vs PyTorch = ~1e-5
   C++ latency (batch=1, n=256) = ~5-30 ms
==> Pure-Python latency benchmark
   Python latency (batch=1, n=256) = ~50-200 ms
   C++ speedup vs Python = 5-20x
```

If the C++ extension is not built, the script gracefully skips the C++ portion
and only reports the Python baseline.

---

## 6. Use the CLI directly (no Python)

```bash
# Prepare a NumPy input (in Python)
python -c "import numpy as np; np.save('artifacts/x.npy', np.random.randn(1, 256, 10).astype('float32'))"

# Run inference
./cpp/build/nflow_infer --model artifacts/fno1d_burgers.nneuroir \
                       --input artifacts/x.npy \
                       --output artifacts/y.npy

# Inspect the output
python -c "import numpy as np; y = np.load('artifacts/y.npy'); print('shape', y.shape, 'mean', y.mean())"
```

---

## 7. Run Python tests

```bash
pytest tests/ -v
```

Expected: ~5 tests passing, including FNO forward, IR roundtrip, binary magic,
Burgers dataset shapes.

---

## What's NOT in Stage 1 (and where to look in the roadmap)

| Feature | Stage | Tracking |
|---|---|---|
| 2D / 3D FNO | 2 | ROADMAP §阶段 2 |
| DeepONet, Transolver, GNO | 2 | ROADMAP §阶段 2 |
| CUDA / ROCm / 苹果 Silicon | 4 | ROADMAP §阶段 4 |
| Hybrid Solve Graph (FEM + FNO) | 3 | ROADMAP §阶段 3 |
| OpenFOAM / ANSYS 插件 | 3 | ROADMAP §阶段 3 |
| Quantization (INT8 / FP8) | 2 | ROADMAP §阶段 2 |
| Distributed (MPI / NCCL) | 4 | ROADMAP §阶段 4 |

See `../ROADMAP.md` for the full plan.
