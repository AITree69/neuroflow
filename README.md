# NeuroFlow

An open framework for **neural-operator-based PDE solving**.  A
Python research layer (PyTorch) trains operator models on standard
scientific data; a C++ production runtime loads the trained model
from a portable Intermediate Representation (`NeuroIR`) with **zero
third-party dependencies**, exposing inference through both a CLI
(`nflow_infer`) and a pybind11 module (`neuroflow_cpp`).  The same
trained model ships unchanged from a researcher's notebook to a
production C++ host.

The framework currently supports **eight operator families**
(FNO1d, FNO2d, FNO3d, DeepONet, TokenMixer, GraphOp, and their 2D
analogs), **per-tensor / per-channel / per-token INT8 quantisation,
FP8 (E4M3) quantisation, and QAT**, plus an AVX2/AVX-512-VNNI
accelerated INT8 GEMM, and a first domain SDK for LAMMPS.

---

## 1. When to use NeuroFlow

| You want to... | NeuroFlow is for you if... | If not, try... |
|---|---|---|
| Train an FNO / DeepONet / GNN surrogate for a PDE | Yes — 8 operator families, all in `neuroflow.nn` | `neuraloperator` (research only) |
| Deploy a trained model inside a C++ host | Yes — single static lib, zero deps, ONNX-style IR | ONNX Runtime (heavier, less operator-specific) |
| Mix a neural surrogate with a classical solver (FEM/FVM) | Yes — designed for hybrid compute (Stage 3) | A hand-written C++ plugin per host |
| Quantise an operator for edge / embedded deployment | Yes — INT8/FP8 with C++ runtime path | TFLite (more general, but no operator support) |
| Study new neural-operator architectures | Partial — adding a 9th op = 1 schema + 1 `model_class` rebind | `neuraloperator` (more research-focused) |
| Image / video / text tasks | **No** — wrong domain | PyTorch + TIMM / HF |

The current sweet spot is **PDE surrogates inside scientific
computing pipelines** (chip thermal, CFD, MD surrogate, weather
downscaling, structural mechanics).

---

## 2. Architecture

```
                        ┌──────────────────────────┐
                        │   Python research layer   │
                        │   (PyTorch + NumPy)       │
                        │                           │
                        │   neuroflow.nn.{FNO1d,    │
                        │   FNO2d, FNO3d, DeepONet,│
                        │   TokenMixer, GraphOp,…}  │
                        │                           │
                        │   neuroflow.quant.{       │
                        │   static, qat, fake_quant}│
                        └────────────┬──────────────┘
                                     │  export_to_binary()
                                     ▼
                        ┌──────────────────────────┐
                        │   NeuroIR (v0.18)        │
                        │   JSON + .nneuroir       │
                        │   8 op-codes + NIRQ      │
                        └────────────┬──────────────┘
                                     │  InferenceRuntime::Create
                                     ▼
                        ┌──────────────────────────┐
                        │   C++ production runtime │
                        │   (zero external deps)   │
                        │                           │
                        │   nflow-tensor  Eigen     │
                        │   nflow-fft     radix-2   │
                        │   nflow-quant   NIRQ      │
                        │   nflow-ir      parser    │
                        │   nflow-lammps  fix shim  │
                        └──────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        nflow_infer.exe     neuroflow_cpp.pyd       libnflow_core.a
        (file→file CLI)     (pybind11)               (LAMMPS fix_nflow)
```

Key design decisions, briefly:

- **One IR, two encodings.** JSON for human inspection
  (`.neuroir`); binary (`.nneuroir`) for zero-dep C++ parsing.
  The Python exporter is the single source of truth — both
  encodings come from the same `NeuroIRSpec` dataclass tree.
- **A single IR version field, op-codes for everything.**
  Adding a 9th op family = add a row to the `_OP_SCHEMAS`
  registry, not a breaking format change.
- **C++ runtime is independent of Python.** A model that exists
  only as a `.nneuroir` file plus the C++ binary is enough for
  deployment; no Python interpreter, no `libtorch`, no BLAS, no
  FFTW on the target.

## 3. NeuroIR v0 binary layout

`.nneuroir` is a single-file, little-endian, versioned format:

| Offset | Size | Field |
|---|---|---|
| 0  | 4  | Magic = `"NIR0"` |
| 4  | 2  | Version (uint16, = 1) |
| 6  | 1  | Op code (uint8) |
| 7  | 1  | Reserved |
| 8  | 28 | Config: 7 × int32 = `[in_ch, out_ch, width, modes, n_layers, pad_factor, _]` |
| 36 | 1  | Activation (0=gelu, 1=relu) |
| 37 | 3  | Reserved |
| 40 | 4  | `n_weights` (uint32) |
| 44 |  | per-weight records (repeating) |

Per-weight record:

| Size | Field |
|---|---|
| 1 byte | name_len |
| name_len bytes | utf-8 name |
| 1 byte | ndim |
| ndim × 4 bytes | int32 dims |
| numel × 4 bytes | float32 little-endian data |

**Op codes** (v0.18.0):

| Code | Op | Layout |
|---|---|---|
| 0x01 | FNO1d | `specs.{i}.weights_{real,imag}: (in, out, modes)`; `locs.{i}.{weight,bias}: (out, in) / (out,)` |
| 0x02 | FNO2d | + `modes_h, modes_w` instead of `modes` |
| 0x03 | FNO3d | + `modes_h, modes_w, modes_d` |
| 0x04 | DeepONet | `branch.{w,b}` + `trunk.{w,b}` + `head.{w,b}` |
| 0x05 | TokenMixer | per-block `qkv.{w,b}` + `proj.{w,b}` + `ln1.{g,b}` + `ln2.{g,b}` |
| 0x06 | GraphOp | `lift.{w,b}` + `W_self` + `W_neigh` + `head.{w,b}` + CSR `adj_offsets/adj_indices/deg_inv` |
| 0x07 | TokenMixer2D | flatten-spatial TokenMixer + same weight names |
| 0x08 | GraphOp2D | 8-conn grid GraphOp + same weight names |

**Quantisation block (NIRQ)**: an **optional** trailing block
appended after the last weight record. Presence is signalled by a
weight whose name equals the sentinel `__nirq__`; the actual
quantisation metadata follows in a structured form
(per-tensor / per-channel / per-token / FP8 / per-mode).  Older
C++ runtimes that don't know NIRQ simply stop reading at the
sentinel weight, so the format stays forward-compatible.

## 4. Python API

### 4.1 Train a model

```python
import torch
from neuroflow.nn.fno import FNO1d, FNO1dConfig
from neuroflow.data.burgers import Burgers1dDataset
from neuroflow.train import Trainer, TrainConfig

model = FNO1d(FNO1dConfig(
    in_channels=10, out_channels=20,
    width=64, modes=16, n_layers=4,
))
ds = Burgers1dDataset(n_samples=200, t_in=10, t_out=20)

trainer = Trainer(model, lambda p, t: torch.mean((p - t) ** 2), ds,
                  TrainConfig(epochs=5, batch_size=32, lr=1e-3))
trainer.fit()
```

### 4.2 Export to NeuroIR

```python
from neuroflow.ir.export import export_all
json_path, bin_path = export_all(model, out_dir="./artifacts",
                                  basename="fno1d_burgers")
# json_path = artifacts/fno1d_burgers.neuroir
# bin_path = artifacts/fno1d_burgers.nneuroir
```

### 4.3 Quantise

```python
from neuroflow.quant.static_quant import quantise_model, calibrate

# calibration: forward ~80 batches, then build a fake-quant clone
fake_model = quantise_model(
    model,
    calib_inputs=calib_loader,
    mode="per_tensor",          # or "per_channel", "per_token"
    percentile=99.5,           # for p99.5 calibration
    ema_decay=0.99,            # TensorRT-style running min/max
)
fake_model.export("model_int8.neuroir")  # NIRQ block appended
```

Or in **FP8 (E4M3)**:

```python
fake_model = quantise_model(model, mode="fp8_e4m3",
                            calib_inputs=calib_loader)
```

Or **QAT** (closes ~77% of the INT8 accuracy gap on Burgers 1D):

```python
from neuroflow.quant.qat import qat_train_step
# wrap standard training loop with qat_train_step
```

### 4.4 Run inference from Python (C++ backend)

```python
import neuroflow_cpp            # the pybind11 module
y = neuroflow_cpp.infer_arrays(
    model_path="artifacts/fno1d_burgers.nneuroir",
    x=x_np,                       # (B, n, in_channels) float32
)
# y has shape (B, n, out_channels), same dtype
```

## 5. C++ runtime

### 5.1 Public surface

```
nflow_core        : libnflow_core.a       # zero-dep static library
nflow_infer       : nflow_infer.exe       # CLI  (file → file)
neuroflow_cpp     : neuroflow_cpp.pyd     # pybind11 module
test_runtime      : test_runtime.exe     # Tensor + FFT roundtrip + 8 op fwd
test_int8_gemm    : test_int8_gemm.exe   # INT8 GEMM micro-benchmark
test_fix_nflow    : test_fix_nflow.exe   # LAMMPS shim standalone
```

Public headers: `cpp/include/neuroflow/{tensor,fft,fno,runtime,ir_loader,npy_io,quant_types,lammps/*}.h`.

### 5.2 Build

```bash
cmake -S cpp -B cpp/build -DNFLOW_BUILD_PYBIND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

Optional flags:

| Flag | Default | Purpose |
|---|---|---|
| `NFLOW_BUILD_CLI`             | `ON`  | Build `nflow_infer` |
| `NFLOW_BUILD_PYBIND`          | `OFF` | Build `neuroflow_cpp` |
| `NFLOW_BUILD_TESTS`           | `OFF` | Build `test_runtime` |
| `NFLOW_BUILD_INT8_GEMM_BENCH`  | `OFF` | Build `test_int8_gemm` |
| `NFLOW_BUILD_LAMMPS_SHIM`     | `OFF` | Build `test_fix_nflow` standalone |

The build is C++20, with no non-portable features.  Tested on
GCC 13.1, Clang 14, and MSVC 19.30.  On Windows MinGW you need
`#ifndef M_PI` guards around `<cmath>` (MinGW does not expose it by
default) — these are already in the source.

### 5.3 Kernel selection — which Linear / GEMM gets called

The C++ runtime has a **compile-time dispatch ladder** for the
`Linear` layers, applied per-layer.  At inference, the runtime
picks the most specific kernel that matches the operator's
quantisation state and the CPU's capabilities:

```
┌─────────────────────────────────────────────────────────────┐
│ FNO1d::Forward, per-Linear dispatch                          │
└─────────────────────────────────────────────────────────────┘
                            │
   ┌────────────────────────┴────────────────────────┐
   │ NIRQ present?                                    │
   │   ├─ kind = 0/1 (INT8, per-tensor or per-channel)│
   │   │     ├─ AVX-512 VNNI available?               │
   │   │     │     └─→ int8_gemm::LinearForwardVNNI   │
   │   │     ├─ AVX2 available?                      │
   │   │     │     └─→ int8_gemm::LinearForwardAVX2   │
   │   │     └─ else                                 │
   │   │           └─→ int8_gemm::LinearForwardScalar │
   │   ├─ kind = 2 (per-token)                       │
   │   │     └─→ FP32 Linear (per-token needs more    │
   │   │         data-dependent shape handling)       │
   │   ├─ kind = 3 (FP8 E4M3)                        │
   │   │     └─→ FP32 Linear + inline E4M3 quantise   │
   │   └─ no NIRQ (FP32 weights)                     │
   │         └─→ LinearDispatchTryEigen → Eigen GEMM │
└─────────────────────────────────────────────────────────────┘
```

The same dispatch is mirrored in `FNO2d::Forward` and
`FNO3d::Forward`.  The exact selection is logged at
`DEBUG=1` (Stage 4 enhancement to make this runtime-configurable).

## 6. Quantisation — formulas and trade-offs

Given a float32 tensor `x` with min/max `[x_min, x_max]`:

### 6.1 INT8 (per-tensor)

```
qmax  = 127
scale = (x_max - x_min) / qmax
zp    = round(-x_min / scale)              # zero-point, int8
q     = round(x / scale) + zp              # int8
x_hat = (q - zp) * scale                   # dequantised
```

`scale` and `zp` are scalars (whole tensor).  Used in NIRQ `kind=0`.

### 6.2 INT8 (per-channel weight, per-tensor activation)

Same as per-tensor, but the **weight** `scale` and `zp` are
per-output-row.  `W_hat = (W_q - zp_w) * scale_w` then `y = W_hat @ x`.
The activation is still per-tensor.  Used in NIRQ `kind=1`.

### 6.3 INT8 (per-token activation)

`scale_a` and `zp_a` are per-token (per-row of the activation).
Weight is typically per-tensor or per-channel.  Used in NIRQ
`kind=2`.  More robust on activations whose per-sample range
varies wildly; for a narrow per-token range with only a few
calibration samples it can be *worse* than per-tensor (honest
negative result, see Sprint 3.11).

### 6.4 FP8 (E4M3)

E4M3 encodes 1 sign + 4 exponent + 3 mantissa bits; range ±448,
precision degrades quickly near zero.  Used for activations only
in Stage 3 (weights stay INT8 or FP32).  Effective per-step
quantisation:

```
q = round_to_e4m3(x / scale_a)         # scale_a is per-tensor running max
x_hat = e4m3_to_float(q) * scale_a
```

Closes ~50% of the INT8 floor on Burgers 1D (5-seed mean
~2.8e-1 vs PTQ INT8 ~5.4e-1); on narrow-range 2D Poisson it can be
*worse* than INT8 (honest conditional, Sprint 3.19).

### 6.5 QAT (closes ~77% of the INT8 floor on Burgers 1D)

Insert a `FakeQuantLinear` (straight-through estimator) **before
each `Linear` layer** in the training graph and train end-to-end.
Best-val early-stop is essential — the quantised-loss surface is
non-monotonic.  The current `recalibrate_qat()` helper runs
periodic re-calibration every K epochs but is empirically neutral
on Burgers (within noise).

### 6.6 Calibration refinement (percentile + EMA)

The textbook `min/max` calibration captures a few outliers and
wastes most of the int8 range.  Two mitigations:

- **Percentile**: clip `x_max, x_min` to the 99.5th percentile of
  observed activations, recompute scale/zp from the clipped range.
- **EMA**: replace `min/max` over the calibration set with a
  TensorRT-style running `min/max` over training batches.

On Burgers 1D with 80 calibration batches, `p99.5` alone closes
~14% of the per-tensor INT8 floor.

### 6.7 The "INT8 floor" — what it actually means

For a 1D Burgers model trained in FP32 to ~1.2e-4 rel L2, **any**
INT8 PTQ produces a val rel L2 of order 5e-1.  This is not a
quantisation bug; it is the noise floor of a 8-bit narrow-range
non-uniform signal.  QAT and FP8 are the two paths to push past
it.  On 2D narrow-range PDEs, the INT8 floor is even higher.

## 7. Cross-language parity

Trained on Burgers 1D, exported, reloaded, run in C++.  All
operators, all quantisation modes:

| Op | C++ vs PyTorch max abs diff |
|---|---:|
| FNO1d       | 5.2e-5 |
| FNO2d       | 2.2e-6 |
| FNO3d       | 4.1e-6 |
| DeepONet    | 8.4e-5 |
| TokenMixer  | 1.8e-4 |
| GraphOp     | 3.6e-4 |
| TokenMixer2D | 5.4e-5 |
| GraphOp2D   | 1.3e-4 |

The residual is dominated by float32 summation-order differences
between PyTorch's BLAS-backed GEMM and the unrolled C++ loops.
Targets are < 1e-3 across the board.

## 8. Repository layout

```
neuroflow/
├── README.md                       # this file
├── LICENSE                         # Apache-2.0
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── GOVERNANCE.md
├── CITATION.cff
├── CHANGELOG.md
├── ROADMAP.md
├── STATUS.md
├── pyproject.toml
├── .github/
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── neuroflow/                      # Python research layer
│   ├── nn/                         # FNO1d, FNO2d, FNO3d, DeepONet, TokenMixer, GraphOp + 2D variants
│   ├── data/                       # Burgers 1D, Poisson 2D/3D, integral operator
│   ├── train/                      # Trainer
│   ├── ir/                         # spec.py, export.py, load.py
│   ├── quant/                      # static_quant.py, qat.py
│   └── utils/
├── cpp/                            # C++ production runtime
│   ├── CMakeLists.txt
│   ├── include/neuroflow/           # tensor.h, fft.h, fno.h, runtime.h, ...
│   ├── src/                        # tensor.cpp, fft.cpp, fno.cpp, ...
│   ├── bindings/                   # pybind_module.cpp
│   ├── tests/                      # test_runtime.cpp + INT8 / shim tests
│   ├── third_party/eigen-3.4.0/    # vendored (linear-algebra backend)
│   └── lammps/                     # `fix nflow` shim (LAMMPS-agnostic core)
├── domains/
│   └── lammps/                     # NeuroFlow × LAMMPS SDK
├── examples/                       # 27 end-to-end examples
├── tests/                          # pytest suite
├── paper/                          # arXiv-ready LaTeX
└── artifacts/                      # pre-trained models, plots (gitignored)
```

## 9. Build & install

### Python

```bash
git clone https://github.com/AITree69/neuroflow
cd neuroflow
python -m venv .venv
source .venv/Scripts/activate     # Git Bash on Windows
pip install -e .[dev]            # adds pytest, hypothesis, ruff, mypy
```

### C++ runtime + pybind11

```bash
cmake -S cpp -B cpp/build -DNFLOW_BUILD_PYBIND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
# Linux / macOS
cp cpp/build/neuroflow_cpp*.so .
# Windows (MinGW)
cp cpp/build/neuroflow_cpp.cp312-win_amd64.pyd ./neuroflow_cpp.pyd
```

### Verify

```bash
pytest tests/ -v
ctest --test-dir cpp/build --output-on-failure
```

## 10. Five-minute end-to-end demo

The smallest reproducible run — train an FNO1d, export to
`NeuroIR`, run through the C++ runtime:

```bash
python examples/01_train_burgers1d.py --epochs 5 --n-train 100
python examples/02_export_and_infer.py
```

Expected output:

```
==> C++ runtime inference
   max abs diff vs PyTorch = 5.20e-05
   C++ latency (batch=1, n=256) = 11.0 ms
==> Pure-Python latency benchmark
   Python latency (batch=1, n=256) = 6.0 ms
```

## 11. Adding a new operator

1. Add an op code to `neuroflow/ir/spec.py` and a config field set
   in `_OP_SCHEMAS`.
2. Implement the Python reference in `neuroflow/nn/my_op.py`
   inheriting the `state_dict_for_ir()` convention from
   `FNO1d`.
3. Implement the C++ forward in `cpp/src/fno.cpp` (or a new
   `my_op.cpp`) and add a `RunMyOp` method to `InferenceRuntime`.
4. Add a unit test in `tests/test_my_op.py` that:
   - Checks the forward shape
   - Loads the exported IR and checks forward
   - Loads the binary IR and asserts the byte layout
5. Add a C++ test in `cpp/tests/test_runtime.cpp` that loads the
   exported `.nneuroir` and asserts the C++-vs-Python max abs diff
   is within the target (< 1e-3 for FP32 weights; < 1e-1 for INT8).
6. Add an `examples/NN_my_op_*.py` (train + export) pair.
7. Update the op-code table in §3 above.

The codegen refactor (Sprint 3.27) makes (1) a single-row
addition to the `_OP_SCHEMAS` registry; the existing dispatcher
in `export_to_neuroir` and `export_to_binary` picks up new ops
without per-op code.

## 12. Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development
workflow, PR conventions, and the Apache-2.0 CLA.
[`GOVERNANCE.md`](GOVERNANCE.md) explains the TSC and SIG structure.
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) is the Contributor
Covenant v2.1.

Bug reports, feature requests and design discussions use the GitHub
issue templates under `.github/ISSUE_TEMPLATE/`.  Open a
**Pull Request template** PR for code; use **Discussions** for
everything else.

## 13. Citing

[`CITATION.cff`](CITATION.cff) holds machine-readable metadata; the
Zenodo DOI is filled in automatically on the first tagged GitHub
release.  The Stage-1 writeup is
[`paper/paper.pdf`](paper/paper.pdf) (arXiv-ready).

## 14. License

Code: Apache-2.0.  Documentation: CC BY 4.0.  See [`LICENSE`](LICENSE).
