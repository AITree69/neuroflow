# Stage 2 Plan — Operator Coverage (M9–M15)

> **Goal of Stage 2:** grow the operator zoo from "FNO1d only" to
> "the four operators that cover 80% of PDE-surrogate use cases",
> while keeping the IR v0 stable.

## Status (2026-06-05)

Sprint 1 (FNO2d end-to-end) is **fully closed** in v0.3.0. Sprint 2
FNO3d is **fully closed** in v0.4.0. Sprint 2 DeepONet is **fully
closed** in v0.5.0. Sprint 2 Eigen integration is **fully closed**
in v0.6.0 — DeepONet latency dropped ~2.8× (15 ms -> 5.4 ms); all
four operators preserve their Sprint 1 / 2.1 / 2.2 cross-language
parity. See `docs/ir_v1_migration.md` for the IR compatibility
statement and `CHANGELOG.md` for the v0.6.0 performance numbers.

## Headline deliverables

| Operator | Python | C++ | NeuroIR | Target | Status |
|---|---|---|---|---|---|
| FNO2d | v0.2.0 | v0.3.0 | v1 (modes_h, modes_w) | Sprint 1 | **done** |
| FNO3d | v0.4.0 | v0.4.0 | v1.1 (modes_d, 8 int32) | Sprint 2 | **done** |
| DeepONet | v0.5.0 | v0.5.0 | v1.1 (op_code 0x04, 7 int32) | Sprint 2 | **done** |
| Eigen integration | v0.6.0 | v0.6.0 | n/a | Sprint 2.3 | **done** |
| Transolver | pending | pending | v1.2 | Sprint 3 | next |
| Quantization (INT8/FP8) | pending | pending | v1.2 | Sprint 3 | pending |
| First SDK `heat` (chip thermal) | pending | reuses FNO2d | v1.1 | Sprint 4 | pending |

## Sprint 1 — FNO2d end-to-end

**Exit criteria — all closed:**

1. ✅ `FNO2d` in `neuroflow/nn/fno2d.py` — same API shape as `FNO1d`.
2. ✅ Trained FNO2d reaches **< 1% relative L2 error** on a 2D PDE
   task (Poisson, 32x32 grid, default config → 0.73% rel L2 on
   val[0]; avg val = 0.96%).
3. ✅ C++ `SpectralConv2d` matches PyTorch forward to **< 1e-3 max
   abs diff** on the same checkpoint. **Achieved 2.19e-06**.
4. ✅ `NeuroIR` v1 (`version=0.2.0`) carries `modes_h` / `modes_w`
   while remaining backward-compatible with v0 readers.
5. ✅ `examples/03_train_fno2d.py` and
   `examples/04_export_and_infer_fno2d.py` mirror the Burgers 1D pair.
6. ✅ `tests/test_fno2d.py` (6 tests, all passing).
7. ⏳ CI green across the OS × Python matrix. The CI workflow already
   runs the full `pytest` on Linux/macOS/Windows × Python 3.10-3.12;
   this becomes "true" after we push to GitHub. The `cpp/build-cpp`
   job already runs `cmake -S cpp -B build` and `test_runtime`; that
   passes locally and will pass on Linux CI.
8. ✅ `CHANGELOG.md` updated; new entry tagged `0.3.0`.

## Headline deliverables

| Operator | Python | C++ | NeuroIR | Target |
|---|---|---|---|---|
| FNO2d | ✅ v0.2.0 | ⏳ Sprint 1 Phase 2 | v1 (modes_h, modes_w) | Sprint 1 |
| FNO3d | pending | pending | v1.1 | Sprint 2 |
| DeepONet | pending | pending | v1.1 (new op code) | Sprint 2 |
| Transolver | pending | pending | v1.2 | Sprint 3 |
| Quantization (INT8/FP8) | pending | pending | v1.2 | Sprint 3 |
| First SDK `heat` (chip thermal) | pending | reuses FNO2d | v1.1 | Sprint 4 |

## Sprint 1 — FNO2d end-to-end

**Exit criteria:**

1. ✅ `FNO2d` in `neuroflow/nn/fno2d.py` — same API shape as `FNO1d`.
2. ✅ Trained FNO2d reaches **< 1% relative L2 error** on a 2D PDE
   task (Poisson, 32x32 grid, default config → 0.73% rel L2 on
   val[0]; avg val = 0.96%).
3. ⏳ C++ `SpectralConv2d` matches PyTorch forward to **< 1e-3 max
   abs diff** on the same checkpoint. **Phase 2**.
4. ✅ `NeuroIR` v1 (`version=0.2.0`) carries `modes_h` / `modes_w`
   while remaining backward-compatible with v0 readers.
5. ✅ `examples/03_train_fno2d.py` and
   `examples/04_export_and_infer_fno2d.py` mirror the Burgers 1D pair.
6. ✅ `tests/test_fno2d.py` (6 tests, all passing).
7. ⏳ CI green across the OS × Python matrix. The CI workflow already
   runs the full `pytest` on Linux/macOS/Windows × Python 3.10-3.12;
   this becomes "true" after we push to GitHub.
8. ✅ `CHANGELOG.md` updated; new entry tagged `0.2.0` (Python side).

**Cross-cutting (parallel to Sprint 1):**

- **Eigen-backed GEMM.** Swap the unrolled Linear + spectral-convolution
  loops in C++ for `Eigen::Matrix` operations. This is the single biggest
  CPU-only speedup available without leaving C++.
- **Multi-seed Burgers 1D baseline.** 5 seeds; report mean ± std of
  (a) test L2, (b) C++-vs-PyTorch max abs diff. Required for the
  camera-ready paper.

## Sprint 1 Phase 2 — C++ side (next session)

Open work, in execution order:

1. **C++ `fft2d`**: row-wise `Rfft` + column-wise `fft`. Both H and W
   must be powers of two. Re-use the existing radix-2 `ComplexFftInPlace`
   from `cpp/src/fft.cpp`; no third-party FFT library.
2. **`SpectralConv2d` + `FNO2d`** in `cpp/include/neuroflow/fno.h` /
   `cpp/src/fno.cpp`. Mirror the Python layout exactly:
   - `weights_real`, `weights_imag`: shape `(in=w, out=w, modes_h, modes_w)`
     row-major; einsum `"bimn,iomn->bomn"`.
   - Lifting / locs / proj_q / proj_out stay in the existing
     `Linear` helper, just with extra axes for the permutations.
3. **IR v1 reader** in `cpp/src/ir_loader.cpp` (header-only file
   `cpp/include/neuroflow/ir_loader.h`):
   - Accept `version = 1` (legacy FNO1d) and `version = 2` (FNO1d | FNO2d).
   - `op_code = 0x02` dispatches to FNO2d weight reads:
     `specs.{i}.weights_real/imag` have `ndim = 4`.
4. **`nflow::LoadedModel`** gains `fno2d_cfg` / `fno2d_weights`; runtime
   constructs `nflow::fno::FNO2d` accordingly. The high-level
   `InferenceRuntime::Run` must take a 4D input/output shape for FNO2d.
5. **pybind11 binding** in `cpp/bindings/pybind_module.cpp`:
   - `infer_arrays` accepts both 3D (FNO1d) and 4D (FNO2d) numpy
     arrays; dispatches by `op` returned from the loaded model.
6. **Tests**:
   - `cpp/tests/test_runtime.cpp` gets a 2D FFT roundtrip and a
     `SpectralConv2d` parity check.
   - `examples/04_export_and_infer_fno2d.py` already drives this end-
     to-end via `neuroflow_cpp`; if it prints "max abs diff vs PyTorch
     = 1.0e-04" or smaller, Sprint 1 is closed.

## Why this order

FNO2d is the highest-leverage single addition: it unlocks the entire
2D PDE universe (Darcy, 2D heat, shallow water, Navier–Stokes 2D) and
the `heat` domain SDK. FNO3d is mostly "FNO2d with an extra axis" and
is fast to add once FNO2d ships. DeepONet and Transolver are
qualitatively different operator families; they need their own
NeuroIR op codes and are best done in parallel with FNO3d in Sprint 2.

## Risks

- **PyTorch 2D FFT semantics.** `torch.fft.rfftn(x, dim=(-2,-1))` and
  `torch.fft.irfftn(...)` have a `s=(H, W)` argument to disambiguate
  when the last axis is odd. The C++ path must match this exactly.
  Expect 1–2 days of parity debugging.
- **IR v1 migration.** v0 readers must keep working. The plan is to
  keep v0 untouched and add v1; new operators write v1; v0 readers
  reject v1 files with `UnsupportedOp` (op=0x02 unknown).
- **Numerical floor.** The current Stage 1 FNO1d C++ vs PyTorch
  parity is 5.20e-05 (float32 summation order). We aim for < 1e-3 on
  FNO2d, which is much looser than the 1d case and should be easy to
  hit; the real risk is getting trapped in 1e-4 noise.

## Open questions

- Eigen is header-only in the standard distribution. Bundle a single
  pinned version, or require it via `find_package(Eigen3)`? Decision
  deferred.
- Do we need CUDA detection in the C++ build (Stage 2 sets the hooks,
  Stage 4 implements)? Lean yes, with a CMake `option(NFLOW_WITH_CUDA
  OFF)` defaulting to off.
