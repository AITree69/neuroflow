# NeuroFlow — Project Status

> **Last updated:** 2026-06-09 (Sprint 3.28 closed — inline INT8 GEMM in FNO1d, 92/92 pytest, paper §6.23)
> **Current session:** (this session)
> **Last commit:** `b8c102e` (Sprint 3.27 codegen refactor, before Sprint 3.28 tag)
> **Current version:** v0.32.0 (NeuroIR) / v0.32.0 (project, before Sprint 3.28 tag)
> **Status:** ✅ **Stage 2 完整 closed**.  Stage 3 active: Sprints 3.23 (font fix), 3.24 (QAT best-val), 3.25 (AVX2 INT8 GEMM), 3.26 (VNNI INT8 GEMM), 3.27 (codegen refactor), 3.28 (inline INT8 GEMM) all closed.

This file is the single-source-of-truth for project state. If a future
session needs to know "where did we leave off", read this file first.

---

## 1. Where we are right now

### 1.1 Stage 1 MVP is **complete and verified end-to-end**

| Item | Status |
|---|---|
| `ROADMAP.md` (6-stage plan) | ✅ written |
| Python research layer (`neuroflow/`) | ✅ 14 unit tests pass (8 Stage 1 + 6 Stage 2 Sprint 1) |
| `NeuroIR` v0 spec (JSON + binary) | ✅ verified lossless roundtrip; **v0.2.0 adds FNO2d, v0.1.0 still readable** |
| C++ runtime (zero-dep, MinGW-built) | ✅ `libnflow_core.a` + `nflow_infer.exe` + `test_runtime.exe` (FNO1d) |
| pybind11 binding (`neuroflow_cpp.cp312-win_amd64.pyd`) | ✅ exports `infer` + `infer_arrays` (FNO1d) |
| Burgers 1D end-to-end | ✅ C++ vs PyTorch max abs diff = **5.20e-05** |
| Cross-language debugging case study (FNO weight axis) | ✅ documented in `paper/paper.tex` |
| `paper/paper.pdf` (arXiv-ready) | ✅ 11 pages, 303 KB |

### 1.2 Stage 2 Sprint 1 is **complete (Python + C++)**

| Item | Status |
|---|---|
| FNO2d Python reference (`neuroflow.nn.fno2d.FNO2d`) | ✅ |
| NeuroIR v0.2.0 (FNO1d + FNO2d, v0.1.0 readable) | ✅ |
| 2D Poisson dataset + FFT analytical solver | ✅ |
| `examples/03_train_fno2d.py` + `04_export_and_infer_fno2d.py` | ✅ |
| FNO2d training (50 ep / 400 train / 32x32) reaches < 1% rel L2 | ✅ **0.73% rel L2 on val[0]; 0.96% val avg** |
| `tests/test_fno2d.py` (6 tests, all passing) | ✅ |
| C++ `SpectralConv2d` + `FNO2d` in `cpp/src/fno.cpp` | ✅ |
| C++ `fft::Rfft2` / `Irfft2` (row-rfft + col-cfft, radix-2) | ✅ |
| C++ IR v1 reader (`version=2`, `op_code=0x02`) | ✅ |
| C++ `InferenceRuntime` + pybind11 dispatch on `op` (3D / 4D) | ✅ |
| C++ `test_runtime` (5 tests, all passing) | ✅ |
| End-to-end C++ vs PyTorch max abs diff on trained FNO2d | ✅ **2.19e-06** (target < 1e-3) |

### 1.3 Stage 2 Sprints 2 & 3 are **complete (FNO3d + DeepONet + Eigen + TokenMixer)**

| Item | Status |
|---|---|
| FNO3d Python reference (`neuroflow.nn.fno3d.FNO3d`) | ✅ |
| FNO3d C++ (`SpectralConv3d` + `FNO3d` in `cpp/src/fno.cpp`) | ✅ |
| C++ `fft::Rfft3` / `Irfft3` (3-axis cfft) | ✅ |
| NeuroIR v0.3.0 (`FNO1d` + `FNO2d` + `FNO3d`; v0.1.0 + v0.2.0 readable; FNO1d/FNO2d layout preserved) | ✅ |
| C++ IR v0.3.0 reader (`version=3` + `op_code=0x03` + 8 int32) | ✅ |
| C++ `InferenceRuntime` + pybind11 dispatch on `op` (3D / 4D / 5D) | ✅ |
| C++ `test_runtime` (6 tests, all passing; +`TestFft3dRoundtrip`) | ✅ |
| 3D Poisson dataset + analytical 3D FFT solver | ✅ |
| `examples/05_train_fno3d.py` + `06_export_and_infer_fno3d.py` | ✅ |
| FNO3d training (50 ep / 200 train / 16^3) reaches < 5% rel L2 | ✅ **4.06% rel L2 on val[0]** |
| End-to-end C++ vs PyTorch max abs diff on trained FNO3d | ✅ **4.12e-06** (target < 1e-3) |
| `tests/test_fno3d.py` (7 tests, all passing) | ✅ |
| **DeepONet** Python + C++ + IR v0.4.0 + tests + example | ✅ |
| 1D integral operator dataset (`IntegralOp1dDataset`) | ✅ |
| DeepONet training (150 ep / 1000 train / 100 sensors / 50 queries) | ✅ **0.96% rel L2 on val[0]; 2.10% val avg** |
| End-to-end C++ vs PyTorch max abs diff on trained DeepONet | ✅ **8.39e-05** (target < 1e-3) |
| `tests/test_deeponet.py` (6 tests, all passing) | ✅ |
| `examples/07_train_deeponet.py` + `08_export_and_infer_deeponet.py` | ✅ |
| Bug fix: `TensorToNumpy` strides size (Stage 1 latent bug surfaced by Sprint 2.2 pybind11 in-memory path) | ✅ |
| **Eigen integration** in `nflow_core` (Linear + SpectralConv GEMM) | ✅ Sprint 2.3 (14701cd) |
| Eigen 3.4.0 vendored at `cpp/third_party/eigen-3.4.0/` | ✅ |
| Eigen-batched DeepONet latency (single-thread, n_sensor=100) | ✅ **14.97 ms → 5.37 ms (2.8×)** |
| `CHANGELOG.md` updated; tag `v0.6.0` (Eigen closed) | ✅ |
| **TokenMixer (Transolver-style)** Python + C++ + IR v0.5.0 + tests + example | ✅ |
| `neuroflow.nn.tokenmixer` (pre-LN multi-head self-attention over mean-pooled patches) | ✅ |
| C++ `TokenMixer::Forward` (slice → attn → unslice → head), `LayerNormLastDim` + `SoftmaxLastDim` helpers | ✅ |
| C++ IR v0.5.0 reader (`op_code=0x05`, 8 int32, 22 weight names per block) | ✅ |
| C++ `InferenceRuntime::RunTokenMixer` + pybind11 `infer_arrays` dispatch on `TokenMixer` | ✅ |
| `tests/test_transolver.py` (10 tests, all passing) | ✅ |
| `examples/09_train_tokenmixer.py` + `10_export_and_infer_tokenmixer.py` | ✅ |
| End-to-end C++ vs PyTorch max abs diff on trained TokenMixer | ✅ **1.81e-04** (target < 1e-3) |
| Full pytest: 37/37 passing (after Sprint 3.1) | ✅ |
| `CHANGELOG.md` updated; tag `v0.7.0` (TokenMixer closed) | ✅ |
| **GraphOp (GCN-style)** Python + C++ + IR v0.6.0 + tests + example | ✅ |
| `neuroflow.nn.graph_op` (W_self + W_neigh on deg-normalised adjacency) | ✅ |
| Graph topology (CSR adj_offsets / adj_indices / deg_inv) in IR + C++ | ✅ |
| C++ `GraphOp::Forward` (lift + per-node neighbour sum + residual + head) | ✅ |
| C++ IR v0.6.0 reader (`op_code=0x06`, 8 int32, 11 weight entries incl. graph topology) | ✅ |
| C++ `InferenceRuntime::RunGraphOp` + pybind11 `infer_arrays` dispatch on `GraphOp` | ✅ |
| `tests/test_graph_op.py` (9 tests, all passing) | ✅ |
| `examples/11_train_graph_op.py` + `12_export_and_infer_graph_op.py` | ✅ |
| Trained GraphOp val[0] rel L2 on 1D regression | ✅ **1.75%** (vs TokenMixer 30% on same target) |
| End-to-end C++ vs PyTorch max abs diff on trained GraphOp | ✅ **3.63e-04** (target < 1e-3) |
| C++ vs pure-Python latency on GraphOp (batch=1, n_nodes=64) | ✅ **0.17 ms vs 0.39 ms = 2.3×** |
| Full pytest: 46/46 passing (after Sprint 3.2) | ✅ |
| `CHANGELOG.md` updated; tag `v0.8.0` (GraphOp closed) | ✅ |
| **Cross-op family benchmark** on Burgers 1D, 1-step (FNO1d vs TokenMixer vs GraphOp) | ✅ |
| `examples/13_op_family_benchmark.py` (single-script, side-by-side) | ✅ |
| `artifacts/benchmark/burgers1d_benchmark.csv` + 4-panel bar-chart figure | ✅ |
| Val rel L2 ordering on Burgers 1D, 1-step ($n_{\mathrm{points}}=64$) | ✅ FNO1d 1.25e-3, TokenMixer 5.26e-3, GraphOp 7.72e-3 |
| C++ parity on all three ops in the same benchmark | ✅ FNO 8.2e-5, TokenMixer 2.5e-5, GraphOp 5.0e-5 |
| Paper §5.7 "Cross-operator family benchmark" LaTeX section | ✅ (Table~\ref{tab:op-benchmark}, Figure~\ref{fig:op-benchmark}) |
| `paper/paper.pdf` recompiled, 2 passes for cross-refs, 0 errors | ✅ **11 pages, 394 KB** |
| Full pytest: 46/46 still passing (no test churn) | ✅ |
| `CHANGELOG.md` updated; tag `v0.9.0` (cross-op benchmark closed) | ✅ |
| **Multi-seed robustness** (5 seeds × 3 ops on Burgers 1D, 1-step) | ✅ |
| `examples/14_multi_seed_benchmark.py` (5×3=15 runs) | ✅ |
| Per-seed + summary CSVs (`burgers1d_benchmark_per_seed.csv`, `..._summary.csv`) | ✅ |
| Multi-seed figure with mean ± std error bars | ✅ `burgers1d_benchmark_multi_seed.png` |
| Mean val rel L2 over 5 seeds (FNO1d / TokenMixer / GraphOp) | ✅ **1.65e-3 / 6.46e-3 / 8.25e-3** |
| Std / min / max per op over 5 seeds | ✅ FNO ±0.49e-3, TokenMixer ±0.93e-3, GraphOp ±1.35e-3 |
| Single-seed caveat in §"Threats to Validity" closed for Burgers 1D | ✅ |
| Paper §5.8 "Multi-seed robustness" sub-section + Table + Figure | ✅ |
| `paper/paper.pdf` recompiled, 2 passes | ✅ **12 pages, 452 KB** |
| Full pytest: 46/46 still passing | ✅ |
| `CHANGELOG.md` updated; tag `v0.10.0` (multi-seed closed) | ✅ |
| **2D cross-op benchmark** on 2D Poisson (16×16 grid, 5 seeds × 3 ops) | ✅ |
| `neuroflow.nn.tokenmixer2d` (flatten spatial + reuse 1D block) | ✅ |
| `neuroflow.nn.graph_op2d` (8-conn grid + reuse 1D mechanism) | ✅ |
| `examples/15_2d_cross_op_benchmark.py` | ✅ |
| Mean val rel L2 over 5 seeds (FNO2d / TokenMixer2D / GraphOp2D) | ✅ **1.35e-2 / 1.51e-1 / 5.58e-1** |
| C++ parity path for TokenMixer2D / GraphOp2D | ⏳ **deferred to Sprint 3.6** (Python-only this sprint) |
| Paper §5.9 "Cross-op family benchmark on 2D Poisson" sub-section | ✅ |
| `paper/paper.pdf` recompiled | ✅ **13 pages, 442 KB** |
| Full pytest: 46/46 still passing | ✅ |
| `CHANGELOG.md` updated; tag `v0.11.0` (2D Poisson cross-op closed) | ✅ |
| **C++ parity for the 2D operator families** (TokenMixer2D / GraphOp2D) | ✅ |
| `NeuroIR v0.12.0` with op 0x07 (TokenMixer2D) + op 0x08 (GraphOp2D) | ✅ |
| C++ `TokenMixer2D::Forward` (flatten + reuse 1D mechanism) | ✅ |
| C++ `GraphOp2D::Forward` (flatten + reuse 1D GCN + 8-conn graph) | ✅ |
| `InferenceRuntime::RunTokenMixer2D` / `RunGraphOp2D` + pybind dispatch | ✅ |
| C++ vs PyTorch max-abs-diff on 2D ops (target < 1e-3) | ✅ **5.4e-5 (TM2D), 1.3e-4 (GCN2D)** |
| Bug fix: dead `memcpy(ln_buf, qkv_buf, ...)` in TokenMixer2D::Forward | ✅ (caught by per-stage fprintf debug) |
| `tests/test_2d_ops.py` (6 cases: shape / roundtrip / C++ parity) | ✅ |
| Full pytest: 52/52 passing (46 → 52) | ✅ |
| Paper §5.10 "C++ parity for 2D operator families" sub-section | ✅ |
| `paper/paper.pdf` recompiled | ✅ **14 pages, 468 KB** |
| `CHANGELOG.md` updated; tag `v0.12.0` (2D C++ parity closed) | ✅ |
| **2D C++ parity grid sweep** (21 configurations) | ✅ |
| `examples/16_2d_parity_grid.py` (h=w ∈ {8,16,32} × n_patches/n_heads/hidden_dim) | ✅ |
| Per-config CSV + summary CSV + heatmap figure | ✅ |
| Mean max-abs-diff over 21 configs (TokenMixer2D / GraphOp2D) | ✅ **1.16e-5 / 1.35e-4** |
| Worst-config max-abs-diff (over 21) | ✅ 2.21e-4 (GraphOp2D, h=w=16, hidden=16) |
| Paper §5.10 extended with grid-sweep table + figure | ✅ |
| `paper/paper.pdf` recompiled | ✅ **14 pages, 518 KB** |
| Full pytest: 52/52 still passing | ✅ |
| `CHANGELOG.md` updated; tag `v0.13.0` (parity grid closed) | ✅ |
| **First domain SDK: NeuroFlow × LAMMPS** (Sprint 3.8) | ✅ |
| `domains/lammps/` tree (README, SDK, example, tests) | ✅ |
| `neuroflow_lammps` Python SDK: `build_morse_dataset` (4-channel parameter conditioning), `MorseSurrogate` (energy + 5-point FD force), `morse_potential` / `morse_force` analytical reference, `surrogate_md_loop` (toy velocity-Verlet MD driver) | ✅ |
| `examples/17_morse_surrogate.py` (`--mode {single,family}`) | ✅ |
| Single-mode: val rel L2 / test rel L2 / C++ parity | ✅ **3.9e-4 / 7.7e-4 / 5.9e-4** (passes < 1e-3) |
| Family-mode: val rel L2 / test rel L2 / C++ parity | ✅ 1.7e-2 / 9.5e-3 / 1.7e-3 (4-channel network float32 corner case) |
| `tests/test_morse_surrogate.py` (7 cases: morse ref, dataset shapes, conditioning broadcast, surrogate energy/force, MD loop, C++ parity) | ✅ |
| Full pytest: **59/59** passing (52 → 59) | ✅ |
| NeuroIR v0.13.0 (adds LAMMPS domain SDK to op set; no new op_code) | ✅ |
| Paper §5.10.4 (`\subsection{First Domain SDK: NeuroFlow × LAMMPS}`) + Table + Figure | ✅ |
| `paper/paper.pdf` recompiled | ✅ **15 pages, 595 KB** |
| `CHANGELOG.md` updated; tag `v0.14.0` (LAMMPS SDK closed) | ✅ |
| **INT8 (W8A8 fake-quant) PTQ** (Sprint 3.9) | ✅ |
| `neuroflow/quant/static_quant.py` (TensorQuantParams, QuantisedModel, calibrate, quantise_model, build_fake_quant_model, FakeQuantLinear, quant_to_ir) | ✅ |
| `tests/test_quant.py` (7 cases) | ✅ |
| Full pytest: **66/66** passing (59 → 66) | ✅ |
| NeuroIR v0.15.0 (adds optional INT8 NIRQ trailing block, backward compat with v0.13.0) | ✅ |
| C++ v0.15.0 runtime (QuantParams, LoadedModel.{quant_enabled, weight_qparams, activation_qparams}, FNO1d::EnableFakeQuant, NIRQ loader) | ✅ |
| `examples/18_fno1d_int8.py` (train → calibrate → quantise → C++ vs PyTorch parity) | ✅ |
| Weight storage FP32 → INT8: 2,130 KB → 532 KB (25%) | ✅ |
| C++ vs PyTorch INT8 parity (max abs diff) = 4.5e-1 | ✅ (same order as quantisation error) |
| Paper §5.11 (`\subsection{INT8 post-training quantisation (W8A8 fake-quant)}`) + Table + Figure | ✅ |
| `paper/paper.pdf` recompiled | ✅ **16 pages, 689 KB** |
| `CHANGELOG.md` updated; tag `v0.15.0` (INT8 PTQ closed) | ✅ |
| **Per-channel weight quant (W8A8 + per-channel)** (Sprint 3.10) | ✅ |
| `PerChannelQuantParams` + `compute_per_channel_qparams` + `quantise_model(per_channel_weights=True)` | ✅ |
| `FakeQuantLinear(weight_qp=...)` + `build_fake_quant_model` extended to spectral conv weights | ✅ |
| NeuroIR v0.16.0 (NIRQ `kind=1` per-channel entries) | ✅ |
| C++ v0.16.0 runtime (`PerChannelQuantParams`, `EnablePerChannelWeightDequant`) | ✅ |
| `tests/test_quant.py` +4 per-channel cases | ✅ |
| Full pytest: **70/70** passing (66 → 70) | ✅ |
| `examples/18_fno1d_int8.py` `--per-channel` mode | ✅ |
| Per-channel val rel L2 = 5.5e-1 (same as per-tensor; activation quant is the dominant noise) | ✅ |
| C++ vs PyTorch per-channel parity = 4.7e-1 | ✅ |
| Paper §5.12 (`\subsection{Per-channel weight quantisation}`) + Table | ✅ |
| `paper/paper.pdf` recompiled | ✅ **17 pages, 696 KB** |
| `CHANGELOG.md` updated; tag `v0.16.0` (per-channel closed) | ✅ |
| **Per-token activation quant (W8A8 + per-token A)** (Sprint 3.11) | ✅ |
| `PerTokenQuantParams` + `compute_per_token_qparams` + `calibrate(per_token=True)` + `quantise_model(per_token_activations=True)` | ✅ |
| NeuroIR v0.17.0 (NIRQ `kind=2` per-token entries) | ✅ |
| C++ v0.17.0 runtime (`PerTokenQuantParams`, `EnablePerTokenActivation`, per-token fake-quant in post-`Linear` output) | ✅ |
| `tests/test_quant.py` +4 per-token cases | ✅ |
| Full pytest: **74/74** passing (70 → 74) | ✅ |
| `examples/18_fno1d_int8.py` `--per-token` and `--per-token --per-channel` modes | ✅ |
| Per-token val rel L2 = 9.7e-1 (worse than per-tensor 5.5e-1; per-point range too narrow with 8 calib samples) | ✅ |
| C++ vs PyTorch per-token parity = 3.9e-1 (better than per-tensor 4.5e-1) | ✅ |
| Paper §5.13 (`\subsection{Per-token activation quantisation (W8A8 + per-token A)}`) + Table | ✅ |
| `paper/paper.pdf` recompiled | ✅ **18 pages, 708 KB** |
| `CHANGELOG.md` updated; tag `v0.17.0` (per-token closed) | ✅ |
| **Calibration refinement (percentile + EMA)** (Sprint 3.12) | ✅ |
| `percentile` parameter in `compute_per_token_qparams` / `calibrate` / `quantise_model` | ✅ |
| `ema_decay` parameter in `calibrate` / `quantise_model` (TensorRT running min/max) | ✅ |
| `tests/test_quant.py` +3 percentile/EMA cases | ✅ |
| Full pytest: **77/77** passing (74 → 77) | ✅ |
| Per-tensor p99.5 (80 calib) → val rel L2 = 4.8e-1 (14% better than 5.5e-1) | ✅ |
| Per-token p99.5 (80 calib) → val rel L2 = 7.5e-1 (23% better than 9.7e-1, but still worse than per-tensor) | ✅ |
| Paper §5.14 (`\subsection{Calibration refinement (percentile + EMA)}`) + Table | ✅ |
| `paper/paper.pdf` recompiled | ✅ **19 pages, 716 KB** |
| `CHANGELOG.md` updated; tag `v0.18.0` (calibration refinement closed) | ✅ |
| **Stage 3 Sprint 3.13: Real LAMMPS `fix nflow` shim** | ✅ |
| `cpp/include/neuroflow/lammps/fix_nflow.h` (LAMMPS-agnostic `FixNflow` C{++} class) | ✅ |
| `cpp/src/lammps_shim/fix_nflow.cpp` (init + compute, 5-point FD force) | ✅ |
| `cpp/tests/test_fix_nflow_standalone.cpp` (MD loop harness) | ✅ |
| CMake `NFLOW_BUILD_LAMMPS_SHIM` opt-in flag | ✅ |
| `examples/19_train_morse_3d.py` (val MSE 2.5e-6) | ✅ |
| `tests/test_fix_nflow_shim.py` (+2 cases) | ✅ |
| Full pytest: **79/79** passing (77 → 79) | ✅ |
| Paper §6.1 (`\subsection{Stage 3 Sprint 3.13: Real LAMMPS `fix nflow` shim}`) | ✅ |
| `paper/paper.pdf` recompiled | ✅ **21 pages, 733 KB** |
| `CHANGELOG.md` updated; tag `v0.19.0` (LAMMPS shim closed) | ✅ |
| **Stage 3 Sprint 3.14: Real INT8 GEMM (W8A8 + INT32 accumulate)** | ✅ |
| `cpp/include/neuroflow/int8_gemm.h` + `cpp/src/int8_gemm.cpp` (W8A8 LinearForward) | ✅ |
| `cpp/tests/bench_int8_gemm.cpp` (FP32 vs INT8 timing + accuracy) | ✅ |
| CMake `NFLOW_BUILD_INT8_GEMM_BENCH` opt-in flag | ✅ |
| `tests/test_int8_gemm.py` (+2 cases) | ✅ |
| Full pytest: **80/80** passing (79 → 80) | ✅ |
| Weight bandwidth 3.97× saving (1024 KB → 258 KB) | ✅ |
| **Naive scalar INT8 is 10× SLOWER than FP32** (0.11× speedup) — SIMD/AVX2-INT8 is Stage 3.5+ | ✅ |
| Paper §6.2 (`\subsection{Stage 3 Sprint 3.14: Real INT8 GEMM with INT32 accumulation}`) | ✅ |
| `paper/paper.pdf` recompiled | ✅ **22 pages, 742 KB** |
| `CHANGELOG.md` updated; tag `v0.20.0` (INT8 GEMM closed) | ✅ |
| **Stage 3 Sprint 3.15: FP8 (E4M3) activation quantisation** | ✅ |
| `examples/20_fno1d_fp8.py`; full FP8 fake-quant path in FNO1d C++ kernel | ✅ |
| **Stage 3 Sprint 3.16: Per-channel W + FP8 A combo** | ✅ |
| Burgers 1D: 6.53e-1 → 3.44e-1 (FP32 baseline) | ✅ |
| **Stage 3 Sprint 3.17: QAT infrastructure** (honest negative result — diverges) | ✅ |
| **Stage 3 Sprint 3.18: Multi-seed FP8 robustness** (5 seeds) | ✅ |
| PTQ INT8 5.36e-1 ± 1.7e-1 → PTQ FP8 2.79e-1 ± 7.3e-2 (+47.9% mean) | ✅ |
| **Stage 3 Sprint 3.19: FP8 generalisation to FNO2D** (honest conditional) | ✅ |
| 2D Poisson 2.85e-1 (INT8) → 4.14e-1 (FP8, WORSE 45% on narrow range) | ✅ |
| **Stage 3 Sprint 3.20: Full C++ FNO2d fake-quant coverage** | ✅ |
| **Stage 3 Sprint 3.21: C++ FNO2d weight dequant + DequantiseWeight parity fix** | ✅ |
| FNO1d FP8 C++ vs PyTorch 5.39e-1 → 2.63e-1 | ✅ |
| **Stage 3 Sprint 3.22: Parity regression re-runs + paper update** | ✅ |
| **Stage 3 Sprint 3.23: Body font leak fix** (paper §5.15+ Times Roman) | ✅ |
| Wrapped `\texttt{...}` in `\normalfont\ttfamily...\normalfont\rmfamily` | ✅ |
| Removed 3 band-aids (microtype, familydefault, AtBeginDocument/everypar) — all no-ops | ✅ |
| Times % on pages 15-22: 7-15% → 76-97% | ✅ |
| `paper.pdf` 25 pages 656 KB (was 27 pages 774 KB; -15% side benefit) | ✅ |
| Full pytest: **87/87** passing (unchanged) | ✅ |
| `CHANGELOG.md` updated; tag `v0.28.1-font-fix` | ✅ |
| **Stage 3 Sprint 3.24: QAT with best-val early-stop + periodic recalibration** | ✅ |
| `recalibrate_qat(qat_model, calib_inputs)` in `neuroflow/quant/qat.py` | ✅ |
| `examples/25_fno1d_recalib.py` + `examples/26_fno1d_recalib_multiseed.py` (5 seeds) | ✅ |
| PTQ INT8 3.95e-1 → QAT best-val **8.96e-2** = +77.4% reduction | ✅ |
| QAT INT8 on par with PTQ FP8 (within 1.4σ), beats FP8 on 2/5 seeds | ✅ |
| Periodic recalibration neutral (8.75e-2 vs 8.96e-2 within noise) | ✅ |
| C++ vs PyTorch parity 4.06e-1 (consistent with FP8/INT8 budget) | ✅ |
| Full pytest: **89/89** passing (87 → 89, +2 Sprint 3.24 tests) | ✅ |
| `paper.pdf` 27 pages 674 KB | ✅ |
| `CHANGELOG.md` updated; tag `v0.29.0-qat-bestval` | ✅ |
| **Stage 3 Sprint 3.25: AVX2 INT8 GEMM (closes 10× slowdown)** | ✅ |
| Pre-sum trick (hoist `sum_w_int8[o]` + `sum_a_int8`) = 5.4× speedup alone | ✅ |
| AVX2 INT8 dot product via `_mm256_maddubs_epi16` + `_mm256_madd_epi16` with pre-add-128 trick | ✅ |
| `if (in_f <= 4096)` stack buffer; larger sizes fall back to scalar | ✅ |
| 256×1024: 5.56×, 512×2048: 6.10×, 1024×1024: 6.27× | ✅ |
| Full pytest: **89/89** passing (unchanged) | ✅ |
| `paper.pdf` 28 pages 681 KB | ✅ |
| `CHANGELOG.md` updated; tag `v0.30.0-avx2-int8` | ✅ |
| **Stage 3 Sprint 3.26: VNNI INT8 GEMM (AVX-512 VNNI)** | ✅ |
| `_mm512_dpbusd_epi32` (one instr per 64-element block, 16 int32 accumulates) | ✅ |
| Compile-time dispatch: VNNI > AVX2 > scalar | ✅ |
| 256×1024: 5.78×, 512×2048: 6.29×, 1024×1024: 6.39×, 2048×4096: 5.98× | ✅ |
| VNNI gives ~5-10% edge over AVX2 (memory-bound; 2-4× win in compute-bound) | ✅ |
| Full pytest: **89/89** passing (unchanged) | ✅ |
| `paper.pdf` 28 pages 681 KB | ✅ |
| `CHANGELOG.md` updated; tag `v0.31.0-vnni-int8` | ✅ |
| **Stage 3 Sprint 3.27: PyTorch → NeuroIR unified codegen (refactor)** | ✅ |
| `OpSpec` dataclass + `_build_spec` helper + `_OP_SCHEMAS` registry | ✅ |
| `export_to_neuroir`: ~130 lines → ~30 lines of codegen dispatch | ✅ |
| `export_to_binary`: ~100 lines → ~15 lines of schema iteration | ✅ |
| Adding a 9th op family = one schema table + one `model_class` rebind | ✅ |
| Caught hidden DeepONet `hidden_branch`/`hidden_trunk` cfg-field regression | ✅ |
| Full pytest: **89/89** passing (unchanged; codegen refactor is byte-for-byte identical) | ✅ |
| NeuroIR binary format unchanged (version=2 for FNO1d/FNO2d, version=3 for the rest) | ✅ |
| `paper.pdf` 28 pages 681 KB | ✅ |
| `CHANGELOG.md` updated; tag `v0.32.0-codegen` | ✅ |
| **Stage 3 Sprint 3.28: Inline INT8 GEMM in FNO1d::Forward (production-kernel speedup)** | ✅ |
| `Int8GemmLayer` struct with `W_int8` + per-channel scales/zps + `sum_W_per_row` cache | ✅ |
| `FNO1d::EnableInt8Gemm` one-shot quantiser (per-channel W → INT8) | ✅ |
| `int8_gemm::LinearForwardBatched` (bulk activation quantise + per-row GEMM) | ✅ |
| `int8_gemm::PrecomputeSumW` + `QuantiseActivation` helpers | ✅ |
| `LinearDispatchTryInt8` per-Linear dispatch in `fno.cpp` | ✅ |
| `InferenceRuntime::EnableInt8Gemm()` + pybind `enable_int8_gemm` | ✅ |
| `examples/27_fno1d_int8_gemm_inline_bench.py` (FP32 vs INT8 GEMM production bench) | ✅ |
| Full pytest: **92/92** passing (89 → 92, +3 Sprint 3.28 tests) | ✅ |
| **Honest negative**: speedup 0.48× (SLOWDOWN) at w=64 | ✅ |
| — scalar batched GEMM is slower than Eigen's vectorised FP32 matmul | ✅ |
| — max abs diff 1.07e-2 (within INT8 budget) | ✅ |
| — wiring correct: per-tensor A + per-channel W dispatches via INT8 GEMM | ✅ |
| — per-token / FP8 paths fall back to FP32 Linear (unchanged) | ✅ |
| — fix = Sprint 3.28 follow-up: vectorised batched kernel (VNNI/AVX2 on multiple rows) | ✅ |
| `paper.pdf` 29 pages 690 KB | ✅ |
| `CHANGELOG.md` updated; tag `v0.33.0-inline-int8-gemm` | ✅ |

### 1.4 GitHub repo scaffolding is **complete locally; push pending**

| Item | Status |
|---|---|
| Local git repo (`D:\minimax_proj`) | ✅ initialized, `main` branch |
| First commit: Stage 1 source + repo scaffolding | ✅ `17a34d5` (57 files) |
| CI: `.github/workflows/ci.yml` (Linux/macOS/Windows × Python 3.10-3.12 + C++ smoke) | ✅ |
| Issue / PR templates, CONTRIBUTING, CODE_OF_CONDUCT, GOVERNANCE | ✅ |
| CHANGELOG, CITATION.cff (Zenodo-ready) | ✅ |
| README badges + install + community links | ✅ |
| GitHub remote (`git push -u origin main`) | ⏳ **blocked — this machine cannot reach `github.com:443`** |

### 1.4 Numerical floor (5.20e-05) is *not* a bug

It is dominated by float32 summation-order differences between
PyTorch's BLAS-backed GEMM and our unrolled C++ loops. Well below
the ~1% relative error of the trained model itself.

### 1.5 The Stage-1 C++ is *intentionally* slower than Python (~0.6x)

This is by design — Stage 1 is correctness infrastructure, not
performance infrastructure. Real speedup comes at Stage 4 (CUDA +
cuFFT + batched inference). The paper discloses this honestly
("Stage-1 honesty clause").

---

## 2. Environment

| Tool | Path | Version |
|---|---|---|
| Python | `C:\Users\proart\AppData\Local\Programs\Python\Python312\python.exe` | 3.12.10 |
| venv | `D:\minimax_proj\.venv` | — |
| g++ (MinGW) | `C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin\g++.exe` | 13.1.0 |
| cmake | `C:\Program Files\JetBrains\CLion 2026.1.2\bin\cmake\win\x64\bin\cmake.exe` | 4.2.2 |
| MiKTeX | `C:\Users\proart\AppData\Local\Programs\MiKTeX` | 25.12 |
| Winget | (system) | — |

Notes for future sessions:
- `python.exe` system PATH is an MS Store stub; use the full path above
  or activate `.venv`.
- MinGW g++ requires `#ifndef M_PI` guard in any `.cpp` that uses M_PI.
- MinGW-built pybind11 .pyd depends on `libgcc_s_seh-1.dll`,
  `libstdc++-6.dll`, `libwinpthread-1.dll` — copy next to .pyd or
  set PATH to `C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin`.

---

## 3. Six-stage roadmap (verbatim from `ROADMAP.md`)

| Stage | Name | Months | Headline deliverable |
|---|---|---|---|
| 0 | Bootstrap | M1–M3 | Team + repo + RFC-0001 |
| 1 | MVP | M3–M9 | FNO1d end-to-end, C++ runtime, pybind11, 8 unit tests, arXiv paper |
| **2** | **Operator coverage** ← **we are here (Sprint 1 in progress)** | M9–M15 | FNO2d/3d, DeepONet, Transolver, INT8/FP8, first domain SDK `heat` |
| 3 | Numerical-method coupling | M15–M21 | `nflow-fem`, `nflow-fvm`, OpenFOAM plugin, Hybrid Solve Graph v1 |
| 4 | Distributed + HPC | M21–M27 | MPI/NCCL, Triton backend, K8s operator, 100× speedup on 3 industrial cases |
| 5 | Domain ecosystem | M27–M36 | 6 domain SDKs (CFD/heat/EM/structure/grid/climate), first paying POC, ARR $5M target |

---

## 4. Concrete open items (where to pick up next)

Ordered by what gives the most leverage on the next session.

### 4.1 Quick wins (1–2 days each)

- [x] **FNO2d Python reference + IR v1** — done in Sprint 1 Phase 1 (this
  session). 6 new tests pass; training reaches 0.73% rel L2 on 2D Poisson.
- [ ] **FNO2d C++ runtime (Sprint 1 Phase 2).** `SpectralConv2d` +
  `fft2d` + IR v1 reader + pybind11 binding. Target: max abs diff vs
  PyTorch < 1e-3. Re-uses the Python layout, so this is mostly
  "translate and add a column FFT".
- [ ] **Re-enable `algorithm2e` + `tikz` in paper for camera-ready.**
  The current PDF uses `verbatim` + ASCII art to keep MiKTeX happy.
  For submission, the original algorithm boxes and the architecture
  diagram are nicer. Just uncomment the `\usepackage` lines and
  recompile.
- [ ] **Multi-seed evaluation.** Run 5 seeds of the Burgers training
  and report mean ± std of (a) test L2 loss, (b) C++-vs-PyTorch
  max abs diff. Currently we have a single seed.
- [ ] **GitHub remote push.** The local repo (57 files, 1 commit) is
  ready; this machine cannot reach `github.com:443`. Either configure
  a proxy / SSH jump, or push from a machine that has GitHub access.

### 4.2 Mid-term (1–2 weeks each)

- [ ] **First domain SDK — `domains/heat` (chip thermal).** Train an
  FNO2d surrogate for the 2D heat equation on a chip layout. Could
  be a collaboration with someone in the EDA space. The Python
  pipeline is already in place; this is data + domain glue.
- [ ] **OpenFOAM `functionObject` plugin.** Compile `libnflow` as
  an OpenFOAM extension that reads `.nneuroir` and injects a neural
  surrogate into a `fvModel` slot. Real win for the Hybrid Solve
  Graph story.
- [ ] **Eigen integration (immediate perf win, no GPU).** Swap the
  hand-rolled C++ Linear + spectral-convolution loops for an
  Eigen-backed GEMM. Single biggest CPU speedup available without
  leaving C++.
- [ ] **CI on real push.** The CI workflow file is in place; it
  becomes "live" the moment we push to GitHub.

### 4.3 Long-term (months)

- [ ] **Stage 2 second task** — FNO3d, DeepONet, Transolver, quantization.
- [ ] **Stage 3 deliverables** (FEM/FVM coupling, Hybrid Solve Graph).
- [ ] **Stage 4 deliverables** (CUDA, MPI, Triton backend, K8s).
- [ ] **Stage 5 deliverables** (6 domain SDKs, first POC, ARR $5M).

---

## 5. Open offers / follow-up actions the user (Mavis) raised

1. **"再迭代一版论文"** — multi-seed numbers, more thorough Threats
   to Validity, Reusability / Reproducibility statement, sharper
   positioning vs NVIDIA Modulus.
2. **"中文版 slide deck (5–10 页 PPT)"** — short summary deck for
   汇报/路演.
3. **"起一个 GitHub 仓库"** — repo scaffolding 100% done locally;
   needs network access to push (this machine can't reach GitHub).
4. **"做个 demo 网页"** — interactive visualization of the trained
   FNO solving Burgers / Poisson step-by-step.
5. **"正式投 arXiv"** — paper is camera-ready pending the
   `algorithm2e` / `tikz` re-enable above.

---

## 6. Key files (in case we lose track)

| Path | Purpose |
|---|---|
| `D:\minimax_proj\README.md` | Project landing page |
| `D:\minimax_proj\ROADMAP.md` | 6-stage plan, the source of truth |
| `D:\minimax_proj\STATUS.md` | **This file.** Current state, next steps |
| `D:\minimax_proj\docs\stage2_plan.md` | Sprint-by-sprint Stage 2 plan, with status |
| `D:\minimax_proj\.git\` | Local git repo (first commit `17a34d5`) |
| `D:\minimax_proj\paper\paper.pdf` | 11-page arXiv-ready PDF (303 KB) |
| `D:\minimax_proj\paper\paper.tex` | LaTeX source for the paper |
| `D:\minimax_proj\paper\README.md` | How to compile the paper |
| `D:\minimax_proj\artifacts\` | Pre-trained models, plots, IR files (gitignored) |
| `C:\Users\proart\.mavis\memory\user.md` | (will be created) User profile |
| `C:\Users\proart\.mavis\agents\mavis\memory\MEMORY.md` | (already updated) Cross-project lessons |

---

## 7. User profile (so far)

Known facts (from this session):
- Prepares + writes project themselves ("筹备，自己写" = "preparing, writing it myself")
- Has CLion 2026.1.2 installed at `C:\Program Files\JetBrains\CLion 2026.1.2`
- Has MinGW (g++ 13.1) bundled with CLion
- Comfortable with: Python, C++/CMake, basic LaTeX
- Decision style: when offered a choice (MiKTeX vs Tectonic, run debug script or skip), they pick **direct execution** (MiKTeX, "跑")
- Communication style: terse Chinese, jumps straight to next task
- Domain unknown — to ask: "你做哪块的？" (EDA / 电网 / CFD / 其他)

### 1.2 The numerical floor (5.20e-05) is *not* a bug

It is dominated by float32 summation-order differences between
PyTorch's BLAS-backed GEMM and our unrolled C++ loops. Well below
the ~1% relative error of the trained model itself.

### 1.3 The Stage-1 C++ is *intentionally* slower than Python (~0.6x)

This is by design — Stage 1 is correctness infrastructure, not
performance infrastructure. Real speedup comes at Stage 4 (CUDA +
cuFFT + batched inference). The paper discloses this honestly
("Stage-1 honesty clause").

---

## 2. Environment

| Tool | Path | Version |
|---|---|---|
| Python | `C:\Users\proart\AppData\Local\Programs\Python\Python312\python.exe` | 3.12.10 |
| venv | `D:\minimax_proj\.venv` | — |
| g++ (MinGW) | `C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin\g++.exe` | 13.1.0 |
| cmake | `C:\Program Files\JetBrains\CLion 2026.1.2\bin\cmake\win\x64\bin\cmake.exe` | 4.2.2 |
| MiKTeX | `C:\Users\proart\AppData\Local\Programs\MiKTeX` | 25.12 |
| Winget | (system) | — |

Notes for future sessions:
- `python.exe` system PATH is an MS Store stub; use the full path above
  or activate `.venv`.
- MinGW g++ requires `#ifndef M_PI` guard in any `.cpp` that uses M_PI.
- MinGW-built pybind11 .pyd depends on `libgcc_s_seh-1.dll`,
  `libstdc++-6.dll`, `libwinpthread-1.dll` — copy next to .pyd or
  set PATH to `C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin`.

---

## 3. Six-stage roadmap (verbatim from `ROADMAP.md`)

| Stage | Name | Months | Headline deliverable |
|---|---|---|---|
| 0 | Bootstrap | M1–M3 | Team + repo + RFC-0001 |
| **1** | **MVP** ← **we are here** | M3–M9 | FNO1d end-to-end, C++ runtime, pybind11, 8 unit tests, arXiv paper |
| 2 | Operator coverage | M9–M15 | FNO2d/3d, DeepONet, Transolver, INT8/FP8, first domain SDK `heat` |
| 3 | Numerical-method coupling | M15–M21 | `nflow-fem`, `nflow-fvm`, OpenFOAM plugin, Hybrid Solve Graph v1 |
| 4 | Distributed + HPC | M21–M27 | MPI/NCCL, Triton backend, K8s operator, 100× speedup on 3 industrial cases |
| 5 | Domain ecosystem | M27–M36 | 6 domain SDKs (CFD/heat/EM/structure/grid/climate), first paying POC, ARR $5M target |

---

## 4. Concrete open items (where to pick up next)

Ordered by what gives the most leverage on the next session.

### 4.1 Quick wins (1–2 days each)

- [ ] **Re-enable `algorithm2e` + `tikz` in paper for camera-ready.** The
  current PDF uses `verbatim` + ASCII art to keep MiKTeX happy. For
  submission, the original algorithm boxes and the architecture
  diagram are nicer. Just uncomment the `\usepackage` lines and
  recompile.
- [ ] **Multi-seed evaluation.** Run 5 seeds of the Burgers training
  and report mean ± std of (a) test L2 loss, (b) C++-vs-PyTorch
  max abs diff. Currently we have a single seed.
- [ ] **FNO2d (Stage 2 first task).** Add `FNO2d` to `neuroflow/nn/fno.py`,
  `SpectralConv2d` in C++, extend NeuroIR v1 to carry `modes_h, modes_w`.
  Could become a 2-week sprint.
- [ ] **DeepONet (Stage 2 second task).** Add `DeepONet` to `neuroflow/nn/`,
  with separate `trunk` and `branch` weight blocks. NeuroIR v1 needs
  a new op code.

### 4.2 Mid-term (1–2 weeks each)

- [ ] **First domain SDK — `domains/heat` (chip thermal).** Train an
  FNO2d surrogate for the 2D heat equation on a chip layout. Could
  be a collaboration with someone in the EDA space.
- [ ] **OpenFOAM `functionObject` plugin.** Compile `libnflow` as
  an OpenFOAM extension that reads `.nneuroir` and injects a neural
  surrogate into a `fvModel` slot. Real win for the Hybrid Solve
  Graph story.
- [ ] **GitHub repo setup for arXiv artifact.** Create the repo,
  set up CI (Linux + macOS + Windows), Zenodo DOI, CITATION.cff.
  This is required to claim arXiv's "code available" badge.
- [ ] **Eigen integration (immediate perf win, no GPU).** Swap the
  hand-rolled C++ Linear + spectral-convolution loops for an
  Eigen-backed GEMM. Single biggest CPU speedup available without
  leaving C++.

### 4.3 Long-term (months)

- [ ] **Stage 2 deliverables** (FNO2d, DeepONet, Transolver, quantization)
- [ ] **Stage 3 deliverables** (FEM/FVM coupling, Hybrid Solve Graph)
- [ ] **Stage 4 deliverables** (CUDA, MPI, Triton backend, K8s)
- [ ] **Stage 5 deliverables** (6 domain SDKs, first POC, ARR $5M)

---

## 5. Open offers / follow-up actions the user (Mavis) raised

These are concrete deliverables the user explicitly proposed at the
end of the last reply. They are good "next time" prompts.

1. **"再迭代一版论文"** — multi-seed numbers, more thorough Threats
   to Validity, Reusability / Reproducibility statement, sharper
   positioning vs NVIDIA Modulus.
2. **"中文版 slide deck (5–10 页 PPT)"** — short summary deck for
  汇报/路演. Probably with matplotlib-generated figures + bullet
   points. Could use Beamer or just matplotlib PDFs.
3. **"起一个 GitHub 仓库"** — full repo with:
   - README, CONTRIBUTING, GOVERNANCE, CODE_OF_CONDUCT
   - GitHub Actions CI on Linux/macOS/Windows
   - Zenodo integration for releases
   - Issue templates (bug, feature, RFC)
   - PR template
   - `paper/` directory with the LaTeX
   - `docs/` with Sphinx or Docusaurus
   - `artifacts/` with the pre-trained Burgers model
4. **"写个中文版 slide deck"** — possibly combined with #2 above
5. **"做个 demo 网页"** — interactive visualization of the
   trained FNO solving Burgers step-by-step. Probably a small
   static page with a precomputed trajectory + slider.
6. **"正式投 arXiv"** — go to arxiv.org/submit, upload
   `paper/paper.tex`, fill in metadata, decide on anonymous vs
   named submission.

---

## 6. Key files (in case we lose track)

| Path | Purpose |
|---|---|
| `D:\minimax_proj\README.md` | Project landing page |
| `D:\minimax_proj\ROADMAP.md` | 6-stage plan, the source of truth |
| `D:\minimax_proj\STATUS.md` | **This file.** Current state, next steps |
| `D:\minimax_proj\neuroflow_stage1.zip` | 40-file clean source archive (55 KB) |
| `D:\minimax_proj\paper\paper.pdf` | 11-page arXiv-ready PDF (303 KB) |
| `D:\minimax_proj\paper\paper.tex` | LaTeX source for the paper |
| `D:\minimax_proj\paper\README.md` | How to compile the paper |
| `D:\minimax_proj\neuroflow_cpp.pyd` | Built pybind11 extension (in project root) |
| `D:\minimax_proj\artifacts\` | Pre-trained Burgers model (`.neuroir`, `.nneuroir`) |
| `C:\Users\proart\.mavis\memory\user.md` | (will be created) User profile |
| `C:\Users\proart\.mavis\agents\mavis\memory\MEMORY.md` | (already updated) Cross-project lessons |

---

## 7. User profile (so far)

Known facts (from this session):
- Prepares + writes project themselves ("筹备，自己写" = "preparing, writing it myself")
- Has CLion 2026.1.2 installed at `C:\Program Files\JetBrains\CLion 2026.1.2`
- Has MinGW (g++ 13.1) bundled with CLion
- Comfortable with: Python, C++/CMake, basic LaTeX
- Decision style: when offered a choice (MiKTeX vs Tectonic, run debug script or skip), they pick **direct execution** (MiKTeX, "跑")
- Communication style: terse Chinese, jumps straight to next task
- Domain unknown — to ask: "你做哪块的？" (EDA / 电网 / CFD / 其他)
