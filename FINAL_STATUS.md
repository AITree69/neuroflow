# NeuroFlow — Stage 2 收官总结

> **最后更新**: 2026-06-08
> **当前 session**: `mvs_ce833c5b5d5b4364a64661e05fd1cc`
> **最后 commit**: `e18b088` (Sprint 3.12 calibration refinement)
> **当前版本**: v0.18.0 (NeuroIR)
> **状态**: **Stage 2 完整 closed, 准备进入 Stage 3**

---

## 1. 一句话总结

**NeuroFlow Stage 2** = **6 个 operator families + 4 个 cross-op 子实验 + 1 个
domain SDK + 3 个 INT8 量化方案 + 1 个 calibration refinement**,全部端到端
Python + C++/pybind11,C++ vs PyTorch 在 FP32 下 < 1e-3,在 INT8 下符合
per-tensor 量化噪声基线。

---

## 2. 关键数字

| 指标 | 值 |
|---|---|
| Git commits | 24 |
| Op families | 6 (FNO1d/2d/3d, DeepONet, TokenMixer, GraphOp + 2D versions) |
| Paper pages | 19 (716 KB) |
| pytest pass | **77/77** |
| NeuroIR version | v0.18.0 |
| FP32 parity (6 ops) | C++ vs PyTorch < 1e-5 (1D) / < 1e-3 (general) |
| INT8 (per-tensor) | val rel L2 = 5.5e-1 |
| INT8 (per-tensor p99.5 + 80 calib) | val rel L2 = 4.8e-1 (14% better) |
| INT8 (per-token) | val rel L2 = 7.5e-1 with calib refinement (still worse than per-tensor) |
| On-disk weight storage | 25% of FP32 (2130 KB → 532 KB) |
| Domain SDK | NeuroFlow × LAMMPS (Morse potential surrogate) |

---

## 3. Stage 2 完整体系结构

### 3.1 Operator families (6)
- **FNO1d/2d/3d** — spectral convolution
- **DeepONet** — branch + trunk
- **TokenMixer (Transolver-style)** — multi-head self-attention over mean-pooled patches
- **GraphOp (GCN-style)** — degree-normalised neighbour aggregation
- **TokenMixer2D / GraphOp2D** — 2D regular-grid versions of the above

### 3.2 Cross-op sub-sections in paper (4)
- **§5.7** — single-seed Burgers 1D benchmark
- **§5.8** — multi-seed 5×3 robustness
- **§5.9** — 2D Poisson 16×16 cross-op benchmark
- **§5.10** — 2D C++ parity + 21-config grid sweep

### 3.3 First domain SDK (§5.10.4)
- `domains/lammps/` — NeuroFlow × LAMMPS
- 1D Morse potential surrogate (`MorseSurrogate` + `surrogate_md_loop`)
- SDK is a **contract**, not a LAMMPS plugin. Stage 3 first item:
  real LAMMPS `fix nflow` shim.

### 3.4 INT8 quant pipeline (4 sprints)
- **§5.11 / v0.15.0** — W8A8 fake-quant (per-tensor)
- **§5.12 / v0.16.0** — per-channel weight quant
- **§5.13 / v0.17.0** — per-token activation quant
- **§5.14 / v0.18.0** — calibration refinement (percentile + EMA)

### 3.5 IR extensibility
- NeuroIR trailing **NIRQ** block supports any mix of:
  - per-tensor qparams (kind=0)
  - per-channel qparams (kind=1)
  - per-token qparams (kind=2)
- Backward-compat: v0.13.0 readers ignore trailing NIRQ block;
  v0.15.0+ readers detect by the "NIRQ" magic.

---

## 4. 关键路径文件(未来续 session 必读)

| 路径 | 用途 |
|---|---|
| `D:\minimax_proj\STATUS.md` | 完整 stage-by-stage 表格,每行一个 sprint 决策 + 数字 |
| `D:\minimax_proj\CHANGELOG.md` | semver-by-sprint changelog, v0.1.0 → v0.18.0 |
| `D:\minimax_proj\neuroflow\ir\export.py` docstring | NeuroIR binary layout 全版本(读这段理解 IR 演进) |
| `D:\minimax_proj\paper\paper.pdf` (19 pages, 716 KB) | 整体叙事的 arXiv-ready 草稿 |
| `D:\minimax_proj\neuroflow\quant\static_quant.py` | INT8 PTQ 实现 + zp saturation fix + percentile + EMA |
| `D:\minimax_proj\cpp\src\fno.cpp` + `cpp/include/neuroflow/fno.h` | C++ v0.18.0 fake-quant 集成 (per-tensor/per-channel/per-token 在 Linear output) |
| `D:\minimax_proj\cpp\include\neuroflow\ir_loader.h` | NIRQ trailing block 解析 |
| `D:\minimax_proj\cpp\include\neuroflow\quant_types.h` | QuantParams / PerChannelQuantParams / PerTokenQuantParams struct 定义 |
| `D:\minimax_proj\examples\18_fno1d_int8.py` | INT8 PTQ 端到端 demo (--per-channel, --per-token, --percentile, --ema-decay 四个 flag) |
| `C:\Users\proart\.mavis\agents\mavis\memory\MEMORY.md` | 7 条技术 lesson (C++/MinGW, .npy, GELU, Windows App Aliases, pybind11, einsum, mavis-team) |

---

## 5. Stage 2 关键发现(诚实)

1. **Per-channel weight quant alone does not close the per-tensor
   accuracy floor** on FNO1d / Burgers 1D (both 5.5e-1).  The
   dominant noise is per-tensor *activation* quant, not per-tensor
   weight quant.

2. **Per-token activation quant** (Sprint 3.11) **is WORSE than
   per-tensor** with 8 calibration samples (9.7e-1 vs 5.5e-1).
   Per-point activation range is too narrow → INT8 grid too
   coarse → test samples clip ±128 → large error.

3. **Calibration refinement** (Sprint 3.12 — percentile + EMA + 80
   calib) gives a **14% improvement on per-tensor** (5.5e-1 → 4.8e-1)
   and **23% improvement on per-token** (9.7e-1 → 7.5e-1).  The
   mechanism is correct and useful.

4. **Per-token still does not close the floor** on FNO1d — this
   is a model property (small activation range per spatial
   point), not a calibration issue. Stage 3 fix: QAT or FP8.

---

## 6. Stage 3 候选(按优先级)

| Sprint | 内容 | Scope |
|---|---|---|
| 3.13 | Real LAMMPS `fix nflow` shim | 大 (需要 LAMMPS 源码) |
| 3.14 | Real INT8 GEMM with INT32 accumulation | 中 (真正实现 INT8 GEMM) |
| 3.15 | FP8 (E4M3/E5M2) demo | 中 (下代量化方案) |
| 3.16 | QAT (quantisation-aware training) for per-token | 中 (closes per-token floor) |

### 6.1 Open offers (user hasn't acted on)
1. Chinese slide deck (5-10 pages) for 汇报/路演
2. Set up the GitHub repo (network blocked to github.com:443)
3. Interactive web demo of the trained FNO
4. Formally submit to arXiv
5. More paper iteration (positioning vs NVIDIA Modulus, etc.)

---

## 7. 用法 (重新跑 demo)

```powershell
# FP32 demo (Stage 1, ~30s)
& "D:\minimax_proj\.venv\Scripts\python.exe" examples/01_train_fno1d.py

# 2D C++ parity (Sprint 3.6)
& "D:\minimax_proj\.venv\Scripts\python.exe" examples/04_export_and_infer_fno1d.py
& D:\minimax_proj\cpp\build\test_runtime.exe

# INT8 PTQ (Sprint 3.9)
& "D:\minimax_proj\.venv\Scripts\python.exe" examples/18_fno1d_int8.py

# INT8 PTQ with per-token + percentile
& "D:\minimax_proj\.venv\Scripts\python.exe" examples/18_fno1d_int8.py `
    --per-token --n-calib 80 --percentile 99.5

# LAMMPS SDK Morse surrogate (Sprint 3.8)
& "D:\minimax_proj\.venv\Scripts\python.exe" `
    examples/17_morse_surrogate.py
```

---

## 8. 复现 (从零开始)

```powershell
# 1. Clone
cd D:\minimax_proj

# 2. Python deps
& D:\minimax_proj\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Build C++
$env:PATH = "C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin;" +
            "C:\Program Files\JetBrains\CLion 2026.1.2\bin\cmake\win\x64\bin;" +
            $env:PATH
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target nflow_core neuroflow_cpp test_runtime -j 4

# 4. Copy .pyd + MinGW DLLs
Copy-Item build\neuroflow_cpp.cp312-win_amd64.pyd D:\minimax_proj\
Copy-Item "C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin\*.dll" D:\minimax_proj\

# 5. Run all tests
& D:\minimax_proj\.venv\Scripts\python.exe -m pytest tests/ domains/lammps/tests/ -q

# 6. Recompile paper
& "C:\Users\proart\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" `
    -interaction=nonstopmode -output-directory D:\minimax_proj\paper `
    D:\minimax_proj\paper\paper.tex
# (run twice for cross-refs)
```

---

**Status**: ✅ Stage 2 完整 closed.  Awaiting user decision on Stage 3 next-up.
