# NeuroFlow — Architecture Overview

```
+---------------------------------------------------------------+
|                         Python (PyTorch)                       |
|                                                                |
|  neuroflow.nn.fno         FNO1d, FNO2d, FNO3d (models)         |
|  neuroflow.nn.deeponet    DeepONet  (Stage 2)                  |
|  neuroflow.ir.spec        NeuroIRSpec dataclass (JSON view)    |
|  neuroflow.ir.export      PyTorch -> NeuroIR                   |
|  neuroflow.ir.load        NeuroIR -> inference                 |
|                                                                |
|  predict_with_spec():                                        |
|     if neuroflow_cpp available -> C++ runtime                  |
|     else                       -> torch fallback               |
+----------------------------+----------------------------------+
                             |
                             |   NeuroIR v0 (.neuroir, .nneuroir)
                             v
+---------------------------------------------------------------+
|                       C++ Runtime (zero-dep)                   |
|                                                                |
|  nflow::ir::Spec                  weight / config blob         |
|  nflow::ops::SpectralConv1d       FNO1d spectral convolution    |
|  nflow::ops::Linear               affine transform              |
|  nflow::runtime::Inference        end-to-end forward            |
|                                                                |
|  Optional: pybind11 binding (neuroflow_cpp .pyd/.so)           |
+---------------------------------------------------------------+
                             |
                             v
                  Traditional numerics
                  (FEM/FVM/OpenFOAM/CFD/heat)
                  [Stage 3 integration]
```

## Design rules (Stage 1)

1. **NeuroIR v0 is a frozen wire format.** Any change that breaks v0
   triggers a v1 bump and goes through the [RFC process](../CONTRIBUTING.md).
2. **C++ runtime has zero third-party deps.** The runtime core links only
   the standard library. Pybind11 is an *opt-in* binding, gated behind a
   CMake option.
3. **The C++ runtime is correctness infrastructure, not performance
   infrastructure (Stage 1).** It is intentionally ~0.6× the speed of the
   PyTorch reference. The real speedup arrives in Stage 4 (CUDA + cuFFT +
   batched inference). This is stated explicitly in the paper.
4. **Python is the source of truth for training.** C++ is a *deployable
   inference* runtime; weights flow PyTorch -> NeuroIR -> C++.
5. **Numerical floor is 5.20e-05** for a 256-grid, 64-mode FNO1d. This is
   float32 summation-order noise from PyTorch's BLAS GEMM vs. our unrolled
   loops. Documented; not a bug.
