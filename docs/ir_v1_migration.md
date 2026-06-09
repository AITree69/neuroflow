# NeuroIR v1 (v0.2.0) migration guide

This document explains the v0.1.0 → v0.2.0 changes for users of the
`.neuroir` / `.nneuroir` file format, the Python `neuroflow.ir` API, and
the C++ runtime. It is the formal compatibility statement for Stage 2
Sprint 1.

## TL;DR

- New op: **FNO2d** (`op_code = 0x02`).
- JSON `version` is now `"0.2.0"`. New config fields: `modes_h`, `modes_w`.
- Native binary `version` field is now `2` (was `1`).
- A v0.1.0 file is still readable by a v0.2.0 reader.
- A v0.2.0 file is **rejected** by a v0.1.0 reader (op_code `0x02` is
  unknown, magic and `version` differ).

## Format changes

### JSON (`.neuroir`)

```jsonc
// v0.1.0
{
  "version": "0.1.0",
  "op": "FNO1d",
  "config": {
    "in_channels": 1, "out_channels": 1, "width": 64,
    "modes": 16, "n_layers": 4,
    "activation": "gelu", "pad_factor": 1, "name": "fno1d_burgers"
  },
  "weights": { ... }
}

// v0.2.0
{
  "version": "0.2.0",
  "op": "FNO1d" | "FNO2d",     // <-- new: "FNO2d"
  "config": {
    // for FNO2d, the config differs:
    "in_channels": 1, "out_channels": 1, "width": 24,
    "modes_h": 8, "modes_w": 8, "n_layers": 4,    // <-- new
    "activation": "gelu", "pad_factor": 1, "name": "fno2d_poisson"
  },
  "weights": { ... }
}
```

Key points:

- `op` is the dispatch key. `FNO1d` and `FNO2d` are the only two values
  in v0.2.x.
- FNO1d config is unchanged: `modes` (singular) is the 1D truncation
  count.
- FNO2d config replaces `modes` with `modes_h` and `modes_w` — the
  rectangle of low frequencies in the 2D rfft spectrum.
- All other fields (`width`, `n_layers`, `activation`, `pad_factor`,
  `name`) are shared.

### Native binary (`.nneuroir`)

```
// Layout, little-endian. Same length as v0.1.0 (4 + 2 + 1 + 1 + 28 + 1 + 3 + 4 = 44 bytes
// header + weights).

magic      : 4 bytes   = "NIR0"
version    : uint16    = 1 | 2                          // <-- v0.2.0 writes 2
op_code    : uint8     = 0x01 (FNO1d) | 0x02 (FNO2d)    // <-- new
reserved   : uint8     = 0
config     : 7 int32s — meaning depends on op_code:     // <-- new interpretation
              FNO1d: [in_ch, out_ch, width, modes,       n_layers, pad_factor, _]
              FNO2d: [in_ch, out_ch, width, modes_h, modes_w, n_layers, pad_factor]
activation : uint8     = 0 (gelu) | 1 (relu)
reserved2  : 3 bytes
n_weights  : uint32
weights... : same as v0.1.0 (per-weight: name_len, name, ndim, dims[], data)
```

The 7-int32 config block length is unchanged from v0.1.0. `op_code`
selects the per-slot meaning. The `specs.{i}.weights_real` /
`weights_imag` tensors for FNO2d have `ndim = 4` (was `3` for FNO1d);
the rest of the weight layout is identical.

### Compatibility matrix

| Writer \ Reader | v0.1.0 reader (FNO1d only) | v0.2.0 reader (FNO1d + FNO2d) |
|---|---|---|
| **v0.1.0 FNO1d file** | OK | OK |
| **v0.2.0 FNO1d file** | Reads (binary `version=2` is rejected by old reader — **regression**) | OK |
| **v0.2.0 FNO2d file** | Rejected (`op_code=0x02` unknown) | OK |

The v0.1.0 writer still produces `version=1` files (it has not been
upgraded), and those remain the only files a v0.1.0 reader accepts.
The v0.2.0 writer writes `version=2` and the new op_code.

A v0.1.0 reader that sees a v0.2.0 FNO1d file (version=2) fails with
`ParseError("unsupported IR binary version")`. To roll out the new
op_code without breaking v0.1.0 reader FNO1d usage, the practical
deployment plan is:

1. Push the v0.2.0 runtime everywhere.
2. Re-export any pre-existing FNO1d checkpoints with the v0.2.0
   exporter so their `version=2`. The math is bit-identical — only
   the header changes.

There is no Python-side `v0.1.0 FNO1d file → v0.2.0 file` migration
script yet; one-line change via:

```python
import torch
from neuroflow.ir.export import export_all
from neuroflow.ir.load import load_neuroir
model = load_neuroir("old.fno1d_burgers.neuroir")
export_all(model, ".", basename="fno1d_burgers")
```

## API changes

### Python (`neuroflow.ir`)

- `neuroflow.ir.NEUROIR_VERSION` is now `"0.2.0"`.
- `neuroflow.ir.spec.SUPPORTED_OPS = {"FNO1d", "FNO2d"}`.
- `neuroflow.ir.spec.from_json` accepts both v0.1.0 and v0.2.0
  versions; it uses the version-appropriate `allowed_ops` table.
- `neuroflow.ir.export.export_to_neuroir` dispatches by model type:
  `FNO1d` writes the v0.1.0 config, `FNO2d` writes the v0.2.0 config
  with `modes_h` / `modes_w`.
- `neuroflow.ir.export.export_to_binary` writes `version=2`.
- `neuroflow.ir.load.load_neuroir` returns either an `FNO1d` or an
  `FNO2d`, depending on `spec.op`.
- `neuroflow.ir.load.predict_with_spec` has an `FNO2d` branch.
- New: `neuroflow.nn.fno2d.FNO2d` / `FNO2dConfig` / `SpectralConv2d`.

### C++ (`nflow::`)

- New: `nflow::fno::FNO2d` / `FNO2dConfig` / `FNO2dWeights` /
  `SpectralConv2d` (header-only `cpp/include/neuroflow/fno.h`).
- New: `nflow::fft::Rfft2` / `Irfft2`
  (`cpp/include/neuroflow/fft.h`).
- `nflow::LoadedModel` gains `fno2d_cfg` / `fno2d_weights` fields.
- `nflow::ir_native::LoadBinary` accepts `version=1 | 2` and dispatches
  on `op_code` (0x01 / 0x02).
- `nflow::InferenceRuntime::Run` accepts both 3D (FNO1d) and 4D
  (FNO2d) input/output shapes.
- The CLI `nflow_infer` now dispatches by `op`: 3D input for FNO1d,
  4D for FNO2d.
- The pybind11 binding (`infer`, `infer_arrays`) auto-detects by
  loading the IR and inspecting `op`.

## Test parity

| Path | Stage 1 baseline | Stage 2 Sprint 1 |
|---|---|---|
| C++ vs PyTorch, FNO1d (Burgers 1D, 256-grid, 64-mode) | **5.20e-05** max abs | (unchanged) |
| C++ vs PyTorch, FNO2d (Poisson 32x32, modes 8x8) | n/a | **2.19e-06** max abs |

Both numbers are dominated by float32 summation-order noise from
PyTorch's BLAS GEMM. They are documented in the paper and the
CHANGELOG.
