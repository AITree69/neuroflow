"""NeuroIR — NeuroFlow Intermediate Representation.

Stage 1 (v0.1.0): JSON-based format, FNO1d only.
Stage 2 (v0.2.0): adds FNO2d support (Sprint 1).
Stage 2 (v0.3.0): adds FNO3d support (Sprint 2). FNO1d / FNO2d
  on-disk layout preserved for backward compatibility.
Stage 2 (v0.4.0): adds DeepONet support (Sprint 2). FNO1d / FNO2d /
  FNO3d on-disk layouts preserved.

Layout (NeuroIR v0.4.0):
    {
        "version": "0.4.0",
        "op": "FNO1d" | "FNO2d" | "FNO3d" | "DeepONet",
        "config": { ... FNOXxxConfig ... or DeepONetConfig ... },
        "weights": {
            "<op-specific weight names>": {"shape": [...], "dtype": "float32", "data_b64": "..."},
            ...
        }
    }
"""

from neuroflow.ir.export import export_to_neuroir
from neuroflow.ir.load import load_neuroir
from neuroflow.ir.spec import NEUROIR_VERSION, NeuroIRSpec

__all__ = ["NEUROIR_VERSION", "NeuroIRSpec", "export_to_neuroir", "load_neuroir"]
