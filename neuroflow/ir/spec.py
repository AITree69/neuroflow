"""NeuroIR v0 spec dataclasses."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

NEUROIR_VERSION = "0.21.0"

# Op set supported in v0.20.x.
#   - FNO1d: v0.1.0+
#   - FNO2d: v0.2.0+
#   - FNO3d: v0.3.0 (Stage 2 Sprint 2)
#   - DeepONet: v0.4.0 (Stage 2 Sprint 2)
#   - TokenMixer (Transolver-style): v0.5.0 (Stage 2 Sprint 3)
#   - GraphOp (GCN-style): v0.6.0 (Stage 2 Sprint 3)
#   - TokenMixer2D: v0.12.0 (Stage 2 Sprint 3.6, 2D version)
#   - GraphOp2D:    v0.12.0 (Stage 2 Sprint 3.6, 2D version)
#   - LAMMPS domain SDK: v0.13.0 (Stage 2 Sprint 3.8)
#   - INT8 (W8A8 fake-quant) quant block: v0.15.0 (Sprint 3.9)
#   - Per-channel weight quant: v0.16.0 (Sprint 3.10)
#   - Per-token activation quant: v0.17.0 (Sprint 3.11)
#   - Calibration refinement (percentile + EMA): v0.18.0 (Sprint 3.12)
#   - LAMMPS `fix nflow` shim (Stage 3 Sprint 3.13)
#   - Real INT8 GEMM with INT32 accumulation (Stage 3 Sprint 3.14)
SUPPORTED_OPS = frozenset({
    "FNO1d", "FNO2d", "FNO3d", "DeepONet",
    "TokenMixer", "GraphOp",
    "TokenMixer2D", "GraphOp2D",
})

# Op set supported by older format versions (for backward compat reads).
_LEGACY_OPS = {
    "0.1.0": frozenset({"FNO1d"}),
    "0.2.0": frozenset({"FNO1d", "FNO2d"}),
    "0.3.0": frozenset({"FNO1d", "FNO2d", "FNO3d"}),
    "0.4.0": frozenset({"FNO1d", "FNO2d", "FNO3d", "DeepONet"}),
    "0.5.0": frozenset({"FNO1d", "FNO2d", "FNO3d", "DeepONet", "TokenMixer"}),
    "0.6.0": frozenset({"FNO1d", "FNO2d", "FNO3d", "DeepONet", "TokenMixer", "GraphOp"}),
    "0.12.0": frozenset({
        "FNO1d", "FNO2d", "FNO3d", "DeepONet",
        "TokenMixer", "GraphOp", "TokenMixer2D", "GraphOp2D",
    }),
}


@dataclass
class TensorEntry:
    name: str
    shape: list[int]
    dtype: str  # "float32" for v0
    data_b64: str

    def to_numpy(self) -> np.ndarray:
        if self.dtype != "float32":
            raise NotImplementedError(f"v0 only supports float32, got {self.dtype}")
        raw = base64.b64decode(self.data_b64)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(self.shape)
        return arr

    def to_torch(self) -> torch.Tensor:
        # `np.frombuffer` returns a read-only view. Copy to a writable array
        # before handing it to PyTorch (avoids UserWarning about non-writable
        # tensors and undefined behavior on writes).
        arr = self.to_numpy().copy()
        return torch.from_numpy(arr)


@dataclass
class NeuroIRSpec:
    version: str = NEUROIR_VERSION
    op: str = "FNO1d"
    config: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, TensorEntry] = field(default_factory=dict)
    # Optional INT8 (W8A8 fake-quant) block — see export.py
    # docstring for the binary layout.  When None / empty,
    # the binary export is bit-for-bit identical to the
    # pre-v0.15.0 format and older readers ignore the
    # trailing bytes.
    quant: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "op": self.op,
                "config": self.config,
                "weights": {
                    name: {
                        "shape": list(t.shape),
                        "dtype": t.dtype,
                        "data_b64": t.data_b64,
                    }
                    for name, t in self.weights.items()
                },
            },
            indent=2,
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> "NeuroIRSpec":
        obj = json.loads(text)
        version = obj.get("version")
        if version is None:
            raise ValueError("NeuroIR file missing 'version' field")
        # Accept the current version and any older versions we still know how
        # to read. Forward compat (newer than runtime) is rejected to avoid
        # silently dropping fields.
        if version == NEUROIR_VERSION:
            allowed_ops = SUPPORTED_OPS
        elif version in _LEGACY_OPS:
            allowed_ops = _LEGACY_OPS[version]
        else:
            raise ValueError(
                f"NeuroIR version mismatch: file {version!r} "
                f"vs runtime {NEUROIR_VERSION!r}"
            )
        if obj["op"] not in allowed_ops:
            raise ValueError(
                f"unsupported op {obj['op']!r} for NeuroIR version {version!r} "
                f"(allowed: {sorted(allowed_ops)})"
            )
        weights = {
            name: TensorEntry(
                name=name,
                shape=info["shape"],
                dtype=info["dtype"],
                data_b64=info["data_b64"],
            )
            for name, info in obj["weights"].items()
        }
        return cls(
            version=version,
            op=obj["op"],
            config=obj["config"],
            weights=weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NeuroIRSpec":
        return cls.from_json(Path(path).read_text())
