"""Export a PyTorch neural operator (FNO1d / FNO2d / FNO3d / DeepONet /
TokenMixer / GraphOp / TokenMixer2D / GraphOp2D) to NeuroIR spec.

Two formats are produced:
    - .neuroir : JSON, human-readable, single source of truth for tooling
    - .nneuroir: binary, fast to parse, used by the C++ runtime (no deps)

NeuroIR native binary layout (little-endian), versions 1 / 2 / 3:
    magic      : 4 bytes   = "NIR0"
    version    : uint16    = 1 | 2 | 3
    op_code    : uint8     = 0x01 (FNO1d) | 0x02 (FNO2d) | 0x03 (FNO3d) |
                            0x04 (DeepONet) | 0x05 (TokenMixer) |
                            0x06 (GraphOp) | 0x07 (TokenMixer2D) |
                            0x08 (GraphOp2D)
    reserved   : uint8     = 0
    config     : int32s   — length depends on op_code:
                  FNO1d / FNO2d / DeepONet:                   7 int32s
                  FNO3d / TokenMixer / GraphOp /
                  TokenMixer2D / GraphOp2D:                    8 int32s
                  Meaning by op_code:
                    FNO1d:       [in_ch, out_ch, width, modes,       n_layers, pad_factor, _]
                    FNO2d:       [in_ch, out_ch, width, modes_h, modes_w, n_layers, pad_factor]
                    FNO3d:       [in_ch, out_ch, width, modes_h, modes_w, modes_d, n_layers, pad_factor]
                    DeepONet:    [in_branch, in_trunk, latent_dim, out_channels,
                                  n_layers_branch, n_layers_trunk, _]
                    TokenMixer:  [in_dim, out_dim, latent_dim, n_points,
                                  n_patches, n_heads, n_layers, _]
                    GraphOp:     [in_dim, out_dim, n_nodes, hidden_dim, n_layers, _, _, _]
                    TokenMixer2D:[in_dim, out_dim, latent_dim, h, w,
                                  n_patches, n_heads, n_layers]
                    GraphOp2D:   [in_dim, out_dim, h, w, hidden_dim, n_layers, _, _]
    activation : uint8     = 0 (gelu) | 1 (relu)  (DeepONet / TokenMixer /
                  GraphOp / TokenMixer2D / GraphOp2D use this byte too,
                  for symmetry; the value is written from cfg.activation
                  when present, else defaults to gelu=0)
    reserved2  : 3 bytes
    n_weights  : uint32
    for each weight:
        name_len : uint8
        name     : name_len bytes (utf-8, no null terminator)
        ndim     : uint8
        dims[]   : int32 * ndim
        data     : float32 * numel

Versioning:
    - v0.1.0 (binary version=1): FNO1d only.
    - v0.2.0 (binary version=2): FNO1d + FNO2d. FNO1d layout preserved.
    - v0.3.0 (binary version=3): adds FNO3d (8 int32). FNO1d / FNO2d
      still write version=2 + 7 int32 for backward compat.
    - v0.4.0 (binary version=3): adds DeepONet (op_code=0x04, 7 int32).
      FNO1d / FNO2d / FNO3d layouts preserved.
    - v0.5.0 (binary version=3): adds TokenMixer / Transolver-style
      operator (op_code=0x05, 8 int32).
    - v0.6.0 (binary version=3): adds GraphOp / GCN-style operator
      (op_code=0x06, 8 int32).  The 3 extra "graph.*" entries
      (`graph.adj_offsets`, `graph.adj_indices`, `graph.deg_inv`) are
      stored as float32 weights for IR-loader uniformity; the C++
      loader casts them back to int32 / float32 at use time.
    - v0.12.0 (binary version=3): adds TokenMixer2D (op_code=0x07, 8
      int32) and GraphOp2D (op_code=0x08, 8 int32).  These are
      2D regular-grid versions of the corresponding 1D ops; the
      C++ forward flattens the (h, w) grid into a hw sequence
      and reuses the 1D mechanism.
    - v0.15.0 (binary version=4): adds an optional INT8
      (W8A8 fake-quant) quantisation block.  When
      `quant_flag == 0` the file is bit-for-bit identical
      to a v0.12.0 file and is rejected by readers that
      only know version=3 (use the v0.15.0 reader to load
      v0.12.0 files).  When `quant_flag == 1` the file
      contains a `quant` section after the last weight,
      holding per-tensor scale / zero_point for every
      weight tensor and a small set of activation
      quantisation parameters.  The C++ runtime applies
      the same fake-quant round-trip the Python reference
      does (W8A8 with FP32 compute and INT8 storage for
      weights).
    - v0.16.0 (binary version=4): extends the NIRQ
      block with a `kind` byte per qparam entry.  When
      `kind == 0` the qparam is per-tensor (the v0.15.0
      format).  When `kind == 1` the qparam is
      per-channel and the entry is followed by
      `n_channels + channel_axis + scales[n_channels] +
      zero_points[n_channels]`.  Per-channel qparams are
      for weight tensors of `nn.Linear`
      (`channel_axis=0`) and `SpectralConv1d`
      (`channel_axis=1`).
    - v0.17.0 (binary version=4): extends the NIRQ
      block with a `kind=2` (per-token) entry for
      activation qparams.  The entry is followed by
      `n_tokens + width + scales[n_tokens] +
      zero_points[n_tokens]`.  Per-token qparams are
      for activation tensors of shape `(batch, n, width)`
      (e.g. the post-`Linear` output of every FNO1d
      layer); each spatial point `(n_idx, w_idx)` gets
      its own `(scale, zero_point)`.
"""

# Bump to v0.15.0 — adds the optional INT8 quantisation
# block at the end of the binary file.  Backward-compat
# reads: v0.15.0 readers must accept version=1, 2, 3 files
# (no `quant` block present) and skip the block when
# version=4 + quant_flag=0.

from __future__ import annotations

import base64
import struct
from pathlib import Path

import torch

from neuroflow.ir.spec import NeuroIRSpec, TensorEntry
from neuroflow.nn.deeponet import DeepONet
from neuroflow.nn.fno import FNO1d
from neuroflow.nn.fno2d import FNO2d
from neuroflow.nn.fno3d import FNO3d
from neuroflow.nn.graph_op import GraphOp
from neuroflow.nn.graph_op2d import GraphOp2D
from neuroflow.nn.tokenmixer import TokenMixer
from neuroflow.nn.tokenmixer2d import TokenMixer2D

# Op code table — keep in sync with cpp/include/neuroflow/ir_loader.h
_OP_FNO1D = 0x01
_OP_FNO2D = 0x02
_OP_FNO3D = 0x03
_OP_DEEPONET = 0x04
_OP_TOKENMIXER = 0x05
_OP_GRAPHOP = 0x06
_OP_TOKENMIXER2D = 0x07
_OP_GRAPHOP2D = 0x08
_OP_NAME_TO_CODE = {
    "FNO1d": _OP_FNO1D,
    "FNO2d": _OP_FNO2D,
    "FNO3d": _OP_FNO3D,
    "DeepONet": _OP_DEEPONET,
    "TokenMixer": _OP_TOKENMIXER,
    "GraphOp": _OP_GRAPHOP,
    "TokenMixer2D": _OP_TOKENMIXER2D,
    "GraphOp2D": _OP_GRAPHOP2D,
}

# Activation code table
_ACT_GELU = 0
_ACT_RELU = 1
_ACT_NAME_TO_CODE = {"gelu": _ACT_GELU, "relu": _ACT_RELU}
_ACT_CODE_TO_NAME = {_ACT_GELU: "gelu", _ACT_RELU: "relu"}

# ----------------------------------------------------------------------------
# Sprint 3.27: PyTorch -> NeuroIR codegen.
# ----------------------------------------------------------------------------
# Before Sprint 3.27, each op family had a
# hand-written branch in `export_to_neuroir` (the
# spec-dict builder) AND a hand-written branch in
# `export_to_binary` (the int32 config-block
# packer).  Adding a new op (e.g.\ a new FNO
# dimension, a new transformer-style op) required
# two coordinated edits in two functions.  This
# Sprint consolidates the per-op metadata into a
# single `OpSpec` dataclass and rewrites the two
# exporters to dispatch through a single
# `_codegen` helper.  Net effect:
#   - OpSpec is the single source of truth for
#     the op -> int32-config-block mapping.
#   - New ops are added by writing one OpSpec
#     entry, not by editing two functions.
#   - export_to_neuroir becomes ~30 lines
#     instead of ~140 lines of branching.
#   - export_to_binary becomes ~15 lines
#     instead of ~100 lines of branching.
# ----------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class OpSpec:
    """Codegen metadata for a single NeuroFlow op.

    Fields
    ------
    name:        str — the spec op name (e.g. "FNO1d")
    op_code:     int — the .nneuroir op code (e.g. 0x01)
    version:     int — the .nneuroir native binary version
    config_len:  int — number of int32 fields in the
                 config block (7 for the FNO family /
                 DeepONet, 8 for the rest).
    cfg_keys:    tuple of str — ordered list of
                 config-dict keys to read.  Each
                 key is mapped to its int32 slot
                 in `int32_index` (0-indexed).
    int32_index: dict[str, int] — maps each
                 cfg_key to its position in the
                 int32 config block.
    defaults:    dict[str, int] — defaults for
                 optional cfg_keys (e.g. pad_factor
                 defaults to 1).
    cfg_extras:  tuple of str — extra cfg-dict
                 keys that don't go into the int32
                 block (e.g. "name", "activation").
    model_class: type — the PyTorch model class
                 this op maps to (used by
                 export_to_neuroir to dispatch by
                 isinstance).

    The schema is constructed by hand for each
    op; the codegen helpers below turn it into
    the spec dict and the int32 block.
    """
    name: str
    op_code: int
    version: int
    config_len: int
    cfg_keys: tuple
    int32_index: dict
    defaults: dict = field(default_factory=dict)
    cfg_extras: tuple = ("name", "activation")
    model_class: type = None  # set lazily after class defs


# Per-op schemas.  Each tuple is
# (cfg_key, int32_position) — the position is
# 0-indexed in the int32 config block; -1 means
# "not packed into the block" (cfg-only key).
#
# The schema is the single source of truth for
# both the spec dict builder and the binary
# packer.  If you change one, change the other
# is automatic.
_OP_SCHEMAS: dict[str, OpSpec] = {}


# A small helper to build an OpSpec from a list
# of (key, position) pairs.
def _build_spec(name: str, op_code: int, version: int,
                config_len: int, key_positions: list,
                defaults: Optional[dict] = None,
                cfg_extras: tuple = ("name", "activation")
                ) -> OpSpec:
    cfg_keys = tuple(k for k, _ in key_positions)
    int32_index = {k: pos for k, pos in key_positions
                    if pos >= 0}
    return OpSpec(
        name=name, op_code=op_code, version=version,
        config_len=config_len, cfg_keys=cfg_keys,
        int32_index=int32_index, defaults=defaults or {},
        cfg_extras=cfg_extras)


# Per-op key -> int32 position tables.
# The "0" in FNO1d position 6 (the trailing zero)
# is the v0.1.0 padding byte.  v0.1.0 readers
# expect exactly 7 int32 fields even if the
# "real" config is only 6 values.
_FNO1D_KEYS = [
    ("in_channels", 0), ("out_channels", 1),
    ("width", 2), ("modes", 3),
    ("n_layers", 4), ("pad_factor", 5),
    ("_pad", 6),
]
_FNO2D_KEYS = [
    ("in_channels", 0), ("out_channels", 1),
    ("width", 2), ("modes_h", 3), ("modes_w", 4),
    ("n_layers", 5), ("pad_factor", 6),
]
_FNO3D_KEYS = [
    ("in_channels", 0), ("out_channels", 1),
    ("width", 2), ("modes_h", 3), ("modes_w", 4),
    ("modes_d", 5), ("n_layers", 6), ("pad_factor", 7),
]
_DEEPONET_KEYS = [
    ("in_branch", 0), ("in_trunk", 1),
    ("latent_dim", 2), ("out_channels", 3),
    ("n_layers_branch", 4), ("n_layers_trunk", 5),
    ("_pad", 6),
]
_TOKENMIXER_KEYS = [
    ("in_dim", 0), ("out_dim", 1),
    ("latent_dim", 2), ("n_points", 3),
    ("n_patches", 4), ("n_heads", 5),
    ("n_layers", 6), ("_pad", 7),
]
_GRAPHOP_KEYS = [
    ("in_dim", 0), ("out_dim", 1),
    ("n_nodes", 2), ("hidden_dim", 3),
    ("n_layers", 4), ("_pad", 5), ("_pad", 6),
    ("_pad", 7),
]
_TOKENMIXER2D_KEYS = [
    ("in_dim", 0), ("out_dim", 1),
    ("latent_dim", 2), ("h", 3), ("w", 4),
    ("n_patches", 5), ("n_heads", 6),
    ("n_layers", 7),
]
_GRAPHOP2D_KEYS = [
    ("in_dim", 0), ("out_dim", 1),
    ("h", 2), ("w", 3),
    ("hidden_dim", 4), ("n_layers", 5),
    ("_pad", 6), ("_pad", 7),
]


# Native binary version (u16 written into the file). The on-disk layout
# depends on the op:
#   - FNO1d / FNO2d always write version=2 + 7 int32 (backward compatible
#     with v0.1.0 / v0.2.0 readers).
#   - FNO3d / DeepONet / TokenMixer / GraphOp / TokenMixer2D / GraphOp2D
#     write version=3 + 8 int32.
_NATIVE_VERSION_FNO_FAMILY = 2
_NATIVE_VERSION_FNO3D = 3
_NATIVE_VERSION_DEEPONET = 3
_NATIVE_VERSION_TOKENMIXER = 3
_NATIVE_VERSION_GRAPHOP = 3
_NATIVE_VERSION_TOKENMIXER2D = 3
_NATIVE_VERSION_GRAPHOP2D = 3
_NATIVE_CONFIG_LEN_FNO_FAMILY = 7
_NATIVE_CONFIG_LEN_FNO3D = 8
_NATIVE_CONFIG_LEN_DEEPONET = 7
_NATIVE_CONFIG_LEN_TOKENMIXER = 8
_NATIVE_CONFIG_LEN_GRAPHOP = 8
_NATIVE_CONFIG_LEN_TOKENMIXER2D = 8
_NATIVE_CONFIG_LEN_GRAPHOP2D = 8

# ----------------------------------------------------------------------------
# Sprint 3.27: register the per-op OpSpec schemas.
# ----------------------------------------------------------------------------
# Each schema is the single source of truth for
# the op -> spec-dict + int32-block mapping.  See
# the comment block above _OP_SCHEMAS for the
# design rationale.
_OP_SCHEMAS.update({
    "FNO1d": _build_spec(
        name="FNO1d", op_code=_OP_FNO1D,
        version=_NATIVE_VERSION_FNO_FAMILY,
        config_len=_NATIVE_CONFIG_LEN_FNO_FAMILY,
        key_positions=_FNO1D_KEYS,
        defaults={"pad_factor": 1}),
    "FNO2d": _build_spec(
        name="FNO2d", op_code=_OP_FNO2D,
        version=_NATIVE_VERSION_FNO_FAMILY,
        config_len=_NATIVE_CONFIG_LEN_FNO_FAMILY,
        key_positions=_FNO2D_KEYS,
        defaults={"pad_factor": 1}),
    "FNO3d": _build_spec(
        name="FNO3d", op_code=_OP_FNO3D,
        version=_NATIVE_VERSION_FNO3D,
        config_len=_NATIVE_CONFIG_LEN_FNO3D,
        key_positions=_FNO3D_KEYS,
        defaults={"pad_factor": 1}),
    "DeepONet": _build_spec(
        name="DeepONet", op_code=_OP_DEEPONET,
        version=_NATIVE_VERSION_DEEPONET,
        config_len=_NATIVE_CONFIG_LEN_DEEPONET,
        key_positions=_DEEPONET_KEYS,
        defaults={}),
    "TokenMixer": _build_spec(
        name="TokenMixer", op_code=_OP_TOKENMIXER,
        version=_NATIVE_VERSION_TOKENMIXER,
        config_len=_NATIVE_CONFIG_LEN_TOKENMIXER,
        key_positions=_TOKENMIXER_KEYS,
        defaults={}),
    "GraphOp": _build_spec(
        name="GraphOp", op_code=_OP_GRAPHOP,
        version=_NATIVE_VERSION_GRAPHOP,
        config_len=_NATIVE_CONFIG_LEN_GRAPHOP,
        key_positions=_GRAPHOP_KEYS,
        defaults={}),
    "TokenMixer2D": _build_spec(
        name="TokenMixer2D", op_code=_OP_TOKENMIXER2D,
        version=_NATIVE_VERSION_TOKENMIXER2D,
        config_len=_NATIVE_CONFIG_LEN_TOKENMIXER2D,
        key_positions=_TOKENMIXER2D_KEYS,
        defaults={}),
    "GraphOp2D": _build_spec(
        name="GraphOp2D", op_code=_OP_GRAPHOP2D,
        version=_NATIVE_VERSION_GRAPHOP2D,
        config_len=_NATIVE_CONFIG_LEN_GRAPHOP2D,
        key_positions=_GRAPHOP2D_KEYS,
        defaults={}),
})


def get_op_spec(name: str) -> OpSpec:
    """Return the OpSpec for an op name, raising
    ValueError if unknown.  This is the canonical
    lookup; the rest of the codegen dispatches
    through it."""
    spec = _OP_SCHEMAS.get(name)
    if spec is None:
        raise ValueError(
            f"unknown op: {name}; known: "
            f"{list(_OP_SCHEMAS.keys())}")
    return spec


def _model_to_op_name(model) -> str:
    """Return the canonical op name for a PyTorch
    model, by looking up its class in the
    registered OpSpec schemas."""
    for spec in _OP_SCHEMAS.values():
        if spec.model_class is not None \
                and isinstance(model, spec.model_class):
            return spec.name
    raise TypeError(
        f"export_to_neuroir: expected FNO1d, FNO2d, "
        f"FNO3d, DeepONet, TokenMixer, GraphOp, "
        f"TokenMixer2D, or GraphOp2D, got "
        f"{type(model).__name__}")


# Bind the model classes to the schemas.  This
# happens after the schemas are built so the
# forward reference to the imported classes
# resolves at module load time.
_OP_SCHEMAS["FNO1d"] = OpSpec(
    **{**_OP_SCHEMAS["FNO1d"].__dict__,
       "model_class": FNO1d})
_OP_SCHEMAS["FNO2d"] = OpSpec(
    **{**_OP_SCHEMAS["FNO2d"].__dict__,
       "model_class": FNO2d})
_OP_SCHEMAS["FNO3d"] = OpSpec(
    **{**_OP_SCHEMAS["FNO3d"].__dict__,
       "model_class": FNO3d})
_OP_SCHEMAS["DeepONet"] = OpSpec(
    **{**_OP_SCHEMAS["DeepONet"].__dict__,
       "model_class": DeepONet})
_OP_SCHEMAS["TokenMixer"] = OpSpec(
    **{**_OP_SCHEMAS["TokenMixer"].__dict__,
       "model_class": TokenMixer})
_OP_SCHEMAS["GraphOp"] = OpSpec(
    **{**_OP_SCHEMAS["GraphOp"].__dict__,
       "model_class": GraphOp})
_OP_SCHEMAS["TokenMixer2D"] = OpSpec(
    **{**_OP_SCHEMAS["TokenMixer2D"].__dict__,
       "model_class": TokenMixer2D})
_OP_SCHEMAS["GraphOp2D"] = OpSpec(
    **{**_OP_SCHEMAS["GraphOp2D"].__dict__,
       "model_class": GraphOp2D})


def _tensor_to_entry(name: str, t: torch.Tensor) -> TensorEntry:
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    arr = t.detach().cpu().contiguous().numpy().astype("float32")
    raw = arr.tobytes()
    return TensorEntry(
        name=name,
        shape=list(arr.shape),
        dtype="float32",
        data_b64=base64.b64encode(raw).decode("ascii"),
    )


def export_to_neuroir(model) -> NeuroIRSpec:
    """Export a trained FNO1d / FNO2d / FNO3d / DeepONet / TokenMixer /
    GraphOp / TokenMixer2D / GraphOp2D to NeuroIRSpec.

    Sprint 3.27: this function used to be 130 lines
    of isinstance() branching with a hand-written
    spec-dict per op family.  It's now a single
    codegen dispatch through the OpSpec schemas
    (see _OP_SCHEMAS).  Adding a new op is a
    one-entry schema + one model_class bind.
    """
    op_name = _model_to_op_name(model)
    op_spec = get_op_spec(op_name)
    cfg = model.config
    spec_dict: dict = {}
    # All config-dict keys (int32-packed + cfg-only
    # fields like hidden_branch + cfg_extras) go
    # into the JSON spec.  The int32-block
    # packer (export_to_binary) only looks at the
    # int32_index of the schema; the loader
    # (load.py) reconstructs the model from the
    # full JSON spec, so all fields must be
    # present here.
    for key in op_spec.cfg_keys:
        if key == "_pad":
            continue
        spec_dict[key] = int(getattr(cfg, key))
    # cfg-only fields (e.g. hidden_branch in
    # DeepONet) are listed in cfg_extras; we
    # also add the standard 'name' / 'activation'
    # fields.  Anything that's an attribute of the
    # model.config and not in cfg_keys goes here.
    cfg_attrs = set(cfg.__dataclass_fields__.keys()) \
        if hasattr(cfg, '__dataclass_fields__') \
        else set(dir(cfg))
    for key in op_spec.cfg_extras:
        if key in spec_dict:
            continue
        if key in cfg_attrs:
            spec_dict[key] = getattr(cfg, key)
    # Finally, sweep in any remaining config
    # attributes that aren't already in spec_dict
    # and aren't private/built-in.  This catches
    # the "cfg-only" fields that aren't in
    # cfg_keys or cfg_extras (e.g. DeepONet's
    # hidden_branch/hidden_trunk) without
    # having to declare each one in the schema.
    for key in sorted(cfg_attrs):
        if key.startswith("_"):
            continue
        if key in spec_dict:
            continue
        # Skip private/internal attributes
        if key in {"__dataclass_fields__"}:
            continue
        spec_dict[key] = getattr(cfg, key)
    # Apply defaults for missing optional keys.
    for key, default in op_spec.defaults.items():
        spec_dict.setdefault(key, default)
    spec = NeuroIRSpec(op=op_name, config=spec_dict)
    for name, t in model.state_dict_for_ir().items():
        spec.weights[name] = _tensor_to_entry(name, t)
    return spec


def export_to_binary(spec: NeuroIRSpec) -> bytes:
    """Serialize a NeuroIRSpec to the native binary format (.nneuroir).

    Sprint 3.27: this function used to be 100 lines
    of elif-branching, one per op family, each
    hand-packing the int32 config block from the
    spec dict.  It's now a single codegen dispatch
    through the OpSpec schema: the schema is the
    single source of truth for "which int32 slot
    holds which cfg-dict key".  Adding a new op
    is a one-entry schema addition.
    """
    op_spec = get_op_spec(spec.op)
    op_code = op_spec.op_code
    version = op_spec.version
    config_len = op_spec.config_len
    cfg = spec.config
    act = _ACT_NAME_TO_CODE.get(
        cfg.get("activation", "gelu"), _ACT_GELU)
    # Pack the int32 config block.  For each
    # cfg_key in the schema, look up its position
    # in the int32 block and write the value (or
    # 0 for "_pad" placeholders, or the value from
    # cfg dict for real keys).
    config_block = [0] * config_len
    for key, pos in op_spec.int32_index.items():
        if pos < 0:
            continue
        if key == "_pad":
            config_block[pos] = 0
        else:
            config_block[pos] = int(cfg[key])
    out = bytearray()
    out += b"NIR0"
    out += struct.pack("<HBB", version, op_code, 0)
    out += struct.pack(f"<{config_len}i", *config_block)
    out += struct.pack("<B", act) + b"\x00\x00\x00"
    out += struct.pack("<I", len(spec.weights))

    for name, entry in spec.weights.items():
        name_b = name.encode("utf-8")
        if len(name_b) > 255:
            raise ValueError(f"weight name too long: {name}")
        out += struct.pack("<B", len(name_b))
        out += name_b
        ndim = len(entry.shape)
        if ndim > 255:
            raise ValueError(f"weight rank too large: {name}")
        out += struct.pack("<B", ndim)
        out += struct.pack(f"<{ndim}i", *entry.shape)
        out += base64.b64decode(entry.data_b64)

    # Optional NIRQ (quantisation) trailing block — v0.15.0
    # (per-tensor) and v0.16.0 (per-channel weights) and
    # v0.17.0 (per-token activations) and v0.21.0
    # (FP8 E4M3 activations).
    # When `quant` is present in the spec, append the block:
    #   magic     : 4 bytes   = "NIRQ"
    #   quant_flag: uint8     = 1 (W8A8 / FP8)
    #   n_qparams : uint32
    #   for each qparam:
    #     kind    : uint8     = 0 (per-tensor INT8) |
    #                           1 (per-channel INT8) |
    #                           2 (per-token INT8) |
    #                           3 (per-tensor FP8 E4M3)
    #     name_len: uint8
    #     name    : name_len bytes
    #     if kind == 0 (per-tensor INT8):
    #         scale   : float32
    #         zp      : int32
    #     if kind == 1 (per-channel INT8):
    #         n_channels : uint32
    #         channel_axis : uint8
    #         scales      : float32 * n_channels
    #         zero_points : int32   * n_channels
    #     if kind == 2 (per-token INT8):
    #         n_tokens : uint32
    #         width    : uint32
    #         scales   : float32 * n_tokens
    #         zero_points : int32 * n_tokens
    #     if kind == 3 (per-tensor FP8 E4M3):
    #         scale   : float32
    quant = spec.quant or {}
    if quant.get("enabled", False):
        out += b"NIRQ"
        out += struct.pack("<B", 1)  # quant_flag = 1
        qparams = quant.get("qparams", [])
        out += struct.pack("<I", len(qparams))
        for name, qp in qparams.items():
            name_b = name.encode("utf-8")
            if len(name_b) > 255:
                raise ValueError(f"qparam name too long: {name}")
            is_per_channel = qp.get("per_channel", False)
            is_per_token = qp.get("per_token", False)
            is_fp8 = qp.get("format") == "fp8_e4m3"
            if is_per_token:
                out += struct.pack("<B", 2)  # kind = 2 (per-token)
                out += struct.pack("<B", len(name_b))
                out += name_b
                scales = qp["scales"]
                zero_points = qp["zero_points"]
                n_tok = len(scales)
                out += struct.pack("<II", n_tok, int(qp.get("width", 0)))
                out += struct.pack(f"<{n_tok}f", *[float(s) for s in scales])
                out += struct.pack(f"<{n_tok}i", *[int(z) for z in zero_points])
            elif is_per_channel:
                out += struct.pack("<B", 1)  # kind = 1 (per-channel)
                out += struct.pack("<B", len(name_b))
                out += name_b
                scales = qp["scales"]
                zero_points = qp["zero_points"]
                n_ch = len(scales)
                out += struct.pack("<IB", n_ch, int(qp.get("channel_axis", 0)))
                out += struct.pack(f"<{n_ch}f", *[float(s) for s in scales])
                out += struct.pack(f"<{n_ch}i", *[int(z) for z in zero_points])
            elif is_fp8:
                out += struct.pack("<B", 3)  # kind = 3 (FP8 E4M3)
                out += struct.pack("<B", len(name_b))
                out += name_b
                out += struct.pack("<f", float(qp["scale"]))
            else:
                out += struct.pack("<B", 0)  # kind = 0 (per-tensor)
                out += struct.pack("<B", len(name_b))
                out += name_b
                out += struct.pack("<fi", float(qp["scale"]), int(qp["zero_point"]))
    return bytes(out)


def export_all(
    model, out_dir: str | Path, basename: str = "model",
    quant: dict | None = None,
) -> tuple[Path, Path]:
    """Export both the JSON (.neuroir) and binary (.nneuroir) formats.

    If `quant` is provided (a dict shaped like
    `quant_to_ir(...)`'s output), the IR is written with
    the v0.15.0 NIRQ trailing block containing per-tensor
    INT8 scale / zero_point for every weight tensor and
    every calibrated activation.  The C++ v0.15.0 runtime
    applies the same fake-quant round-trip the Python
    reference does.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = export_to_neuroir(model)
    if quant is not None:
        spec.quant = quant
    json_path = out_dir / f"{basename}.neuroir"
    bin_path = out_dir / f"{basename}.nneuroir"
    spec.save(json_path)
    bin_path.write_bytes(export_to_binary(spec))
    return json_path, bin_path

