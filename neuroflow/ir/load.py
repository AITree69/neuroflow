"""Load a NeuroIR spec into either PyTorch or a NumPy representation.

The C++ runtime uses its own IR loader (cpp/src/runtime.cpp). This module is
for Python-side validation and roundtrip testing.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Union

import numpy as np
import torch

from neuroflow.ir.spec import NeuroIRSpec
from neuroflow.nn.deeponet import DeepONet, DeepONetConfig
from neuroflow.nn.fno import FNO1d, FNO1dConfig
from neuroflow.nn.fno2d import FNO2d, FNO2dConfig
from neuroflow.nn.fno3d import FNO3d, FNO3dConfig
from neuroflow.nn.graph_op import GraphOp, GraphOpConfig
from neuroflow.nn.graph_op2d import GraphOp2D, GraphOp2DConfig
from neuroflow.nn.tokenmixer import TokenMixer, TokenMixerConfig
from neuroflow.nn.tokenmixer2d import TokenMixer2D, TokenMixer2DConfig


def load_neuroir(path: Union[str, NeuroIRSpec]):
    """Construct an FNO1d / FNO2d / FNO3d / DeepONet / TokenMixer /
    GraphOp / TokenMixer2D / GraphOp2D from a NeuroIR spec."""
    if isinstance(path, str):
        spec = NeuroIRSpec.load(path)
    else:
        spec = path

    if spec.op == "FNO1d":
        cfg = FNO1dConfig(**spec.config)
        model = FNO1d(cfg)
    elif spec.op == "FNO2d":
        cfg = FNO2dConfig(**spec.config)
        model = FNO2d(cfg)
    elif spec.op == "FNO3d":
        cfg = FNO3dConfig(**spec.config)
        model = FNO3d(cfg)
    elif spec.op == "DeepONet":
        cfg = DeepONetConfig(**spec.config)
        model = DeepONet(cfg)
    elif spec.op == "TokenMixer":
        cfg = TokenMixerConfig(**spec.config)
        model = TokenMixer(cfg)
    elif spec.op == "GraphOp":
        cfg = GraphOpConfig(**spec.config)
        model = GraphOp(cfg)
    elif spec.op == "TokenMixer2D":
        cfg = TokenMixer2DConfig(**spec.config)
        model = TokenMixer2D(cfg)
    elif spec.op == "GraphOp2D":
        cfg = GraphOp2DConfig(**spec.config)
        model = GraphOp2D(cfg)
    else:
        raise ValueError(f"unsupported op: {spec.op}")

    expected = model.state_dict_for_ir()
    for name, ref in expected.items():
        if name not in spec.weights:
            raise KeyError(f"missing weight in IR: {name}")
        t = spec.weights[name].to_torch()
        if tuple(t.shape) != tuple(ref.shape):
            raise ValueError(
                f"shape mismatch for {name}: IR {tuple(t.shape)} vs model {tuple(ref.shape)}"
            )
        ref.copy_(t)

    model.eval()
    return model


def load_neuroir_weights(path: Union[str, NeuroIRSpec]) -> "OrderedDict[str, np.ndarray]":
    """Load only the weight arrays (for cross-language validation)."""
    spec = NeuroIRSpec.load(path) if isinstance(path, str) else path
    return OrderedDict((name, t.to_numpy()) for name, t in spec.weights.items())


# ----------------------------------------------------------------------------
# Pure-NumPy forward — FNO1d
# ----------------------------------------------------------------------------

def _act(z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "gelu":
        # tanh approximation
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z ** 3)))
    if activation == "relu":
        return np.maximum(z, 0.0)
    raise ValueError(activation)


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z_max = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _linear(x: np.ndarray, W: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """Apply nn.Linear: y = x @ W^T + b. W: (out, in)."""
    y = x @ W.T
    if b is not None:
        y = y + b
    return y


def _predict_fno1d(spec: NeuroIRSpec, x: np.ndarray) -> np.ndarray:
    """Pure NumPy forward for an FNO1d NeuroIR spec.

    Args:
        spec: NeuroIRSpec with op='FNO1d'.
        x: (batch, n, in_channels) float32.

    Returns:
        y: (batch, n, out_channels) float32.
    """
    if spec.op != "FNO1d":
        raise ValueError(f"_predict_fno1d: spec.op is {spec.op!r}")
    cfg = spec.config
    w = cfg["width"]
    modes = cfg["modes"]
    n_layers = cfg["n_layers"]
    activation = cfg["activation"]
    pad_factor = cfg["pad_factor"]
    in_ch = cfg["in_channels"]
    out_ch = cfg["out_channels"]

    def _get(name: str) -> np.ndarray:
        return spec.weights[name].to_numpy()

    # Lifting: x (b, n, in_ch) @ W (in_ch, w) + b (w)
    Wlift = _get("lift.weight")  # (w, in_ch)
    blift = _get("lift.bias")  # (w,)
    h = x @ Wlift.T + blift  # (b, n, w)
    h = h.transpose(0, 2, 1)  # (b, w, n)

    if pad_factor > 1:
        pad_len = (pad_factor - (h.shape[-1] % pad_factor)) % pad_factor
        if pad_len > 0:
            h = np.pad(h, ((0, 0), (0, 0), (0, pad_len)))

    for i in range(n_layers):
        # Spectral conv
        W_real = _get(f"specs.{i}.weights_real")  # (w, w, modes)
        W_imag = _get(f"specs.{i}.weights_imag")
        Wspec = W_real + 1j * W_imag
        H = np.fft.rfft(h, axis=-1)  # (b, w, n//2+1)
        H_trunc = H[..., :modes]  # (b, w, modes)
        out_ft = np.einsum("bim,iom->bom", H_trunc, Wspec)
        # Pad back
        n_orig = h.shape[-1]
        out_ft_pad = np.pad(out_ft, ((0, 0), (0, 0), (0, (n_orig // 2 + 1) - modes)))
        x1 = np.fft.irfft(out_ft_pad, n=n_orig, axis=-1)

        # Local (skip) linear
        Wloc = _get(f"locs.{i}.weight")  # (w, w)
        bloc = _get(f"locs.{i}.bias")  # (w,)
        x2 = h.transpose(0, 2, 1) @ Wloc.T + bloc  # (b, n, w)
        x2 = x2.transpose(0, 2, 1)  # (b, w, n)

        h = _act(x1 + x2, activation)

    if pad_factor > 1 and pad_len > 0:
        h = h[..., :-pad_len]

    # Project: (b, w, n) -> (b, n, w)
    h = h.transpose(0, 2, 1)
    Wq = _get("proj_q.weight")
    bq = _get("proj_q.bias")
    h = h @ Wq.T + bq
    h = _act(h, activation)
    Wout = _get("proj_out.weight")
    bout = _get("proj_out.bias")
    y = h @ Wout.T + bout
    return y.astype(np.float32)


# ----------------------------------------------------------------------------
# Pure-NumPy forward — FNO2d
# ----------------------------------------------------------------------------

def _predict_fno2d(spec: NeuroIRSpec, x: np.ndarray) -> np.ndarray:
    """Pure NumPy forward for an FNO2d NeuroIR spec.

    Args:
        spec: NeuroIRSpec with op='FNO2d'.
        x: (batch, h, w, in_channels) float32.

    Returns:
        y: (batch, h, w, out_channels) float32.
    """
    if spec.op != "FNO2d":
        raise ValueError(f"_predict_fno2d: spec.op is {spec.op!r}")
    cfg = spec.config
    width = cfg["width"]
    modes_h = cfg["modes_h"]
    modes_w = cfg["modes_w"]
    n_layers = cfg["n_layers"]
    activation = cfg["activation"]
    pad_factor = cfg["pad_factor"]
    in_ch = cfg["in_channels"]
    out_ch = cfg["out_channels"]

    def _get(name: str) -> np.ndarray:
        return spec.weights[name].to_numpy()

    # Lifting: x (b, h, w, in_ch) @ W (in_ch, width) + b (width)
    Wlift = _get("lift.weight")  # (width, in_ch)
    blift = _get("lift.bias")  # (width,)
    h = x @ Wlift.T + blift  # (b, h, w, width)
    h = h.transpose(0, 3, 1, 2)  # (b, width, h, w)

    # Pad to pad_factor multiples
    h_orig, w_orig = h.shape[-2], h.shape[-1]
    pad_h = pad_w = 0
    if pad_factor > 1:
        pad_h = (pad_factor - (h_orig % pad_factor)) % pad_factor
        pad_w = (pad_factor - (w_orig % pad_factor)) % pad_factor
        if pad_h > 0 or pad_w > 0:
            h = np.pad(h, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

    for i in range(n_layers):
        # Spectral conv: 2D rfft on last two dims
        W_real = _get(f"specs.{i}.weights_real")  # (w, w, mh, mw)
        W_imag = _get(f"specs.{i}.weights_imag")
        Wspec = W_real + 1j * W_imag
        H = np.fft.rfftn(h, axes=(-2, -1))  # (b, w, h, w//2+1)
        H_trunc = H[..., :modes_h, :modes_w]
        out_ft = np.einsum("bimn,iomn->bomn", H_trunc, Wspec)
        # Pad back: the h axis needs to go from modes_h -> h, the w axis from
        # modes_w -> w//2+1
        h_eff, w_eff = h.shape[-2], h.shape[-1]
        pad_h_spec = h_eff - modes_h
        pad_w_spec = (w_eff // 2 + 1) - modes_w
        out_ft_pad = np.pad(out_ft, ((0, 0), (0, 0), (0, pad_h_spec), (0, pad_w_spec)))
        x1 = np.fft.irfftn(out_ft_pad, s=(h_eff, w_eff), axes=(-2, -1))

        # Local (skip) linear: 1x1 conv equivalent — apply Linear on the
        # channel dim only. h is (b, w, h, w); permute to (b, h, w, w),
        # matmul on last axis, permute back.
        Wloc = _get(f"locs.{i}.weight")  # (w, w)
        bloc = _get(f"locs.{i}.bias")  # (w,)
        x2_perm = h.transpose(0, 2, 3, 1)  # (b, h, w, w)
        x2 = x2_perm @ Wloc.T + bloc  # (b, h, w, w)
        x2 = x2.transpose(0, 3, 1, 2)  # (b, w, h, w)

        h = _act(x1 + x2, activation)

    if pad_h > 0 or pad_w > 0:
        h = h[..., :h_orig, :w_orig]

    # Project: (b, width, h, w) -> (b, h, w, width) -> (b, h, w, out_ch)
    h = h.transpose(0, 2, 3, 1)
    Wq = _get("proj_q.weight")
    bq = _get("proj_q.bias")
    h = h @ Wq.T + bq
    h = _act(h, activation)
    Wout = _get("proj_out.weight")
    bout = _get("proj_out.bias")
    y = h @ Wout.T + bout
    return y.astype(np.float32)


# ----------------------------------------------------------------------------
# Pure-NumPy forward — FNO3d
# ----------------------------------------------------------------------------

def _predict_fno3d(spec: NeuroIRSpec, x: np.ndarray) -> np.ndarray:
    """Pure NumPy forward for an FNO3d NeuroIR spec.

    Args:
        spec: NeuroIRSpec with op='FNO3d'.
        x: (batch, h, w, d, in_channels) float32.

    Returns:
        y: (batch, h, w, d, out_channels) float32.
    """
    if spec.op != "FNO3d":
        raise ValueError(f"_predict_fno3d: spec.op is {spec.op!r}")
    cfg = spec.config
    width = cfg["width"]
    modes_h = cfg["modes_h"]
    modes_w = cfg["modes_w"]
    modes_d = cfg["modes_d"]
    n_layers = cfg["n_layers"]
    activation = cfg["activation"]
    pad_factor = cfg["pad_factor"]
    in_ch = cfg["in_channels"]
    out_ch = cfg["out_channels"]

    def _get(name: str) -> np.ndarray:
        return spec.weights[name].to_numpy()

    # Lifting: x (b, h, w, d, in_ch) @ W (in_ch, width) + b (width)
    Wlift = _get("lift.weight")  # (width, in_ch)
    blift = _get("lift.bias")  # (width,)
    h = x @ Wlift.T + blift  # (b, h, w, d, width)
    h = h.transpose(0, 4, 1, 2, 3)  # (b, width, h, w, d)

    # Pad to pad_factor multiples
    h_orig, w_orig, d_orig = h.shape[-3], h.shape[-2], h.shape[-1]
    pad_h = pad_w = pad_d = 0
    if pad_factor > 1:
        pad_h = (pad_factor - (h_orig % pad_factor)) % pad_factor
        pad_w = (pad_factor - (w_orig % pad_factor)) % pad_factor
        pad_d = (pad_factor - (d_orig % pad_factor)) % pad_factor
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            h = np.pad(
                h,
                ((0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, pad_d)),
            )

    for i in range(n_layers):
        # Spectral conv: 3D rfft on last three dims
        W_real = _get(f"specs.{i}.weights_real")  # (w, w, mh, mw, md)
        W_imag = _get(f"specs.{i}.weights_imag")
        Wspec = W_real + 1j * W_imag
        H = np.fft.rfftn(h, axes=(-3, -2, -1))  # (b, w, h, w, d//2+1)
        H_trunc = H[..., :modes_h, :modes_w, :modes_d]
        out_ft = np.einsum("bimnp,iomnp->bomnp", H_trunc, Wspec)
        # Pad back: h -> h_eff, w -> w_eff, d axis -> d_eff//2+1
        h_eff, w_eff, d_eff = h.shape[-3], h.shape[-2], h.shape[-1]
        pad_h_spec = h_eff - modes_h
        pad_w_spec = w_eff - modes_w
        pad_d_spec = (d_eff // 2 + 1) - modes_d
        out_ft_pad = np.pad(
            out_ft,
            (
                (0, 0),
                (0, 0),
                (0, pad_h_spec),
                (0, pad_w_spec),
                (0, pad_d_spec),
            ),
        )
        x1 = np.fft.irfftn(out_ft_pad, s=(h_eff, w_eff, d_eff), axes=(-3, -2, -1))

        # Local (skip) linear: apply Linear on the channel dim only.
        Wloc = _get(f"locs.{i}.weight")  # (w, w)
        bloc = _get(f"locs.{i}.bias")  # (w,)
        x2_perm = h.transpose(0, 2, 3, 4, 1)  # (b, h, w, d, w)
        x2 = x2_perm @ Wloc.T + bloc  # (b, h, w, d, w)
        x2 = x2.transpose(0, 4, 1, 2, 3)  # (b, w, h, w, d)

        h = _act(x1 + x2, activation)

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        h = h[..., :h_orig, :w_orig, :d_orig]

    # Project: (b, width, h, w, d) -> (b, h, w, d, width) -> (b, h, w, d, out_ch)
    h = h.transpose(0, 2, 3, 4, 1)
    Wq = _get("proj_q.weight")
    bq = _get("proj_q.bias")
    h = h @ Wq.T + bq
    h = _act(h, activation)
    Wout = _get("proj_out.weight")
    bout = _get("proj_out.bias")
    y = h @ Wout.T + bout
    return y.astype(np.float32)


# ----------------------------------------------------------------------------
# Pure-NumPy forward — DeepONet
# ----------------------------------------------------------------------------

def _mlp_forward_numpy(
    mlp_layers_w: list[np.ndarray],
    mlp_layers_b: list[np.ndarray],
    x: np.ndarray,
    activation: str,
) -> np.ndarray:
    """Apply a sequence of Linear+activation layers in NumPy."""
    for i, (W, b) in enumerate(zip(mlp_layers_w, mlp_layers_b, strict=True)):
        x = x @ W.T + b
        if i < len(mlp_layers_w) - 1:
            x = _act(x, activation)
    return x


def _predict_deeponet(spec: NeuroIRSpec, u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pure NumPy forward for a DeepONet NeuroIR spec.

    Args:
        spec: NeuroIRSpec with op='DeepONet'.
        u: (batch, n_sensor, in_branch) float32 — input function values.
        y: (batch, n_query, in_trunk) float32 — query locations.

    Returns:
        out: (batch, n_query, out_channels) float32.
    """
    if spec.op != "DeepONet":
        raise ValueError(f"_predict_deeponet: spec.op is {spec.op!r}")
    cfg = spec.config
    activation = cfg["activation"]
    out_channels = cfg["out_channels"]
    latent_dim = cfg["latent_dim"]

    def _get(name: str) -> np.ndarray:
        return spec.weights[name].to_numpy()

    # Branch
    branch_w = [_get(f"branch.layers.{i}.weight") for i in range(cfg["n_layers_branch"])]
    branch_b = [_get(f"branch.layers.{i}.bias") for i in range(cfg["n_layers_branch"])]
    b = _mlp_forward_numpy(branch_w, branch_b, u, activation)  # (b, n_sensor, out_ch * latent_dim)
    bsz, n_sensor, _ = b.shape
    b = b.reshape(bsz, n_sensor, out_channels, latent_dim).mean(axis=1)  # (b, out_ch, latent_dim)

    # Trunk
    trunk_w = [_get(f"trunk.layers.{i}.weight") for i in range(cfg["n_layers_trunk"])]
    trunk_b = [_get(f"trunk.layers.{i}.bias") for i in range(cfg["n_layers_trunk"])]
    t = _mlp_forward_numpy(trunk_w, trunk_b, y, activation)  # (b, n_query, latent_dim)

    # Dot product + per-channel bias.
    out = np.einsum("bck,bik->bci", b, t)  # (b, out_ch, n_query)
    out = out.transpose(0, 2, 1)             # (b, n_query, out_ch)
    bias = _get("bias.weight")              # (out_ch,)
    out = out + bias
    return out.astype(np.float32)


def _predict_tokenmixer(spec: NeuroIRSpec, x: np.ndarray) -> np.ndarray:
    """Pure NumPy forward for a TokenMixer (Transolver-style) NeuroIR spec.

    Mirrors neuroflow/nn/tokenmixer.py:TokenMixer exactly. n_layers is fixed
    to 1 for Stage 2; multi-layer is a Stage 3 extension.

    Args:
        spec: NeuroIRSpec with op='TokenMixer'.
        x: (batch, n_points, in_dim) float32.

    Returns:
        y: (batch, n_points, out_dim) float32.
    """
    if spec.op != "TokenMixer":
        raise ValueError(f"_predict_tokenmixer: spec.op is {spec.op!r}")
    cfg = spec.config
    in_dim = cfg["in_dim"]
    out_dim = cfg["out_dim"]
    n_points = cfg["n_points"]
    n_patches = cfg["n_patches"]
    latent_dim = cfg["latent_dim"]
    n_heads = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    activation = cfg["activation"]
    if n_layers != 1:
        raise NotImplementedError(
            f"predict_tokenmixer: n_layers={n_layers} not supported (Stage 2 limit=1)"
        )
    if n_points % n_patches != 0:
        raise ValueError(
            f"predict_tokenmixer: n_points ({n_points}) must be a multiple of "
            f"n_patches ({n_patches})"
        )
    head_dim = latent_dim // n_heads
    scale = head_dim ** -0.5
    pp = n_points // n_patches

    def _get(name: str) -> np.ndarray:
        return spec.weights[name].to_numpy()

    # SliceEmbed: (b, n_points, in_dim) -> (b, n_patches, latent_dim)
    se_w = _get("slice_embed.proj.weight")  # (latent_dim, in_dim)
    se_b = _get("slice_embed.proj.bias")    # (latent_dim,)
    bsz = x.shape[0]
    x_reshape = x.reshape(bsz, n_patches, pp, in_dim).mean(axis=2)  # (b, n_patches, in_dim)
    tokens = _linear(x_reshape, se_w, se_b)  # (b, n_patches, latent_dim)

    # Transformer blocks
    for li in range(n_layers):
        # Pre-LN
        h = _layer_norm(
            tokens,
            _get(f"blocks.{li}.ln1.weight"),
            _get(f"blocks.{li}.ln1.bias"),
        )  # (b, n_patches, latent_dim)
        # Q, K, V: (b, n_patches, latent_dim) each
        Q = _linear(h, _get(f"blocks.{li}.q_proj.weight"), _get(f"blocks.{li}.q_proj.bias"))
        K = _linear(h, _get(f"blocks.{li}.k_proj.weight"), _get(f"blocks.{li}.k_proj.bias"))
        V = _linear(h, _get(f"blocks.{li}.v_proj.weight"), _get(f"blocks.{li}.v_proj.bias"))
        # Reshape to multi-head: (b, n_patches, n_heads, head_dim)
        # -> (b, n_heads, n_patches, head_dim)
        Q_h = Q.reshape(bsz, n_patches, n_heads, head_dim).transpose(0, 2, 1, 3)
        K_h = K.reshape(bsz, n_patches, n_heads, head_dim).transpose(0, 2, 1, 3)
        V_h = V.reshape(bsz, n_patches, n_heads, head_dim).transpose(0, 2, 1, 3)
        # Attention scores: (b, n_heads, n_patches, n_patches)
        attn = (Q_h @ K_h.transpose(0, 1, 3, 2)) * scale
        attn = _softmax(attn, axis=-1)
        # (b, n_heads, n_patches, head_dim) -> (b, n_patches, n_heads, head_dim)
        out_h = (attn @ V_h).transpose(0, 2, 1, 3).reshape(bsz, n_patches, latent_dim)
        out_h = _linear(
            out_h,
            _get(f"blocks.{li}.o_proj.weight"),
            _get(f"blocks.{li}.o_proj.bias"),
        )
        tokens = tokens + out_h  # residual

        # FFN with pre-LN
        h2 = _layer_norm(
            tokens,
            _get(f"blocks.{li}.ln2.weight"),
            _get(f"blocks.{li}.ln2.bias"),
        )
        ffn_h = _linear(
            h2,
            _get(f"blocks.{li}.ffn0.weight"),
            _get(f"blocks.{li}.ffn0.bias"),
        )
        ffn_h = _act(ffn_h, activation)
        ffn_h = _linear(
            ffn_h,
            _get(f"blocks.{li}.ffn1.weight"),
            _get(f"blocks.{li}.ffn1.bias"),
        )
        tokens = tokens + ffn_h

    # UnsliceDecode: broadcast tokens across pp per patch + concat with x features
    tokens_rep = np.repeat(tokens, pp, axis=1)  # (b, n_points, latent_dim)
    h_unslice = np.concatenate([x, tokens_rep], axis=-1)  # (b, n_points, in_dim+latent_dim)
    features = _linear(
        h_unslice,
        _get("unslice.proj.weight"),
        _get("unslice.proj.bias"),
    )  # (b, n_points, latent_dim)

    # Head
    y = _linear(features, _get("head.weight"), _get("head.bias"))  # (b, n_points, out_dim)
    return y.astype(np.float32)


def _predict_graphop(spec: NeuroIRSpec, x: np.ndarray) -> np.ndarray:
    """Pure NumPy forward for a GraphOp (GCN-style) NeuroIR spec.

    Mirrors neuroflow/nn/graph_op.py:GraphOp exactly. n_layers is fixed
    to 1 for Stage 2; multi-block is a Stage 3 extension.

    Args:
        spec: NeuroIRSpec with op='GraphOp'.
        x: (batch, n_nodes, in_dim) float32.

    Returns:
        y: (batch, n_nodes, out_dim) float32.
    """
    if spec.op != "GraphOp":
        raise ValueError(f"_predict_graphop: spec.op is {spec.op!r}")
    cfg = spec.config
    in_dim = cfg["in_dim"]
    out_dim = cfg["out_dim"]
    n_nodes = cfg["n_nodes"]
    hidden_dim = cfg["hidden_dim"]
    n_layers = cfg["n_layers"]
    activation = cfg["activation"]
    if n_layers != 1:
        raise NotImplementedError(
            f"predict_graphop: n_layers={n_layers} not supported (Stage 2 limit=1)"
        )

    def _get(name: str) -> np.ndarray:
        return spec.weights[name].to_numpy()

    # Graph topology (cast back from float32 storage to int / float).
    adj_offsets = _get("graph.adj_offsets").astype(np.int64)
    adj_indices = _get("graph.adj_indices").astype(np.int64)
    deg_inv = _get("graph.deg_inv").astype(np.float32)

    bsz = x.shape[0]
    # Lift
    h = _linear(x, _get("lift.weight"), _get("lift.bias"))  # (b, n_nodes, hidden)

    for li in range(n_layers):
        # Aggregate neighbors: for each node i, sum over j in adj[i] of
        # h[j] * deg_inv[i].
        agg = np.zeros((bsz, n_nodes, hidden_dim), dtype=np.float32)
        for i in range(n_nodes):
            nbs = adj_indices[adj_offsets[i]:adj_offsets[i + 1]]
            agg[:, i, :] = h[:, nbs, :].sum(axis=1) * deg_inv[i]
        # GCN block: act(W_self h + W_neigh agg) + h
        h_self = _linear(h, _get(f"blocks.{li}.lin_self.weight"),
                          _get(f"blocks.{li}.lin_self.bias"))
        h_neigh = _linear(agg, _get(f"blocks.{li}.lin_neigh.weight"),
                           _get(f"blocks.{li}.lin_neigh.bias"))
        h = _act(h_self + h_neigh, activation) + h

    y = _linear(h, _get("head.weight"), _get("head.bias"))
    return y.astype(np.float32)


def _predict_tokenmixer2d(spec: NeuroIRSpec, x: np.ndarray) -> np.ndarray:
    """Pure NumPy forward for a TokenMixer2D (2D Transolver-style) NeuroIR spec.

    Mirrors neuroflow/nn/tokenmixer2d.py:TokenMixer2D exactly. n_layers is
    fixed to 1 for Stage 2; multi-layer is a Stage 3 extension.

    Args:
        spec: NeuroIRSpec with op='TokenMixer2D'.
        x: (batch, h, w, in_dim) float32.

    Returns:
        y: (batch, h, w, out_dim) float32.
    """
    if spec.op != "TokenMixer2D":
        raise ValueError(f"_predict_tokenmixer2d: spec.op is {spec.op!r}")
    cfg = spec.config
    in_dim = cfg["in_dim"]
    out_dim = cfg["out_dim"]
    h = cfg["h"]
    w = cfg["w"]
    n_patches = cfg["n_patches"]
    latent_dim = cfg["latent_dim"]
    n_heads = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    activation = cfg["activation"]
    if n_layers != 1:
        raise NotImplementedError(
            f"predict_tokenmixer2d: n_layers={n_layers} not supported (Stage 2 limit=1)"
        )
    n_points = h * w
    if n_points % n_patches != 0:
        raise ValueError(
            f"predict_tokenmixer2d: h*w ({n_points}) must be a multiple of "
            f"n_patches ({n_patches})"
        )
    pp = n_points // n_patches
    head_dim = latent_dim // n_heads
    scale = head_dim ** -0.5

    def _get(name: str) -> np.ndarray:
        return spec.weights[name].to_numpy()

    bsz = x.shape[0]
    # Flatten (b, h, w, in_dim) -> (b, n_points, in_dim).
    x_flat = x.reshape(bsz, n_points, in_dim)
    # Mean-pool into patches -> (b, n_patches, in_dim) then lift.
    x_patches = x_flat.reshape(bsz, n_patches, pp, in_dim).mean(axis=2)
    tokens = _linear(
        x_patches,
        _get("slice_embed.proj.weight"),
        _get("slice_embed.proj.bias"),
    )  # (b, n_patches, latent_dim)

    for li in range(n_layers):
        # Pre-LN 1.
        h_norm = _layer_norm(
            tokens,
            _get(f"blocks.{li}.ln1.weight"),
            _get(f"blocks.{li}.ln1.bias"),
        )
        Q = _linear(h_norm, _get(f"blocks.{li}.q_proj.weight"),
                     _get(f"blocks.{li}.q_proj.bias"))
        K = _linear(h_norm, _get(f"blocks.{li}.k_proj.weight"),
                     _get(f"blocks.{li}.k_proj.bias"))
        V = _linear(h_norm, _get(f"blocks.{li}.v_proj.weight"),
                     _get(f"blocks.{li}.v_proj.bias"))
        # Multi-head reshape.
        Q_h = Q.reshape(bsz, n_patches, n_heads, head_dim).transpose(0, 2, 1, 3)
        K_h = K.reshape(bsz, n_patches, n_heads, head_dim).transpose(0, 2, 1, 3)
        V_h = V.reshape(bsz, n_patches, n_heads, head_dim).transpose(0, 2, 1, 3)
        attn = (Q_h @ K_h.transpose(0, 1, 3, 2)) * scale
        attn = _softmax(attn, axis=-1)
        out_h = (attn @ V_h).transpose(0, 2, 1, 3).reshape(bsz, n_patches, latent_dim)
        out_h = _linear(out_h, _get(f"blocks.{li}.o_proj.weight"),
                         _get(f"blocks.{li}.o_proj.bias"))
        tokens = tokens + out_h
        # Pre-LN 2 + FFN.
        h_norm2 = _layer_norm(
            tokens,
            _get(f"blocks.{li}.ln2.weight"),
            _get(f"blocks.{li}.ln2.bias"),
        )
        ffn = _linear(h_norm2, _get(f"blocks.{li}.ffn0.weight"),
                        _get(f"blocks.{li}.ffn0.bias"))
        ffn = _act(ffn, activation)
        ffn = _linear(ffn, _get(f"blocks.{li}.ffn1.weight"),
                        _get(f"blocks.{li}.ffn1.bias"))
        tokens = tokens + ffn

    # Broadcast tokens to per-point: (b, n_points, latent).
    tokens_rep = np.repeat(tokens, pp, axis=1)
    h_in = np.concatenate([x_flat, tokens_rep], axis=-1)  # (b, n_points, in_dim+latent)
    features = _linear(h_in, _get("unslice.proj.weight"),
                        _get("unslice.proj.bias"))  # (b, n_points, latent)

    y_flat = _linear(features, _get("head.weight"),
                       _get("head.bias"))  # (b, n_points, out_dim)
    return y_flat.reshape(bsz, h, w, out_dim).astype(np.float32)


def _predict_graphop2d(spec: NeuroIRSpec, x: np.ndarray) -> np.ndarray:
    """Pure NumPy forward for a GraphOp2D (2D GCN-style) NeuroIR spec.

    Mirrors neuroflow/nn/graph_op2d.py:GraphOp2D exactly. n_layers is
    fixed to 1 for Stage 2; multi-block is a Stage 3 extension.

    Args:
        spec: NeuroIRSpec with op='GraphOp2D'.
        x: (batch, h, w, in_dim) float32.

    Returns:
        y: (batch, h, w, out_dim) float32.
    """
    if spec.op != "GraphOp2D":
        raise ValueError(f"_predict_graphop2d: spec.op is {spec.op!r}")
    cfg = spec.config
    in_dim = cfg["in_dim"]
    out_dim = cfg["out_dim"]
    h = cfg["h"]
    w = cfg["w"]
    hidden_dim = cfg["hidden_dim"]
    n_layers = cfg["n_layers"]
    activation = cfg["activation"]
    if n_layers != 1:
        raise NotImplementedError(
            f"predict_graphop2d: n_layers={n_layers} not supported (Stage 2 limit=1)"
        )

    def _get(name: str) -> np.ndarray:
        return spec.weights[name].to_numpy()

    adj_offsets = _get("graph.adj_offsets").astype(np.int64)
    adj_indices = _get("graph.adj_indices").astype(np.int64)
    deg_inv = _get("graph.deg_inv").astype(np.float32)

    bsz = x.shape[0]
    n_nodes = h * w
    # Flatten: (b, n_nodes, in_dim).
    x_flat = x.reshape(bsz, n_nodes, in_dim)
    # Lift.
    h_feat = _linear(x_flat, _get("lift.weight"), _get("lift.bias"))

    for li in range(n_layers):
        agg = np.zeros((bsz, n_nodes, hidden_dim), dtype=np.float32)
        for i in range(n_nodes):
            nbs = adj_offsets[i]
            nbe = adj_offsets[i + 1]
            agg[:, i, :] = h_feat[:, adj_indices[nbs:nbe], :].sum(axis=1) * deg_inv[i]
        h_self = _linear(h_feat, _get(f"blocks.{li}.lin_self.weight"),
                          _get(f"blocks.{li}.lin_self.bias"))
        h_neigh = _linear(agg, _get(f"blocks.{li}.lin_neigh.weight"),
                           _get(f"blocks.{li}.lin_neigh.bias"))
        h_feat = _act(h_self + h_neigh, activation) + h_feat

    y_flat = _linear(h_feat, _get("head.weight"), _get("head.bias"))
    return y_flat.reshape(bsz, h, w, out_dim).astype(np.float32)


def predict_with_spec(
    spec: NeuroIRSpec,
    x: np.ndarray,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Pure NumPy forward pass of an FNO1d / FNO2d / FNO3d / DeepONet /
    TokenMixer / GraphOp from IR.

    Dispatches by ``spec.op``.

    Args:
        spec: NeuroIRSpec.
        x: (batch, n, in_channels) for FNO1d, or
           (batch, h, w, in_channels) for FNO2d, or
           (batch, h, w, d, in_channels) for FNO3d, or
           (batch, n_sensor, in_branch) for DeepONet (the input function), or
           (batch, n_points, in_dim) for TokenMixer, or
           (batch, n_nodes, in_dim) for GraphOp.
        y: only used by DeepONet — (batch, n_query, in_trunk) the query
           locations. Required when ``spec.op == "DeepONet"``.

    Returns:
        y: (batch, n, out_channels) for FNO1d, or
           (batch, h, w, out_channels) for FNO2d, or
           (batch, h, w, d, out_channels) for FNO3d, or
           (batch, n_query, out_channels) for DeepONet, or
           (batch, n_points, out_dim) for TokenMixer, or
           (batch, n_nodes, out_dim) for GraphOp.
    """
    if spec.op == "FNO1d":
        return _predict_fno1d(spec, x)
    if spec.op == "FNO2d":
        return _predict_fno2d(spec, x)
    if spec.op == "FNO3d":
        return _predict_fno3d(spec, x)
    if spec.op == "DeepONet":
        if y is None:
            raise ValueError("predict_with_spec: DeepONet requires both x (u) and y (query)")
        return _predict_deeponet(spec, x, y)
    if spec.op == "TokenMixer":
        return _predict_tokenmixer(spec, x)
    if spec.op == "GraphOp":
        return _predict_graphop(spec, x)
    if spec.op == "TokenMixer2D":
        return _predict_tokenmixer2d(spec, x)
    if spec.op == "GraphOp2D":
        return _predict_graphop2d(spec, x)
    raise ValueError(f"unsupported op: {spec.op}")


def predict_with_spec_torch(
    spec: NeuroIRSpec,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch-based forward (matches ``predict_with_spec`` exactly)."""
    model = load_neuroir(spec)
    model.eval()
    with torch.no_grad():
        if spec.op == "DeepONet":
            assert y is not None, "DeepONet requires both x (u) and y (query)"
            return model(x, y)
        return model(x)
