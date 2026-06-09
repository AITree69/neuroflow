// =============================================================================
// NeuroFlow C++ Runtime — native NeuroIR binary loader
// =============================================================================
//
// Reads the .nneuroir format produced by neuroflow/ir/export.py:export_to_binary.
// Layout (little-endian), version 1 / 2 / 3:
//   magic      : 4 bytes   = "NIR0"
//   version    : uint16    = 1 | 2 | 3
//   op_code    : uint8     = 0x01 (FNO1d) | 0x02 (FNO2d) | 0x03 (FNO3d) |
//                            0x04 (DeepONet) | 0x05 (TokenMixer) |
//                            0x06 (GraphOp) | 0x07 (TokenMixer2D) |
//                            0x08 (GraphOp2D)
//   reserved   : uint8     = 0
//   config     : int32s   — length depends on op_code:
//                  FNO1d / FNO2d / DeepONet:                   7 int32s
//                  FNO3d / TokenMixer / GraphOp /
//                  TokenMixer2D / GraphOp2D:                    8 int32s
//                  Meaning by op_code:
//                    FNO1d:       [in_ch, out_ch, width, modes,       n_layers, pad_factor, _]
//                    FNO2d:       [in_ch, out_ch, width, modes_h, modes_w, n_layers, pad_factor]
//                    FNO3d:       [in_ch, out_ch, width, modes_h, modes_w, modes_d, n_layers, pad_factor]
//                    DeepONet:    [in_branch, in_trunk, latent_dim, out_channels,
//                                  n_layers_branch, n_layers_trunk, _]
//                    TokenMixer:  [in_dim, out_dim, latent_dim, n_points,
//                                  n_patches, n_heads, n_layers, _]
//                    GraphOp:     [in_dim, out_dim, n_nodes, hidden_dim, n_layers, _, _, _]
//                    TokenMixer2D:[in_dim, out_dim, latent_dim, h, w,
//                                  n_patches, n_heads, n_layers]
//                    GraphOp2D:   [in_dim, out_dim, h, w, hidden_dim, n_layers, _, _]
//   activation : uint8     = 0 (gelu) | 1 (relu)  (DeepONet / TokenMixer /
//                  GraphOp / TokenMixer2D / GraphOp2D use this byte too;
//                  the value is written from cfg.activation when present
//                  and defaults to gelu=0)
//   reserved2  : 3 bytes
//   n_weights  : uint32
//   for each weight:
//     name_len : uint8
//     name     : name_len bytes
//     ndim     : uint8
//     dims[]   : int32 * ndim
//     data     : float32 * numel
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "neuroflow/fno.h"
#include "neuroflow/runtime.h"
#include "neuroflow/tensor.h"

namespace nflow {

namespace ir_native {

inline Status LoadBinary(const std::string& path, LoadedModel& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return Status::FileNotFound("cannot open IR file: " + path);
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string buf = ss.str();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf.data());
    const size_t n = buf.size();

    auto read_u8 = [&](size_t& off, uint8_t& v) -> bool {
        if (off + 1 > n) return false;
        v = p[off++];
        return true;
    };
    auto read_i32 = [&](size_t& off, int32_t& v) -> bool {
        if (off + 4 > n) return false;
        std::memcpy(&v, p + off, 4);
        off += 4;
        return true;
    };
    auto read_u16 = [&](size_t& off, uint16_t& v) -> bool {
        if (off + 2 > n) return false;
        std::memcpy(&v, p + off, 2);
        off += 2;
        return true;
    };
    auto read_u32 = [&](size_t& off, uint32_t& v) -> bool {
        if (off + 4 > n) return false;
        std::memcpy(&v, p + off, 4);
        off += 4;
        return true;
    };

    size_t off = 0;
    if (n < 4 || std::memcmp(p, "NIR0", 4) != 0) {
        return Status::ParseError("bad magic in IR file");
    }
    off = 4;
    uint16_t version = 0;
    if (!read_u16(off, version)) return Status::ParseError("truncated header");
    if (version != 1 && version != 2 && version != 3) {
        return Status::ParseError("unsupported IR binary version");
    }

    uint8_t op_code = 0, reserved = 0;
    if (!read_u8(off, op_code)) return Status::ParseError("truncated header");
    if (!read_u8(off, reserved)) return Status::ParseError("truncated header");
    if (op_code == 0x01) {
        out.op = "FNO1d";
    } else if (op_code == 0x02) {
        out.op = "FNO2d";
    } else if (op_code == 0x03) {
        out.op = "FNO3d";
    } else if (op_code == 0x04) {
        out.op = "DeepONet";
    } else if (op_code == 0x05) {
        out.op = "TokenMixer";
    } else if (op_code == 0x06) {
        out.op = "GraphOp";
    } else if (op_code == 0x07) {
        out.op = "TokenMixer2D";
    } else if (op_code == 0x08) {
        out.op = "GraphOp2D";
    } else {
        return Status::UnsupportedOp("unknown op_code");
    }

    // Config block length: 7 for FNO1d / FNO2d / DeepONet, 8 for
    // FNO3d / TokenMixer / GraphOp / TokenMixer2D / GraphOp2D.
    int cfg_len = (op_code == 0x03 || op_code == 0x05 || op_code == 0x06 ||
                   op_code == 0x07 || op_code == 0x08)
                      ? 8
                      : 7;
    int32_t cfg_arr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < cfg_len; ++i) {
        if (!read_i32(off, cfg_arr[i])) return Status::ParseError("truncated config");
    }
    if (op_code == 0x01) {
        // FNO1d: [in_ch, out_ch, width, modes, n_layers, pad_factor, _]
        out.fno1d_cfg.in_channels = cfg_arr[0];
        out.fno1d_cfg.out_channels = cfg_arr[1];
        out.fno1d_cfg.width = cfg_arr[2];
        out.fno1d_cfg.modes = cfg_arr[3];
        out.fno1d_cfg.n_layers = cfg_arr[4];
        out.fno1d_cfg.pad_factor = cfg_arr[5];
        // cfg_arr[6] reserved
    } else if (op_code == 0x02) {
        // FNO2d
        out.fno2d_cfg.in_channels = cfg_arr[0];
        out.fno2d_cfg.out_channels = cfg_arr[1];
        out.fno2d_cfg.width = cfg_arr[2];
        out.fno2d_cfg.modes_h = cfg_arr[3];
        out.fno2d_cfg.modes_w = cfg_arr[4];
        out.fno2d_cfg.n_layers = cfg_arr[5];
        out.fno2d_cfg.pad_factor = cfg_arr[6];
    } else if (op_code == 0x03) {
        // FNO3d
        out.fno3d_cfg.in_channels = cfg_arr[0];
        out.fno3d_cfg.out_channels = cfg_arr[1];
        out.fno3d_cfg.width = cfg_arr[2];
        out.fno3d_cfg.modes_h = cfg_arr[3];
        out.fno3d_cfg.modes_w = cfg_arr[4];
        out.fno3d_cfg.modes_d = cfg_arr[5];
        out.fno3d_cfg.n_layers = cfg_arr[6];
        out.fno3d_cfg.pad_factor = cfg_arr[7];
    } else if (op_code == 0x05) {
        // TokenMixer: [in_dim, out_dim, latent_dim, n_points,
        //              n_patches, n_heads, n_layers, _]
        out.tokenmixer_cfg.in_dim = cfg_arr[0];
        out.tokenmixer_cfg.out_dim = cfg_arr[1];
        out.tokenmixer_cfg.latent_dim = cfg_arr[2];
        out.tokenmixer_cfg.n_points = cfg_arr[3];
        out.tokenmixer_cfg.n_patches = cfg_arr[4];
        out.tokenmixer_cfg.n_heads = cfg_arr[5];
        out.tokenmixer_cfg.n_layers = cfg_arr[6];
        // cfg_arr[7] reserved
    } else if (op_code == 0x06) {
        // GraphOp: [in_dim, out_dim, n_nodes, hidden_dim, n_layers, _, _, _]
        out.graphop_cfg.in_dim = cfg_arr[0];
        out.graphop_cfg.out_dim = cfg_arr[1];
        out.graphop_cfg.n_nodes = cfg_arr[2];
        out.graphop_cfg.hidden_dim = cfg_arr[3];
        out.graphop_cfg.n_layers = cfg_arr[4];
        // cfg_arr[5..7] reserved
    } else if (op_code == 0x07) {
        // TokenMixer2D: [in_dim, out_dim, latent_dim, h, w,
        //                n_patches, n_heads, n_layers]
        out.tokenmixer2d_cfg.in_dim = cfg_arr[0];
        out.tokenmixer2d_cfg.out_dim = cfg_arr[1];
        out.tokenmixer2d_cfg.latent_dim = cfg_arr[2];
        out.tokenmixer2d_cfg.h = cfg_arr[3];
        out.tokenmixer2d_cfg.w = cfg_arr[4];
        out.tokenmixer2d_cfg.n_patches = cfg_arr[5];
        out.tokenmixer2d_cfg.n_heads = cfg_arr[6];
        out.tokenmixer2d_cfg.n_layers = cfg_arr[7];
    } else if (op_code == 0x08) {
        // GraphOp2D: [in_dim, out_dim, h, w, hidden_dim, n_layers, _, _]
        out.graphop2d_cfg.in_dim = cfg_arr[0];
        out.graphop2d_cfg.out_dim = cfg_arr[1];
        out.graphop2d_cfg.h = cfg_arr[2];
        out.graphop2d_cfg.w = cfg_arr[3];
        out.graphop2d_cfg.hidden_dim = cfg_arr[4];
        out.graphop2d_cfg.n_layers = cfg_arr[5];
        // cfg_arr[6..7] reserved
    } else {
        // DeepONet: [in_branch, in_trunk, latent_dim, out_channels,
        //             n_layers_branch, n_layers_trunk, _]
        out.deeponet_cfg.in_branch = cfg_arr[0];
        out.deeponet_cfg.in_trunk = cfg_arr[1];
        out.deeponet_cfg.latent_dim = cfg_arr[2];
        out.deeponet_cfg.out_channels = cfg_arr[3];
        out.deeponet_cfg.n_layers_branch = cfg_arr[4];
        out.deeponet_cfg.n_layers_trunk = cfg_arr[5];
        // cfg_arr[6] reserved
    }

    uint8_t act = 0;
    if (!read_u8(off, act)) return Status::ParseError("truncated activation");
    std::string activation = (act == 1) ? "relu" : "gelu";
    if (op_code == 0x01) {
        out.fno1d_cfg.activation = activation;
    } else if (op_code == 0x02) {
        out.fno2d_cfg.activation = activation;
    } else if (op_code == 0x03) {
        out.fno3d_cfg.activation = activation;
    } else if (op_code == 0x05) {
        out.tokenmixer_cfg.activation = activation;
    } else if (op_code == 0x06) {
        out.graphop_cfg.activation = activation;
    } else if (op_code == 0x07) {
        out.tokenmixer2d_cfg.activation = activation;
    } else if (op_code == 0x08) {
        out.graphop2d_cfg.activation = activation;
    } else {
        out.deeponet_cfg.activation = activation;
    }
    off += 3;  // reserved2

    uint32_t n_weights = 0;
    if (!read_u32(off, n_weights)) return Status::ParseError("truncated n_weights");

    auto read_weight = [&](const std::string& name, Tensor& out_t) -> Status {
        uint8_t name_len = 0;
        if (!read_u8(off, name_len)) return Status::ParseError("truncated name_len");
        if (off + name_len > n) return Status::ParseError("truncated name");
        std::string read_name(reinterpret_cast<const char*>(p + off), name_len);
        off += name_len;
        if (read_name != name) {
            return Status::ParseError("weight name mismatch: " + read_name + " != " + name);
        }
        uint8_t ndim = 0;
        if (!read_u8(off, ndim)) return Status::ParseError("truncated ndim");
        std::vector<int64_t> shape(ndim);
        for (int i = 0; i < ndim; ++i) {
            int32_t d = 0;
            if (!read_i32(off, d)) return Status::ParseError("truncated dim");
            shape[i] = static_cast<int64_t>(d);
        }
        int64_t numel = 1;
        for (auto d : shape) numel *= d;
        size_t bytes = static_cast<size_t>(numel) * sizeof(float);
        if (off + bytes > n) return Status::ParseError("truncated data");
        // Copy out
        out_t = Tensor::Zeros(shape);
        std::memcpy(out_t.data(), p + off, bytes);
        off += bytes;
        return Status::Ok();
    };

    if (op_code == 0x01) {
        // Lifting
        if (auto s = read_weight("lift.weight", out.fno1d_weights.lift_w); !s.ok()) return s;
        if (auto s = read_weight("lift.bias", out.fno1d_weights.lift_b); !s.ok()) return s;

        out.fno1d_weights.spec_w_real.clear();
        out.fno1d_weights.spec_w_imag.clear();
        out.fno1d_weights.loc_w.clear();
        out.fno1d_weights.loc_b.clear();
        for (int i = 0; i < out.fno1d_cfg.n_layers; ++i) {
            Tensor wr, wi, lw, lb;
            if (auto s = read_weight("specs." + std::to_string(i) + ".weights_real", wr); !s.ok()) return s;
            if (auto s = read_weight("specs." + std::to_string(i) + ".weights_imag", wi); !s.ok()) return s;
            if (auto s = read_weight("locs." + std::to_string(i) + ".weight", lw); !s.ok()) return s;
            if (auto s = read_weight("locs." + std::to_string(i) + ".bias", lb); !s.ok()) return s;
            out.fno1d_weights.spec_w_real.push_back(std::move(wr));
            out.fno1d_weights.spec_w_imag.push_back(std::move(wi));
            out.fno1d_weights.loc_w.push_back(std::move(lw));
            out.fno1d_weights.loc_b.push_back(std::move(lb));
        }

        if (auto s = read_weight("proj_q.weight", out.fno1d_weights.proj_q_w); !s.ok()) return s;
        if (auto s = read_weight("proj_q.bias", out.fno1d_weights.proj_q_b); !s.ok()) return s;
        if (auto s = read_weight("proj_out.weight", out.fno1d_weights.proj_out_w); !s.ok()) return s;
        if (auto s = read_weight("proj_out.bias", out.fno1d_weights.proj_out_b); !s.ok()) return s;
    } else if (op_code == 0x02) {
        // FNO2d
        if (auto s = read_weight("lift.weight", out.fno2d_weights.lift_w); !s.ok()) return s;
        if (auto s = read_weight("lift.bias", out.fno2d_weights.lift_b); !s.ok()) return s;

        out.fno2d_weights.spec_w_real.clear();
        out.fno2d_weights.spec_w_imag.clear();
        out.fno2d_weights.loc_w.clear();
        out.fno2d_weights.loc_b.clear();
        for (int i = 0; i < out.fno2d_cfg.n_layers; ++i) {
            Tensor wr, wi, lw, lb;
            if (auto s = read_weight("specs." + std::to_string(i) + ".weights_real", wr); !s.ok()) return s;
            if (auto s = read_weight("specs." + std::to_string(i) + ".weights_imag", wi); !s.ok()) return s;
            if (auto s = read_weight("locs." + std::to_string(i) + ".weight", lw); !s.ok()) return s;
            if (auto s = read_weight("locs." + std::to_string(i) + ".bias", lb); !s.ok()) return s;
            out.fno2d_weights.spec_w_real.push_back(std::move(wr));
            out.fno2d_weights.spec_w_imag.push_back(std::move(wi));
            out.fno2d_weights.loc_w.push_back(std::move(lw));
            out.fno2d_weights.loc_b.push_back(std::move(lb));
        }

        if (auto s = read_weight("proj_q.weight", out.fno2d_weights.proj_q_w); !s.ok()) return s;
        if (auto s = read_weight("proj_q.bias", out.fno2d_weights.proj_q_b); !s.ok()) return s;
        if (auto s = read_weight("proj_out.weight", out.fno2d_weights.proj_out_w); !s.ok()) return s;
        if (auto s = read_weight("proj_out.bias", out.fno2d_weights.proj_out_b); !s.ok()) return s;
    } else if (op_code == 0x03) {
        // FNO3d
        if (auto s = read_weight("lift.weight", out.fno3d_weights.lift_w); !s.ok()) return s;
        if (auto s = read_weight("lift.bias", out.fno3d_weights.lift_b); !s.ok()) return s;

        out.fno3d_weights.spec_w_real.clear();
        out.fno3d_weights.spec_w_imag.clear();
        out.fno3d_weights.loc_w.clear();
        out.fno3d_weights.loc_b.clear();
        for (int i = 0; i < out.fno3d_cfg.n_layers; ++i) {
            Tensor wr, wi, lw, lb;
            if (auto s = read_weight("specs." + std::to_string(i) + ".weights_real", wr); !s.ok()) return s;
            if (auto s = read_weight("specs." + std::to_string(i) + ".weights_imag", wi); !s.ok()) return s;
            if (auto s = read_weight("locs." + std::to_string(i) + ".weight", lw); !s.ok()) return s;
            if (auto s = read_weight("locs." + std::to_string(i) + ".bias", lb); !s.ok()) return s;
            out.fno3d_weights.spec_w_real.push_back(std::move(wr));
            out.fno3d_weights.spec_w_imag.push_back(std::move(wi));
            out.fno3d_weights.loc_w.push_back(std::move(lw));
            out.fno3d_weights.loc_b.push_back(std::move(lb));
        }

        if (auto s = read_weight("proj_q.weight", out.fno3d_weights.proj_q_w); !s.ok()) return s;
        if (auto s = read_weight("proj_q.bias", out.fno3d_weights.proj_q_b); !s.ok()) return s;
        if (auto s = read_weight("proj_out.weight", out.fno3d_weights.proj_out_w); !s.ok()) return s;
        if (auto s = read_weight("proj_out.bias", out.fno3d_weights.proj_out_b); !s.ok()) return s;
    } else if (op_code == 0x05) {
        // TokenMixer
        if (auto s = read_weight("slice_embed.proj.weight",
                                  out.tokenmixer_weights.slice_embed_proj_w);
            !s.ok()) return s;
        if (auto s = read_weight("slice_embed.proj.bias",
                                  out.tokenmixer_weights.slice_embed_proj_b);
            !s.ok()) return s;

        out.tokenmixer_weights.blocks.clear();
        for (int i = 0; i < out.tokenmixer_cfg.n_layers; ++i) {
            fno::TokenMixerBlockWeights blk;
            const auto prefix = "blocks." + std::to_string(i) + ".";
            if (auto s = read_weight(prefix + "ln1.weight", blk.ln1_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ln1.bias", blk.ln1_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "q_proj.weight", blk.q_proj_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "q_proj.bias", blk.q_proj_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "k_proj.weight", blk.k_proj_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "k_proj.bias", blk.k_proj_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "v_proj.weight", blk.v_proj_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "v_proj.bias", blk.v_proj_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "o_proj.weight", blk.o_proj_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "o_proj.bias", blk.o_proj_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ln2.weight", blk.ln2_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ln2.bias", blk.ln2_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ffn0.weight", blk.ffn0_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ffn0.bias", blk.ffn0_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ffn1.weight", blk.ffn1_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ffn1.bias", blk.ffn1_b); !s.ok()) return s;
            out.tokenmixer_weights.blocks.push_back(std::move(blk));
        }
        if (auto s = read_weight("unslice.proj.weight",
                                  out.tokenmixer_weights.unslice_proj_w);
            !s.ok()) return s;
        if (auto s = read_weight("unslice.proj.bias",
                                  out.tokenmixer_weights.unslice_proj_b);
            !s.ok()) return s;
        if (auto s = read_weight("head.weight", out.tokenmixer_weights.head_w);
            !s.ok()) return s;
        if (auto s = read_weight("head.bias", out.tokenmixer_weights.head_b);
            !s.ok()) return s;
    } else if (op_code == 0x06) {
        // GraphOp
        if (auto s = read_weight("lift.weight", out.graphop_weights.lift_w);
            !s.ok()) return s;
        if (auto s = read_weight("lift.bias", out.graphop_weights.lift_b);
            !s.ok()) return s;

        out.graphop_weights.blocks.clear();
        for (int i = 0; i < out.graphop_cfg.n_layers; ++i) {
            fno::GraphOpBlockWeights blk;
            const auto prefix = "blocks." + std::to_string(i) + ".";
            if (auto s = read_weight(prefix + "lin_self.weight", blk.lin_self_w);
                !s.ok()) return s;
            if (auto s = read_weight(prefix + "lin_self.bias", blk.lin_self_b);
                !s.ok()) return s;
            if (auto s = read_weight(prefix + "lin_neigh.weight", blk.lin_neigh_w);
                !s.ok()) return s;
            if (auto s = read_weight(prefix + "lin_neigh.bias", blk.lin_neigh_b);
                !s.ok()) return s;
            out.graphop_weights.blocks.push_back(std::move(blk));
        }
        if (auto s = read_weight("head.weight", out.graphop_weights.head_w);
            !s.ok()) return s;
        if (auto s = read_weight("head.bias", out.graphop_weights.head_b);
            !s.ok()) return s;

        // Graph topology — stored as float32 weights in the IR; cast
        // back to int32 / float32 here.
        Tensor adj_off_t, adj_idx_t, deg_inv_t;
        if (auto s = read_weight("graph.adj_offsets", adj_off_t); !s.ok()) return s;
        if (auto s = read_weight("graph.adj_indices", adj_idx_t); !s.ok()) return s;
        if (auto s = read_weight("graph.deg_inv", deg_inv_t); !s.ok()) return s;
        out.graphop_weights.adj_offsets.resize(adj_off_t.numel());
        out.graphop_weights.adj_indices.resize(adj_idx_t.numel());
        out.graphop_weights.deg_inv.resize(deg_inv_t.numel());
        for (int64_t i = 0; i < adj_off_t.numel(); ++i) {
            out.graphop_weights.adj_offsets[i] = static_cast<int32_t>(adj_off_t.data()[i]);
        }
        for (int64_t i = 0; i < adj_idx_t.numel(); ++i) {
            out.graphop_weights.adj_indices[i] = static_cast<int32_t>(adj_idx_t.data()[i]);
        }
        std::memcpy(out.graphop_weights.deg_inv.data(), deg_inv_t.data(),
                    static_cast<size_t>(deg_inv_t.numel()) * sizeof(float));
    } else if (op_code == 0x07) {
        // TokenMixer2D — weight layout mirrors 1D TokenMixer.
        if (auto s = read_weight("slice_embed.proj.weight",
                                  out.tokenmixer2d_weights.slice_embed_proj_w);
            !s.ok()) return s;
        if (auto s = read_weight("slice_embed.proj.bias",
                                  out.tokenmixer2d_weights.slice_embed_proj_b);
            !s.ok()) return s;

        out.tokenmixer2d_weights.blocks.clear();
        for (int i = 0; i < out.tokenmixer2d_cfg.n_layers; ++i) {
            fno::TokenMixerBlockWeights blk;
            const auto prefix = "blocks." + std::to_string(i) + ".";
            if (auto s = read_weight(prefix + "ln1.weight", blk.ln1_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ln1.bias", blk.ln1_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "q_proj.weight", blk.q_proj_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "q_proj.bias", blk.q_proj_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "k_proj.weight", blk.k_proj_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "k_proj.bias", blk.k_proj_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "v_proj.weight", blk.v_proj_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "v_proj.bias", blk.v_proj_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "o_proj.weight", blk.o_proj_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "o_proj.bias", blk.o_proj_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ln2.weight", blk.ln2_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ln2.bias", blk.ln2_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ffn0.weight", blk.ffn0_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ffn0.bias", blk.ffn0_b); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ffn1.weight", blk.ffn1_w); !s.ok()) return s;
            if (auto s = read_weight(prefix + "ffn1.bias", blk.ffn1_b); !s.ok()) return s;
            out.tokenmixer2d_weights.blocks.push_back(std::move(blk));
        }
        if (auto s = read_weight("unslice.proj.weight",
                                  out.tokenmixer2d_weights.unslice_proj_w);
            !s.ok()) return s;
        if (auto s = read_weight("unslice.proj.bias",
                                  out.tokenmixer2d_weights.unslice_proj_b);
            !s.ok()) return s;
        if (auto s = read_weight("head.weight", out.tokenmixer2d_weights.head_w);
            !s.ok()) return s;
        if (auto s = read_weight("head.bias", out.tokenmixer2d_weights.head_b);
            !s.ok()) return s;
    } else if (op_code == 0x08) {
        // GraphOp2D — weight layout mirrors 1D GraphOp.
        if (auto s = read_weight("lift.weight", out.graphop2d_weights.lift_w);
            !s.ok()) return s;
        if (auto s = read_weight("lift.bias", out.graphop2d_weights.lift_b);
            !s.ok()) return s;

        out.graphop2d_weights.blocks.clear();
        for (int i = 0; i < out.graphop2d_cfg.n_layers; ++i) {
            fno::GraphOpBlockWeights blk;
            const auto prefix = "blocks." + std::to_string(i) + ".";
            if (auto s = read_weight(prefix + "lin_self.weight", blk.lin_self_w);
                !s.ok()) return s;
            if (auto s = read_weight(prefix + "lin_self.bias", blk.lin_self_b);
                !s.ok()) return s;
            if (auto s = read_weight(prefix + "lin_neigh.weight", blk.lin_neigh_w);
                !s.ok()) return s;
            if (auto s = read_weight(prefix + "lin_neigh.bias", blk.lin_neigh_b);
                !s.ok()) return s;
            out.graphop2d_weights.blocks.push_back(std::move(blk));
        }
        if (auto s = read_weight("head.weight", out.graphop2d_weights.head_w);
            !s.ok()) return s;
        if (auto s = read_weight("head.bias", out.graphop2d_weights.head_b);
            !s.ok()) return s;

        Tensor adj_off_t, adj_idx_t, deg_inv_t;
        if (auto s = read_weight("graph.adj_offsets", adj_off_t); !s.ok()) return s;
        if (auto s = read_weight("graph.adj_indices", adj_idx_t); !s.ok()) return s;
        if (auto s = read_weight("graph.deg_inv", deg_inv_t); !s.ok()) return s;
        out.graphop2d_weights.adj_offsets.resize(adj_off_t.numel());
        out.graphop2d_weights.adj_indices.resize(adj_idx_t.numel());
        out.graphop2d_weights.deg_inv.resize(deg_inv_t.numel());
        for (int64_t i = 0; i < adj_off_t.numel(); ++i) {
            out.graphop2d_weights.adj_offsets[i] = static_cast<int32_t>(adj_off_t.data()[i]);
        }
        for (int64_t i = 0; i < adj_idx_t.numel(); ++i) {
            out.graphop2d_weights.adj_indices[i] = static_cast<int32_t>(adj_idx_t.data()[i]);
        }
        std::memcpy(out.graphop2d_weights.deg_inv.data(), deg_inv_t.data(),
                    static_cast<size_t>(deg_inv_t.numel()) * sizeof(float));
    } else {
        // DeepONet
        out.deeponet_weights.branch.weight.clear();
        out.deeponet_weights.branch.bias.clear();
        for (int i = 0; i < out.deeponet_cfg.n_layers_branch; ++i) {
            Tensor w, b;
            if (auto s = read_weight("branch.layers." + std::to_string(i) + ".weight", w); !s.ok()) return s;
            if (auto s = read_weight("branch.layers." + std::to_string(i) + ".bias", b); !s.ok()) return s;
            out.deeponet_weights.branch.weight.push_back(std::move(w));
            out.deeponet_weights.branch.bias.push_back(std::move(b));
        }
        out.deeponet_weights.trunk.weight.clear();
        out.deeponet_weights.trunk.bias.clear();
        for (int i = 0; i < out.deeponet_cfg.n_layers_trunk; ++i) {
            Tensor w, b;
            if (auto s = read_weight("trunk.layers." + std::to_string(i) + ".weight", w); !s.ok()) return s;
            if (auto s = read_weight("trunk.layers." + std::to_string(i) + ".bias", b); !s.ok()) return s;
            out.deeponet_weights.trunk.weight.push_back(std::move(w));
            out.deeponet_weights.trunk.bias.push_back(std::move(b));
        }
        if (auto s = read_weight("bias.weight", out.deeponet_weights.bias); !s.ok()) return s;

        // hidden_branch / hidden_trunk are inferred from the second-to-last
        // weight's output dim (i.e. the last hidden layer's out dim). When
        // n_layers_branch == 1, there is no hidden layer and the field
        // is left at 0; callers should not read it in that case.
        if (out.deeponet_cfg.n_layers_branch >= 2) {
            const auto& w = out.deeponet_weights.branch.weight[out.deeponet_cfg.n_layers_branch - 2];
            out.deeponet_cfg.hidden_branch = static_cast<int>(w.shape()[0]);
        }
        if (out.deeponet_cfg.n_layers_trunk >= 2) {
            const auto& w = out.deeponet_weights.trunk.weight[out.deeponet_cfg.n_layers_trunk - 2];
            out.deeponet_cfg.hidden_trunk = static_cast<int>(w.shape()[0]);
        }
    }

    // Optional NIRQ (quantisation) trailing block — v0.15.0
    // (per-tensor) and v0.16.0 (per-channel weights).
    // Older files (v0.1.0 - v0.13.0) end after the last
    // weight; the loader returns Ok() above without ever
    // trying to read trailing bytes.  When the trailing
    // "NIRQ" magic is present, parse the block.
    if (off + 4 <= n && std::memcmp(p + off, "NIRQ", 4) == 0) {
        off += 4;
        uint8_t quant_flag = 0;
        if (!read_u8(off, quant_flag)) return Status::ParseError("truncated quant_flag");
        if (quant_flag != 1) {
            return Status::ParseError("unsupported quant_flag");
        }
        out.quant_enabled = true;
        uint32_t n_qparams = 0;
        if (!read_u32(off, n_qparams)) return Status::ParseError("truncated n_qparams");
        for (uint32_t i = 0; i < n_qparams; ++i) {
            uint8_t kind = 0;
            if (!read_u8(off, kind)) return Status::ParseError("truncated qparam kind");
            uint8_t name_len = 0;
            if (!read_u8(off, name_len)) return Status::ParseError("truncated qparam name_len");
            if (off + name_len > n) return Status::ParseError("truncated qparam name");
            std::string qp_name(reinterpret_cast<const char*>(p + off), name_len);
            off += name_len;
            if (kind == 0) {
                // Per-tensor (v0.15.0)
                float scale = 0.0f;
                int32_t zp = 0;
                if (off + 8 > n) return Status::ParseError("truncated qparam scale/zp");
                std::memcpy(&scale, p + off, 4); off += 4;
                std::memcpy(&zp, p + off, 4); off += 4;
                QuantParams qp{scale, zp};
                if (qp_name.size() >= 7 &&
                    qp_name.compare(qp_name.size() - 7, 7, ".output") == 0) {
                    out.activation_qparams[qp_name] = qp;
                } else {
                    out.weight_qparams[qp_name] = qp;
                }
            } else if (kind == 1) {
                // Per-channel (v0.16.0)
                uint32_t n_ch = 0;
                uint8_t axis = 0;
                if (off + 5 > n) return Status::ParseError("truncated per-channel header");
                std::memcpy(&n_ch, p + off, 4); off += 4;
                std::memcpy(&axis, p + off, 1); off += 1;
                size_t scales_bytes = static_cast<size_t>(n_ch) * 4;
                size_t zps_bytes = static_cast<size_t>(n_ch) * 4;
                if (off + scales_bytes + zps_bytes > n) {
                    return Status::ParseError("truncated per-channel data");
                }
                PerChannelQuantParams pcp;
                pcp.channel_axis = static_cast<int32_t>(axis);
                pcp.scales.resize(n_ch);
                pcp.zero_points.resize(n_ch);
                for (uint32_t c = 0; c < n_ch; ++c) {
                    std::memcpy(&pcp.scales[c], p + off, 4);
                    off += 4;
                }
                for (uint32_t c = 0; c < n_ch; ++c) {
                    std::memcpy(&pcp.zero_points[c], p + off, 4);
                    off += 4;
                }
                out.weight_per_channel_qparams[qp_name] = pcp;
            } else if (kind == 2) {
                // Per-token activation (v0.17.0)
                uint32_t n_tok = 0;
                uint32_t width = 0;
                if (off + 8 > n) return Status::ParseError("truncated per-token header");
                std::memcpy(&n_tok, p + off, 4); off += 4;
                std::memcpy(&width, p + off, 4); off += 4;
                size_t scales_bytes = static_cast<size_t>(n_tok) * 4;
                size_t zps_bytes = static_cast<size_t>(n_tok) * 4;
                if (off + scales_bytes + zps_bytes > n) {
                    return Status::ParseError("truncated per-token data");
                }
                PerTokenQuantParams ptp;
                ptp.width = static_cast<int32_t>(width);
                ptp.scales.resize(n_tok);
                ptp.zero_points.resize(n_tok);
                for (uint32_t c = 0; c < n_tok; ++c) {
                    std::memcpy(&ptp.scales[c], p + off, 4);
                    off += 4;
                }
                for (uint32_t c = 0; c < n_tok; ++c) {
                    std::memcpy(&ptp.zero_points[c], p + off, 4);
                    off += 4;
                }
                out.activation_per_token_qparams[qp_name] = ptp;
            } else if (kind == 3) {
                // FP8 E4M3 per-tensor (v0.21.0)
                if (off + 4 > n) return Status::ParseError(
                    "truncated fp8 scale");
                FP8E4M3Params fp8p;
                std::memcpy(&fp8p.scale, p + off, 4);
                off += 4;
                out.activation_fp8_qparams[qp_name] = fp8p;
            } else {
                return Status::ParseError("unsupported qparam kind");
            }
        }
    }

    return Status::Ok();
}

}  // namespace ir_native

}  // namespace nflow
