// =============================================================================
// NeuroFlow LAMMPS `fix nflow` shim — implementation
// =============================================================================

#include "neuroflow/lammps/fix_nflow.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "neuroflow/runtime.h"

namespace nflow {
namespace lammps_shim {

FixNflow::FixNflow(const std::string& model_path,
                      int n_points) {
    Status s = init(model_path, n_points);
    if (!s.ok()) {
        std::fprintf(stderr, "FixNflow: init failed: %s\n",
                     s.message().c_str());
    }
}

Status FixNflow::init(const std::string& model_path,
                       int n_points) {
    model_path_ = model_path;
    Status s = LoadNeuroIr(model_path, model_);
    if (!s.ok()) return s;
    s = InferenceRuntime::Create(model_path, runtime_);
    if (!s.ok()) return s;
    if (model_.op != "FNO1d") {
        return Status::InvalidArg(
            "FixNflow currently supports FNO1d only; got op='" +
            model_.op + "'");
    }
    // Per-atom surrogate: each atom's input is
    // (1, n_points, in_channels) where in_channels =
    // ndim (3) + feat_dim_.  n_points is the length
    // of the 1D "spatial" axis.  We require
    // n_points to be a power of two (FNO1d constraint).
    if (n_points <= 0 || (n_points & (n_points - 1)) != 0) {
        return Status::InvalidArg(
            "FixNflow: n_points must be a power of two, got " +
            std::to_string(n_points));
    }
    n_points_ = n_points;
    in_channels_ = 3 + feat_dim_;
    out_channels_ = 1;
    in_buf_.assign(static_cast<size_t>(n_points_) * in_channels_, 0.0f);
    out_buf_.assign(static_cast<size_t>(n_points_) * out_channels_, 0.0f);
    return Status::Ok();
}

Status FixNflow::compute(AtomBlock& atom, double* energy) {
    if (!runtime_) {
        return Status::InvalidArg("FixNflow::init not called");
    }
    if (atom.nlocal <= 0) {
        if (energy) *energy = 0.0;
        return Status::Ok();
    }
    const int64_t nlocal = atom.nlocal;
    const int ndim = atom.ndim;
    if (in_channels_ != ndim + feat_dim_) {
        return Status::InvalidArg(
            "FixNflow: in_channels mismatch (expected " +
            std::to_string(ndim + feat_dim_) + ", got " +
            std::to_string(in_channels_) + ").  Did you call "
            "set_atom_features?");
    }
    if (atom.positions.size() != static_cast<size_t>(nlocal * ndim)) {
        return Status::ShapeMismatch(
            "FixNflow: positions size mismatch");
    }
    if (atom.features.size() != 0 &&
        atom.features.size() != static_cast<size_t>(nlocal * feat_dim_)) {
        return Status::ShapeMismatch(
            "FixNflow: features size mismatch");
    }
    if (atom.forces.size() != static_cast<size_t>(nlocal * ndim)) {
        atom.forces.assign(static_cast<size_t>(nlocal * ndim), 0.0);
    }
    // The shim computes the per-atom force via 5-point
    // central finite difference on the per-atom
    // surrogate.  This is the same as the Python
    // `MorseSurrogate.force` (Sprint 3.8).
    //
    // For each atom i, we perturb the position by ±h
    // along each dimension, evaluate the surrogate
    // (h = 1e-3, in units of the surrogate's input
    // scale; we use h = 1e-4 in the input's natural
    // units, scaled to the model's expected range).
    //
    // Stage 3 extension: the surrogate may be cached
    // (one surrogate call per (atom, dim) perturbation
    // = 10 calls per atom = 10 * nlocal surrogate
    // calls per step).  The cache is trivial: each
    // atom's surrogate is independent.
    const float fd_step = 1e-3f;
    double total_energy = 0.0;
    for (int64_t i = 0; i < nlocal; ++i) {
        // Build the (1, n_points, in_channels) input
        // for the unperturbed atom.  Each point along
        // the spatial axis gets the same per-atom
        // feature vector.  This is the standard
        // "broadcast" pattern for per-atom surrogates.
        auto build_input = [&](int64_t atom_i,
                               std::vector<float>& buf,
                               int dim = -1,
                               float delta = 0.0f) {
            for (int64_t p = 0; p < n_points_; ++p) {
                for (int d = 0; d < ndim; ++d) {
                    float v = static_cast<float>(
                        atom.positions[atom_i * ndim + d]);
                    if (d == dim) v += delta;
                    buf[p * in_channels_ + d] = v;
                }
                for (int f = 0; f < feat_dim_; ++f) {
                    buf[p * in_channels_ + ndim + f] =
                        static_cast<float>(
                            atom.features[atom_i * feat_dim_ + f]);
                }
            }
        };
        // Forward pass to get the per-atom energy at
        // the unperturbed position.
        const std::vector<int64_t> x_shape = {1, n_points_, in_channels_};
        const std::vector<int64_t> y_shape = {1, n_points_, out_channels_};
        out_buf_.assign(out_buf_.size(), 0.0f);
        build_input(i, in_buf_);
        Status s = runtime_->Run(
            in_buf_.data(), x_shape,
            out_buf_.data(), y_shape);
        if (!s.ok()) return s;
        const float e0 = out_buf_[0];
        total_energy += e0;
        // 5-point central FD per dimension.
        for (int d = 0; d < ndim; ++d) {
            // +2h
            build_input(i, in_buf_, d, 2.0f * fd_step);
            s = runtime_->Run(
                in_buf_.data(), x_shape,
                out_buf_.data(), y_shape);
            if (!s.ok()) return s;
            const float e_p2h = out_buf_[0];
            // +h
            build_input(i, in_buf_, d, fd_step);
            s = runtime_->Run(
                in_buf_.data(), x_shape,
                out_buf_.data(), y_shape);
            if (!s.ok()) return s;
            const float e_ph = out_buf_[0];
            // -h
            build_input(i, in_buf_, d, -fd_step);
            s = runtime_->Run(
                in_buf_.data(), x_shape,
                out_buf_.data(), y_shape);
            if (!s.ok()) return s;
            const float e_mh = out_buf_[0];
            // -2h
            build_input(i, in_buf_, d, -2.0f * fd_step);
            s = runtime_->Run(
                in_buf_.data(), x_shape,
                out_buf_.data(), y_shape);
            if (!s.ok()) return s;
            const float e_m2h = out_buf_[0];
            // 5-point central FD: f' = (e_{-2h} - 8 e_{-h}
            // + 8 e_{+h} - e_{+2h}) / (12 h).
            const float force = (e_m2h - 8.0f * e_mh +
                                  8.0f * e_ph - e_p2h) /
                                (12.0f * fd_step);
            // F = -dE/dx
            atom.forces[i * ndim + d] = -force;
        }
    }
    if (energy) *energy = total_energy;
    return Status::Ok();
}

}  // namespace lammps_shim
}  // namespace nflow
