// =============================================================================
// NeuroFlow LAMMPS `fix nflow` shim — header
// =============================================================================
//
// Implements the LAMMPS `fix` interface contract so that
// NeuroFlow surrogate inference can be plugged into a
// LAMMPS MD loop as `fix nflow` without linking the
// full NeuroFlow C++ runtime into LAMMPS — only this
// shim is linked.
//
// The shim uses the *NeuroFlow standalone runtime API*:
//   - `nflow::load_neuroflow(path, model)` to load a
//     `.nneuroir` model
//   - `model.infer(x, y)` to run inference
//
// The standalone harness (`tests/test_fix_nflow_standalone.cpp`)
// mimics what LAMMPS would do — call `FixNflow::init`,
// then `FixNflow::compute(nlocal, x, f, energy)` per step —
// so the shim can be tested without LAMMPS installed.
//
// To plug into LAMMPS:
//   1. Drop `fix_nflow.cpp` into LAMMPS's `src/EXTRA-FIX`
//      directory (or compile as a plugin via Kokkos).
//   2. Implement the LAMMPS `Fix` class — methods
//      `init()`, `setup()`, `post_force()` — by
//      delegating to `FixNflow`.
//   3. `lammps -in in.fix_nflow` runs MD with the
//      NeuroFlow surrogate.
//
// IMPORTANT: this shim is *LAMMPS-agnostic* — it
// implements the FixNflow contract but does NOT depend
// on the LAMMPS source tree.  The LAMMPS-side binding
// is a separate ~50-line shim that subclasses the
// LAMMPS `Fix` class and forwards to FixNflow.
// =============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "neuroflow/runtime.h"

namespace nflow {
namespace lammps_shim {

// In-memory atom block (the only thing the shim needs
// from the LAMMPS atom-style API).  In a real LAMMPS
// integration, `Atom::x[i][dim]` is a `double**`; the
// shim copies the relevant slice into the
// `AtomBlock` struct once per step.
struct AtomBlock {
    int64_t nlocal = 0;            // number of local atoms
    int ndim = 3;                  // spatial dimensions (3 for 3D)
    // Flat (nlocal * ndim) row-major storage.  x[i, d] =
    // positions[i * ndim + d].  We use `double` to
    // match LAMMPS's storage type.
    std::vector<double> positions;       // (nlocal, ndim)
    std::vector<double> forces;          // (nlocal, ndim) — output
    // Per-atom feature (e.g. atom type index, charge) is
    // encoded as a flat `features` array of length
    // `nlocal`.  The shim concatenates it onto the
    // per-atom feature vector that the surrogate
    // expects.
    std::vector<double> features;
};

// `FixNflow` is the LAMMPS-agnostic contract that a
// LAMMPS `fix nflow` would delegate to.  The contract
// is intentionally small:
//
//   init(path)               — load the .nneuroir model
//   set_atom_features(feat_dim)
//                           — declare the per-atom feature dim
//   compute(atom, energy)    — one MD step:
//                              1. build the per-atom input
//                                 tensor from atom.positions
//                                 and atom.features
//                              2. run inference → per-atom
//                                 energy (nlocal, 1)
//                              3. sum to scalar energy
//                              4. compute per-atom force
//                                 = -dE/dr via 5-point FD
//                                 (Stage 3: a single
//                                 surrogate call per atom;
//                                 the FD is the same as
//                                 the Python reference
//                                 `MorseSurrogate`).
//                              5. write to atom.forces
//
// The shim does NOT use the LAMMPS `Fix` class hierarchy
// or depend on `lammps.h`; that binding is a separate
// ~50-line `LAMMPS_NS::FixNflow` adapter.
class FixNflow {
public:
    // Construct without loading a model.  Call `init`
    // before `compute`.
    FixNflow() = default;

    // Construct + load the model from `model_path`.
    // `n_points` is the length of the per-atom
    // 1D "spatial" axis; it must match the model's
    // expected input length (a power of two for
    // FNO1d).  Default = 8.
    explicit FixNflow(const std::string& model_path,
                       int n_points = 8);

    // Load a `.nneuroir` model from disk.  Returns
    // `Status::Ok()` on success; the shim is ready for
    // `compute` after this returns.
    Status init(const std::string& model_path,
                 int n_points = 8);

    // Declare the per-atom feature dim.  Must be called
    // before `compute` if `AtomBlock::features` is
    // non-empty.  The shim concatenates the atom
    // features onto the position input.
    void set_atom_features(int feat_dim) {
        feat_dim_ = feat_dim;
    }

    // The current model.  Useful for inspecting the
    // loaded config / weights in tests.
    const LoadedModel& model() const { return model_; }
    const std::string& model_path() const { return model_path_; }

    // The supported operation.  Returns "FNO1d" (the
    // shim is currently FNO1d-only; extending to FNO2d
    // / FNO3d / DeepONet is a Stage 3 extension).
    const std::string& op() const { return model_.op; }

    // Number of spectral conv layers in the loaded
    // FNO1d (read from cfg).  Useful for tests.
    int n_layers() const { return model_.fno1d_cfg.n_layers; }

    // The width (number of channels after the lift
    // layer).  Useful for tests.
    int width() const { return model_.fno1d_cfg.width; }

    // The model's input n_points (length of the 1D
    // "spatial" axis).  The shim broadcasts the per-
    // atom feature vector to this length.
    int n_points() const { return n_points_; }

    // The model's input channel count.  Must equal
    // ndim (3) + feat_dim_.
    int in_channels() const { return in_channels_; }

    // Per-atom feature dim (default 0).
    int feat_dim() const { return feat_dim_; }

    // One MD step.  Reads `atom.positions` and
    // `atom.features`, computes per-atom forces, writes
    // to `atom.forces`, and returns the scalar total
    // potential energy via `*energy`.
    //
    // The shim assumes the model is an FNO1d that maps
    // (nlocal, n_points, in_channels) → (nlocal, n_points,
    // 1) — i.e. a 1D surrogate per atom with `n_points`
    // 1D samples along the "spatial" axis.  This is the
    // standard "Morse potential surrogate" pattern
    // (Sprint 3.8): each atom has a 1D feature vector
    // (its local environment reduced to a 1D signal),
    // the surrogate outputs a 1D potential, and the
    // forces are computed by 5-point FD on the
    // surrogate's per-atom output.
    Status compute(AtomBlock& atom, double* energy);

private:
    std::string model_path_;
    LoadedModel model_;
    std::unique_ptr<InferenceRuntime> runtime_;
    int feat_dim_ = 0;
    // Cached scratch buffers for the per-atom input /
    // output tensors.  We keep one (1, n_points,
    // in_channels) buffer and reuse it across calls.
    std::vector<float> in_buf_;
    std::vector<float> out_buf_;
    int64_t n_points_ = 0;       // 1D surrogate "spatial" axis
    int in_channels_ = 0;       // ndim (3) + feat_dim_
    int out_channels_ = 0;      // always 1 for energy
};

}  // namespace lammps_shim
}  // namespace nflow
