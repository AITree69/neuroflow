// =============================================================================
// Standalone harness for the NeuroFlow LAMMPS `fix nflow` shim
// =============================================================================
//
// Mimics what LAMMPS would do — call `FixNflow::init` and
// `FixNflow::compute` per step — so the shim can be
// tested without LAMMPS installed.
//
// Usage:
//   test_fix_nflow_standalone <model.nneuroir> [n_atoms] [n_steps] [dt]
//
// Default args: n_atoms=4, n_steps=16, dt=0.01.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "neuroflow/lammps/fix_nflow.h"

namespace {

double simple_lj(double r2, double sigma, double epsilon) {
    // Standard Lennard-Jones potential (for sanity-checking
    // the shim's force output).  Not used for inference —
    // the shim uses the NeuroFlow surrogate.  This is
    // a reference for tests.
    if (r2 < 1e-12) return 0.0;
    const double sr6 = std::pow(sigma * sigma / r2, 3);
    return 4.0 * epsilon * (sr6 * sr6 - sr6);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <model.nneuroir> [n_atoms] [n_steps] [dt]\n",
            argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    int n_atoms = (argc >= 3) ? std::atoi(argv[2]) : 4;
    int n_steps = (argc >= 4) ? std::atoi(argv[3]) : 16;
    double dt = (argc >= 5) ? std::atof(argv[4]) : 0.01;

    std::printf("=== Standalone harness for NeuroFlow `fix nflow` ===\n");
    std::printf("  model = %s\n", model_path);
    std::printf("  n_atoms = %d, n_steps = %d, dt = %.4f\n",
                n_atoms, n_steps, dt);

    nflow::lammps_shim::FixNflow fix;
    nflow::Status s = fix.init(model_path, 8);
    if (!s.ok()) {
        std::fprintf(stderr, "FixNflow::init failed: %s\n",
                     s.message().c_str());
        return 1;
    }
    std::printf("  loaded: op=%s, n_layers=%d, width=%d\n",
                fix.op().c_str(), fix.n_layers(), fix.width());
    std::printf("\n=== MD loop (velocity Verlet, surrogate forces) ===\n");

    // Simple MD state: positions, velocities, forces.
    // We use a 3D system (n_atoms in a row along x) so
    // the per-atom force is along x.
    nflow::lammps_shim::AtomBlock atom;
    atom.nlocal = n_atoms;
    atom.ndim = 3;
    atom.positions.assign(n_atoms * 3, 0.0);
    atom.forces.assign(n_atoms * 3, 0.0);
    // Initial positions: a small lattice along x.
    for (int i = 0; i < n_atoms; ++i) {
        atom.positions[i * 3 + 0] = 0.5 * i;
        atom.positions[i * 3 + 1] = 0.0;
        atom.positions[i * 3 + 2] = 0.0;
    }
    std::vector<double> velocities(n_atoms * 3, 0.0);
    // Small initial velocity for the first atom.
    if (n_atoms > 0) velocities[0] = 0.1;

    double prev_energy = 0.0;
    for (int step = 0; step < n_steps; ++step) {
        double energy = 0.0;
        s = fix.compute(atom, &energy);
        if (!s.ok()) {
            std::fprintf(stderr,
                "FixNflow::compute failed at step %d: %s\n",
                step, s.message().c_str());
            return 1;
        }
        if (step % 4 == 0 || step == n_steps - 1) {
            std::printf(
                "  step %3d: E = %12.6e, "
                "F[0] = (%9.3e, %9.3e, %9.3e), "
                "x[0] = %9.3e\n",
                step, energy,
                atom.forces[0], atom.forces[1], atom.forces[2],
                atom.positions[0]);
        }
        // Velocity Verlet (mass = 1).
        // v(t + dt/2) = v(t) + 0.5 * dt * F(t) / m
        // x(t + dt)   = x(t) + dt * v(t + dt/2)
        // F(t + dt)   = compute at x(t + dt)
        // v(t + dt)   = v(t + dt/2) + 0.5 * dt * F(t + dt) / m
        for (int i = 0; i < n_atoms * 3; ++i) {
            velocities[i] += 0.5 * dt * atom.forces[i];
        }
        for (int i = 0; i < n_atoms * 3; ++i) {
            atom.positions[i] += dt * velocities[i];
        }
        s = fix.compute(atom, &energy);
        if (!s.ok()) {
            std::fprintf(stderr,
                "FixNflow::compute (2nd) failed at step %d: %s\n",
                step, s.message().c_str());
            return 1;
        }
        for (int i = 0; i < n_atoms * 3; ++i) {
            velocities[i] += 0.5 * dt * atom.forces[i];
        }
        if (step == 0) prev_energy = energy;
    }

    std::printf("\n=== Run completed ===\n");
    std::printf("  final E = %12.6e\n", prev_energy);
    std::printf("  final x[0] = %12.6e\n", atom.positions[0]);
    return 0;
}
