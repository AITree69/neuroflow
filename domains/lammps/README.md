# NeuroFlow × LAMMPS — First Domain SDK

This is Stage 2's first domain SDK.  It demonstrates how a
NeuroFlow operator (in this Sprint, an FNO1d) can be
trained as a *neural surrogate* for a classical
inter-atomic potential and then used as a drop-in
replacement inside a molecular-dynamics (MD) loop, with
both the energy $V(r)$ and the force
$F(r) = -\mathrm{d}V / \mathrm{d}r$ queried through the same
NeuroIR + C++ runtime that the rest of the Stage 2
evaluation uses.

The Sprint 3.8 scope is intentionally narrow:

  1. **Domain SDK architecture** — a small Python package
     (`neuroflow_lammps`) that defines the integration
     contract between NeuroFlow and an MD driver.  No
     actual LAMMPS linkage is shipped in this Sprint (that
     is a Stage 3 item — the SDK is designed so the
     LAMMPS-side `fix nflow` plugin becomes a thin shim
     around the same contract).

  2. **1D Morse potential demo** — train an FNO1d to
     fit the classical 1D Morse potential
     $V(r) = D_e (1 - e^{-a(r - r_e)})^2 - D_e$, then
     evaluate energy and force accuracy on a held-out grid
     and compare the C++ runtime to the analytical
     formula.

  3. **End-to-end C++ inference** — drive the exported
     FNO1d through `neuroflow_cpp.infer_arrays` and
     confirm that the energy + force numbers match the
     PyTorch reference within the same $<\! 10^{-3}$
     target that the rest of Stage 2 holds.

  4. **Paper §6** — a new section in the paper that
     documents the SDK architecture, the surrogate-fit
     numbers, and the take-away that the same
     operator-family-agnostic pipeline that closed the
     Stage 2 cross-op benchmarks is what makes this
     domain integration possible.

## Layout

```
domains/lammps/
  README.md                       (this file)
  neuroflow_lammps/
    __init__.py                   (public Python API)
    surrogate.py                  (SurrogatePotential helper that
                                  wraps a NeuroFlow model +
                                  energy / force interface)
    driver.py                     (Minimal MD driver loop that
                                  queries the surrogate instead
                                  of a classical pair potential)
  examples/
    17_morse_surrogate.py        (train + export + C++ eval
                                  end-to-end demo)
  tests/
    test_morse_surrogate.py       (pytest cases for the SDK and
                                  the demo)
```

## Why Morse (and not LJ)?

The 1D Morse potential is a single 1D curve, which fits
naturally into the FNO1d input / output shape.  Lennard-Jones
needs at least 2D geometry (radial distance); we leave the
3D and many-body extensions for a later Sprint, but the
SDK shape does not change.

The Morse potential has three parameters $D_e, a, r_e$
that we can sweep over to synthesise a varied training set,
without needing a real MD trajectory as input.
