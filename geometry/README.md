# Pillar 2: Geometry-adaptive spectral basis (`geometry/`)

Every shape has its own natural vibration modes, the eigenpairs of the
Laplace-Beltrami (LB) operator on the domain. Fixed-grid spectral solvers assume
a rectangular domain and a flat-Fourier basis. Pillar 2 asks whether a small
model can *predict a shape's LB spectrum* from a compact geometric descriptor, so
that one surrogate adapts to new shapes and sizes instead of being tied to one
geometry, and whether those predicted modes make a better coordinate system for a
neural operator than a fixed grid.

An important nuance running through these experiments: the encoder predicts the
spectrum *within the family it has been shown a little of*. Where the underlying
relation is a genuine scaling *law* (rectangles, boxes), it extrapolates well
beyond the training range; for a genuinely new shape family it needs a small
in-family sample. Zero-shot transfer across shape families from a scalar
descriptor is studied separately, in the larger E3 package.

## Shared code

- `encoder.py`: the eigenvalue encoder, a small MLP that maps a geometric
  descriptor to the leading eigenvalues, regressed in log coordinates so that the
  analytic scaling laws become affine (an inductive bias an MLP extrapolates).
- `lb_truth.py`: analytic LB eigenvalues for intervals, rectangles, and boxes.
- `disk_modes.py`, `lshape_modes.py`, `perforated.py`, `triangle_modes.py`:
  eigenpair helpers (closed-form Bessel for disks; finite-element eigensolvers for
  L-shapes, perforated rectangles, and triangles) used as ground truth.

## Experiments kept here

Foundational scaling-law and spectral-basis results:

- **E3a** (`run_e3a.py`, `results.md`). The encoder learns the LB eigenvalue
  scaling law as a geometry-invariant law rather than a per-shape lookup, on 1D
  intervals and 2D rectangles, and the low-error region extends well beyond the
  training box.
- **E3b 1D / 2D** (`run_e3b.py`, `run_e3b_2d.py`, `results.md`). An LB-FNO built
  on the predicted eigenvalues generalises across domain sizes (and, in 2D, aspect
  ratios) where a flat-Fourier FNO trained on a single geometry fails.

Generalisations to harder geometries:

- **E3i** (`run_e3i.py`, `results_e3i.md`). The encoder extends to 3D boxes and
  predicts the spectrum of boxes larger than any seen in training, all ten leading
  modes under 5% error (mean about 2%). Genuine size extrapolation in 3D.
- **E3g** (`run_e3g.py`, `results_e3g.md`). A non-product shape family: a
  rectangle-only model is uniformly wrong on triangles, but after a short exposure
  to a small triangle library the encoder reaches 5% on every mode across the
  family.
- **E3h** (`run_e3h.py`, `results_e3h.md`). A multiply-connected shape: a circular
  hole raises the spectrum. A hole-blind model degrades with hole area exactly as
  the Rayleigh-quotient perturbation predicts, while a hole-aware encoder learns
  the shift to 5% on all modes.
- **E3d** (`run_e3d.py`, `results_e3d.md`). A non-convex, singular geometry: an
  L-shape with a re-entrant corner. A cut-aware encoder captures most of the
  spectrum (6 of 10 modes under 5%); the residual is concentrated in the lowest
  mode, which the corner singularity distorts most.
- **E3f** (`run_e3f.py`, `results_e3f.md`). Curved shapes: on heat flow in disks
  of varying radius, the geometry-adapted (LB) basis represents the solution where
  a flat-Fourier FNO cannot, and feeding the flat FNO more disk sizes does not fix
  it. The citable claim is the basis thesis; see the narrative for the caveat on
  the oracle-eigenvalue comparison.

These experiments are drawn from the larger E3 spectral-geometry package;
reviewers can be granted access to the complete package on request.

## Running

All use closed-form or finite-element ground truth; no external solver is
required. The scripts `run_e3d`, `run_e3f`, `run_e3g`, `run_e3h`, and `run_e3i`
take `--smoke`; the foundational `run_e3a` / `run_e3b` / `run_e3b_2d` take
explicit small sizes (see the repository `run_all.py`).

```bash
python geometry/run_e3i.py --smoke --out_dir /tmp/e3i   # quick check
python geometry/run_e3i.py --device cuda                # full run
```

The frozen figures, `params.txt`, and raw JSON for each experiment are under
`reports/<exp>/`.
