# Geometry-aware PDE foundation models: Phase A proofs of concept

This repository collects small, self-contained numerical experiments that
de-risk the central research bet behind a pretrained neural-operator framework
for industrial partial differential equations (PDEs). Each experiment tests one
falsifiable claim of one architectural pillar, at modest compute cost (minutes to
a few hours of GPU time), before any large-scale pretraining is committed.

## The global approach

Classical finite-element solvers are accurate but expensive, which limits their
use in real-time and many-query settings (design-space exploration, online
monitoring, digital twins). Neural operators learn a solution map at a fraction
of the inference cost, but most are trained on narrow problem classes and do not
transfer to the diverse geometries and physics of industrial practice.

Our thesis is that engineers already decompose hard PDE problems along three
largely independent axes, and that each axis is a *theorem* for classical
solvers rather than a conjecture:

- **operatorially**, by operator splitting (Lie-Trotter, Strang, Yoshida);
- **spectrally**, by eigenfunction bases (Fourier, Laplace-Beltrami);
- **spatially**, by domain decomposition (Schwarz, FETI).

The research question is therefore narrow and testable: not *whether* the
decomposition exists (it provably does), but *whether a neural operator can
learn it* so that pretrained building blocks recompose into solutions for new
problems instead of being retrained from scratch. The three pillars below mirror
the three axes one-to-one. They are designed to combine, and the cross-pillar
folders test those algebraic links directly.

## The three pillars

- **Pillar 1, operator splitting** (`commutator/`). A complicated evolution
  (for example "heat diffuses *and* the material drifts") can be solved by
  alternating a few simple, separately understood moves. The bet: a model that
  has learned the elementary moves can be *composed* to handle new combinations,
  instead of retraining. The theory is the Baker-Campbell-Hausdorff (BCH)
  formula in the linear regime and Lie-Trotter / Crandall-Liggett-Brezis-Pazy in
  the nonlinear regime.

- **Pillar 2, geometry-adaptive spectral basis** (`geometry/`). Every shape has
  its own natural vibration modes (a square drum and a round drum ring at
  different frequencies). Fixed-grid spectral solvers assume a rectangular
  domain. The bet: a small model that *predicts a shape's Laplace-Beltrami
  spectrum* from a compact geometric descriptor gives one surrogate that adapts
  to new shapes and sizes instead of being locked to one geometry.

- **Pillar 3, domain decomposition** (`schwarz/`). A large region is cut into
  pieces, each piece is solved on its own, and the pieces are stitched at the
  seams. The bet: a model trained on canonical pieces, plus a learned interface
  (Dirichlet-to-Neumann) rule, assembles solutions on new, larger geometries
  without retraining. The Schwarz convergence rate is itself controlled by the
  Laplace-Beltrami spectrum of Pillar 2, which is one of the cross-pillar links.

Two cross-pillar folders test how the pillars combine: `commutator_geometry/`
(Pillar 1 x Pillar 2) substitutes Pillar 2's exact diffusion semigroup into
Pillar 1's composition.

## Scope of this repository

The experiments here are drawn from three larger experiment packages, one per
pillar: **E2** (Pillar 1, operator splitting), **E3** (Pillar 2, spectral
geometry), and **E4** (Pillar 3, domain decomposition), together with the
cross-pillar bridges (E5). Reviewers can be granted access to the complete
packages on request.

What is kept here:

| Exp. | Pillar | One-line result |
|------|--------|-----------------|
| E2b  | 1 (deployment)        | The same network, composed by hand at `s = 0`, drops error about 50x for free |
| E2e  | 1 (nonlinear)         | Composing two specialised pieces beats a monolithic model on stiff reaction-diffusion |
| E2d  | 1 (curriculum)        | Covering the missing reaction generator improves zero-shot transfer 2.6x |
| E3a  | 2 (scaling law)       | A small encoder learns the eigenvalue scaling law and extrapolates beyond its training box |
| E3b  | 2 (spectral basis)    | An LB-FNO on predicted modes generalises across sizes where a flat-Fourier FNO fails (1D and 2D) |
| E3i  | 2 (3D)                | The encoder extends to 3D boxes and predicts unseen sizes to within 5% on all 10 leading modes |
| E3g  | 2 (new shape family)  | Learns a triangle's spectrum to within 5% on every mode |
| E3h  | 2 (shape with a hole) | Learns how a circular hole shifts the spectrum, matching the Rayleigh perturbation |
| E3d  | 2 (sharp corner)      | Captures most of an L-shape's spectrum; the re-entrant-corner singularity is the residual |
| E3f  | 2 (curved shapes)     | The shape-matched basis represents a disk where a fixed grid cannot |
| E4e  | 3 (stitching)         | The learned interface rule obeys the classical overlap-convergence law to about 1% |
| E5a  | 1 x 2 bridge          | Plugging Pillar 2's exact diffusion move into Pillar 1's composition wins on the harder problems |

The plain-language story, with figures, compute cost, and network sizes, is in
[`book_of_experiments.md`](book_of_experiments.md). Per-experiment narratives and
headline numbers live in the `results_<exp>.md` file next to each script, and the
frozen figures and raw JSON are under [`reports/`](reports/).

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `torch>=2.0`, `numpy`, `scipy`, `matplotlib`. No GPU is required
for the smoke tests; the full runs were produced on an NVIDIA A40 and run on any
modern data-centre GPU (for example RTX 4090, A100, or A40).

## Running

Every script accepts `--device {auto,cuda,cpu}` (auto picks cuda when available)
and `--out_dir <path>` (default `results_<expid>`). Nine of the scripts
(E2d, E2e, E3d, E3f, E3g, E3h, E3i, E4e, E5a) additionally accept:

```
--smoke           tiny CPU smoke test, finishes in seconds to a minute
--seed <int>      default 0
```

A quick smoke test of any single experiment, for example:

```bash
python geometry/run_e3i.py --smoke --out_dir /tmp/e3i_smoke
```

The four foundational scripts (`run_e2b` for Pillar 1; `run_e3a`, `run_e3b`,
`run_e3b_2d` for Pillar 2) predate the `--smoke`/`--seed` flags and instead take
explicit small sizes; [`run_all.py`](run_all.py) wires those up. To smoke-test
every kept experiment in sequence:

```bash
python run_all.py --smoke      # all experiments, tiny sizes, CPU
python run_all.py              # full sizes (hours; benefits from a GPU)
```

Each run writes three files into its `--out_dir`: a vector figure (`.pdf`), a
raster preview (`.png`), and a `<exp>_raw.json` with the parameters, headline
numbers, and timing. The committed copies under `reports/<exp>/` are the frozen
versions of those files.

## Reproducing the frozen results

Every figure under `reports/<exp>/` ships with a `params.txt` recording the
exact command line used to produce it, so each result is reproducible from a
clean checkout. The runs were executed on an NVIDIA A40 (and CPU for the
two-subdomain Schwarz experiment). All experiments pass a CPU smoke test from a
fresh `requirements.txt` environment.

## Directory structure

```
.
├── README.md                     this file
├── book_of_experiments.md        plain-language tour of the kept experiments
├── requirements.txt
├── run_all.py                    run every kept experiment (--smoke or full)
├── Taskfile.yml                  task clean / smoke / all
├── reports/                      frozen artifacts (PDF + PNG + params + raw JSON)
├── shared/                       simulators, splitting schemes, run utilities
├── commutator/                   Pillar 1: operator splitting (E2b, E2d, E2e)
├── geometry/                     Pillar 2: spectral geometry (E3a, E3b, E3b 2D, E3d, E3f, E3g, E3h, E3i)
├── schwarz/                      Pillar 3: domain decomposition (E4e)
└── commutator_geometry/          Pillar 1 x Pillar 2 bridge (E5a)
```
