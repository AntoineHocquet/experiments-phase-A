# Pillar 3: Domain decomposition (`schwarz/`)

A large region is cut into pieces, each piece is solved on its own, and the
pieces are stitched at the seams, exactly how classical parallel solvers (Schwarz,
FETI) and engineers assembling a structure from parts proceed. Pillar 3 asks
whether a model trained on canonical pieces, plus a learned interface
(Dirichlet-to-Neumann, DtN) rule, can assemble solutions on new geometries
without retraining. The Schwarz convergence rate is itself controlled by the
Laplace-Beltrami spectrum of Pillar 2, one of the cross-pillar links.

Pillar 3 is represented here by one clean result; its further tests (many
subdomains, indefinite operators, time-dependent problems) belong to the larger
E4 package.

## Shared code

- `models.py`: the spectral convolution block. The overlap-conditioned subdomain
  Fourier operators and the learned DtN consensus that build on it are defined in
  `run_e4e.py`.
- `overlap.py`: the classical overlapping-Schwarz reference (subdomain solves,
  grids, right-hand-side sampling, and the contraction-rate measurement) against
  which the learned iteration is compared.
- `poisson.py`: the exact 1D Poisson tridiagonal solve used as the reference.

## Experiment kept here

- **E4e** (`run_e4e.py`, `results_e4e.md`). Two subdomains are stitched with a
  learned DtN rule, and the overlap width between them is varied. Classical theory
  (Toselli-Widlund) predicts that more overlap makes the stitch-and-repeat process
  converge faster; the learned rule reproduces that classical convergence curve to
  about 1% at every overlap, with the same slope. A credibility result: the learned
  component obeys the established convergence law rather than doing something ad
  hoc. The honest framing (see the narrative) is that this validates convergence
  rate, partly by construction, not end-to-end assembled accuracy.

These experiments are drawn from the larger E4 domain-decomposition package
(which also covers many-subdomain, indefinite-Helmholtz, time-dependent, and 2D
settings); reviewers can be granted access to the complete package on request.

## Running

```bash
python schwarz/run_e4e.py --smoke --out_dir /tmp/e4e   # quick check
python schwarz/run_e4e.py --device cuda                # full run
```

The frozen figure, `params.txt`, and raw JSON are under `reports/e4e/`.
