# Pillar 1: Operator splitting (`commutator/`)

A complicated evolution can often be solved by alternating a few simple,
separately understood moves. Pillar 1 asks whether a neural operator can learn
those elementary moves once and then *recompose* them, by classical operator
splitting, to handle new combinations without retraining.

The relevant theory is the Baker-Campbell-Hausdorff (BCH) formula in the linear
regime, where the splitting error of a composition scheme (Lie-Trotter, Strang)
is controlled by the commutator norm of the generators, and the
Lie-Trotter / Crandall-Liggett-Brezis-Pazy theory in the nonlinear regime, where
the clean BCH guarantee weakens to a softer one. Each elementary move is learned
as a *parameter-conditioned* brick (the proposal's Splitting-Mixture-of-Experts
backbone): a brick sees the field, the step size, and its own physical
coefficient, so that one network represents a whole operator family.

## Shared code

- `core.py`: the parameter-conditioned 1D Fourier neural operator (`FNO1d`), the
  brick / dataset generators, and the training and evaluation utilities used
  across the splitting experiments.
- `semilinear.py`: the nonlinear reaction brick and the semilinear
  reaction-diffusion reference solver (used by E2e).

## Experiments kept here

- **E2b** (`run_e2b.py`, `results_e2b.md`). One network is trained only on the
  two elementary moves separately (pure diffusion, pure advection) at zero shear,
  where Strang splitting is algebraically exact. Composing the same network by
  hand (half diffusion, full advection, half diffusion) drops the relative error
  about 50x versus a single direct call, with no extra training. The cleanest 1D
  demonstration of the building-block idea, in the regime where the splitting
  algebra contributes no error of its own.

- **E2d** (`run_e2d.py`, `results_e2d.md`). The generator-coverage form of the
  pillar: covering the elementary moves the target problem is built from improves
  transfer. Adding a third brick (a reaction move) to the training menu lowers
  zero-shot error on a new equation (Allen-Cahn with drift) by 2.6x, and the
  three-brick curriculum beats the two-brick one at four of five fine-tuning
  budgets.

- **E2e** (`run_e2e.py`, `results_e2e.md`). The nonlinear regime, where the
  proposal's fallback theory is Crandall-Liggett-Brezis-Pazy rather than BCH. On
  reaction-diffusion with a tunable reaction stiffness, a two-brick composition
  beats a monolithic model on both data efficiency and accuracy once the reaction
  is non-trivial, and the margin grows with stiffness.

These experiments are drawn from the larger E2 operator-splitting package;
reviewers can be granted access to the complete package on request.

## Running

```bash
python commutator/run_e2d.py --smoke --out_dir /tmp/e2d   # E2d and E2e take --smoke
python commutator/run_e2d.py --device cuda                # full run
```

`run_e2b.py` predates the `--smoke` flag and takes explicit small sizes (see
the repository `run_all.py`). All hyperparameters are exposed via `--help`.
The frozen figures, `params.txt`, and raw JSON for each experiment are under
`reports/<exp>/`.
