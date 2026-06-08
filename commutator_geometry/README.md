# Cross-pillar bridge, Pillar 1 x Pillar 2 (`commutator_geometry/`)

The pillars are designed to combine. The linear diffusion brick of Pillar 1 has
a closed-form expression in the Laplace-Beltrami eigenbasis of Pillar 2: the
diffusion semigroup is diagonal in that basis. So Pillar 2 supplies an *exact*,
parameter-free diffusion move that can be dropped into Pillar 1's composition in
place of a learned brick.

## Experiment kept here

- **E5a** (`run_e5a.py`, `results_e5a.md`). The learned diffusion brick inside
  Pillar 1's Strang composition is replaced by Pillar 2's exact closed-form
  diffusion semigroup (the advection brick stays learned), and the composition is
  compared against a monolithic model across a range of shear strengths. At
  moderate-to-high shear the exact-diffusion composition wins where the
  all-learned composition loses: removing one source of error tips the balance.
  A concrete demonstration that the pillars help each other. See the narrative for
  the caveats (the effect appears at high shear, not at zero as pre-registered,
  and the strongest-shear point rests on a single seed).

E5a reuses `commutator.core` (the parameter-conditioned FNO, dataset generators,
training and evaluation helpers) and `shared.splitting.strang`; the closed-form
brick is `shared.simulators.brick_diffusion` wrapped to the splitting interface.

A companion cross-geometry brick-library experiment (E5b) is part of the same
larger package; reviewers can be granted access to the complete package on
request.

## Running

```bash
python commutator_geometry/run_e5a.py --smoke --out_dir /tmp/e5a   # quick check
python commutator_geometry/run_e5a.py --device cuda                # full run
```

The frozen figure, `params.txt`, and raw JSON are under `reports/e5a/`.
