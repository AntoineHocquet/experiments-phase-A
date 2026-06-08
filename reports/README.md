# Reports: frozen experiment artifacts

This folder holds the canonical, version-controlled outputs of each named run:
the figure, the exact command line, and the raw result JSON. It is the
counterpart of the gitignored, dynamic `results_*/` folders, which are
regenerated every time a script is run locally.

## Layout

One subfolder per kept experiment, each shipping the same four files:

```
reports/<exp>/
├── <exp>_<name>.pdf    vector figure (for the proposal and papers)
├── <exp>_<name>.png    raster preview (for inline rendering in markdown viewers)
├── <exp>_raw.json      parameters, headline numbers, and timing
└── params.txt          the exact command line used to produce the artifacts
```

Kept experiments:

- Pillar 1: `e2b/`, `e2d/`, `e2e/`
- Pillar 2: `e3a/`, `e3b_1d/`, `e3b_2d/`, `e3d/`, `e3f/`, `e3g/`, `e3h/`, `e3i/`
- Pillar 3: `e4e/`
- Pillar 1 x Pillar 2 bridge: `e5a/`

Because each folder records the exact command line in `params.txt`, every figure
here is reproducible from a clean checkout.

## Where to read the narrative

The headline numbers and interpretation for each plot live next to the script
that produced it, in the per-pillar folders:

- E2b, E2d, E2e -> `commutator/results_<exp>.md`
- E3a, E3b 1D, E3b 2D -> `geometry/results.md`
- E3d, E3f, E3g, E3h, E3i -> `geometry/results_<exp>.md`
- E4e -> `schwarz/results_e4e.md`
- E5a -> `commutator_geometry/results_e5a.md`

The plain-language tour across all of them is `book_of_experiments.md` at the
repository root.
