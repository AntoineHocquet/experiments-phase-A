# The Book of Experiments

### Phase A proofs of concept for the Geometry-aware PDE foundation models approach

---

## What this book is

The approach rests on three independent ideas ("pillars") for building one fast,
reusable surrogate model for the partial differential equations (PDEs) that
engineers solve every day (heat flow, diffusion, structural and acoustic
problems, and so on). Before committing to a large training run, each pillar is
de-risked with self-contained numerical experiments, each testing one falsifiable
claim in minutes to hours of compute.

The experiments are drawn from three larger packages, one per pillar (E2, E3,
E4), together with the cross-pillar bridges (E5); reviewers can be granted access
to the complete packages on request. Each experiment here comes with a
plain-language explanation a non-specialist (with a Bachelor-level grasp of
mathematics) can follow, the figure it produced, the exact compute cost, and the
precise size and shape of the neural network involved. Every result is stated
together with the caveats that bound it (for example, that it was demonstrated on
an easier setting, that it needs a small amount of problem-specific data, or that
it was run on a single random seed).

If you read only three results, read the **core pillar experiments** (E2.b, E3i,
E4e), one per pillar, collected at the end under
[Core pillar experiments](#core-pillar-experiments).

## The three pillars, in plain words

- **Pillar 1, operator splitting.** A complicated evolution (say "heat spreads
  *and* the material drifts") can be solved by alternating a few simple, separately
  understood "moves" (a pure-diffusion move, a pure-drift move), like solving a
  Rubik's cube with a known sequence of turns. The bet: a model that has learned
  the simple moves can be *composed* to handle new combinations, instead of
  retraining from scratch.

- **Pillar 2, geometry-adaptive basis.** Every shape has its own natural vibration
  patterns (think of how a square drum and a round drum ring at different
  frequencies). Classical fast solvers assume a fixed rectangular grid. The bet: if
  a small model can *predict the shape's natural frequencies* from a compact
  description of the shape, one surrogate adapts to new shapes and sizes instead of
  being locked to one geometry.

- **Pillar 3, domain decomposition.** A big region is cut into pieces; each piece is
  solved on its own and the pieces are stitched together at the seams (exactly how
  engineers assemble a structure from parts, and how classical parallel solvers
  work). The bet: a model trained on canonical pieces plus a learned "stitching"
  rule assembles solutions on new, larger geometries without retraining.

## At a glance

A star (*) marks the three core pillar experiments, one per pillar, presented in
full at the end.

| Exp.   | Pillar              | One-line result                                                        |
|--------|---------------------|------------------------------------------------------------------------|
| *E2.b  | 1 (deployment)      | The same network, composed by hand, drops error ~50x for free          |
| E2e    | 1 (nonlinear)       | Composing beats the big model on the harder, nonlinear problems        |
| E2d    | 1 (curriculum)      | Adding the missing "move" to training improves transfer 2.6x           |
| *E3i   | 2 (scaling, 3D)     | The shape-frequency model extends to 3D and predicts unseen box sizes  |
| E3g    | 2 (new shape family)| Learns a triangle's frequencies to within 5% on every mode             |
| E3h    | 2 (shape with a hole)| Learns how a hole shifts the frequencies, exactly as theory predicts  |
| *E4e   | 3 (stitching)       | The learned "stitch" obeys the textbook convergence law to ~1%         |
| E5a    | 1 x 2 bridge        | Plugging an exact diffusion move into the chain wins on hard problems  |
| E3d    | 2 (sharp corner)    | Mostly learns an L-shape's frequencies; the sharp corner is the hard part |
| E3f    | 2 (round shapes)    | The shape-matched basis represents a disk where a fixed grid cannot    |

The three starred experiments are presented together at the end
([Core pillar experiments](#core-pillar-experiments)).

---

# Pillar 1: Operator splitting

The aim for Pillar 1 is to learn a small library of elementary "moves" once, then
*reassemble* them to solve new equations cheaply. The three experiments below show
this paying off in three complementary ways: composing a trained network by hand
at use-time (E2.b), covering the right moves during training (E2d), and keeping the
advantage in the genuinely nonlinear regime that industrial problems live in (E2e).

## E2.b: the same network, composed by hand, for free  (core Pillar 1 result)

> **Takeaway.** One network was trained only on the two simple moves separately
> (pure diffusion, pure drift). Asked to predict the combined process, calling it
> once gives 60% error; calling the *same* network three times in the classical
> "half-move, full-move, half-move" order gives ~1% error. A ~50x improvement, with
> no extra training and not a single new parameter.

**What we tested, in plain words.** Imagine a network that has seen water spreading
(diffusion) and, separately, a current carrying things along (drift), but has never
seen both happening at once. We then ask it to predict both at once. There are two
ways: ask it directly (it has to guess how to combine the two), or compose by hand
using a recipe from classical mathematics (Strang splitting): do half a diffusion
step, then a full drift step, then another half diffusion step. When the current is
uniform in space, this recipe is *exactly* correct on paper, so any remaining error
is purely the network's own. The experiment runs both ways and compares to a network
trained from scratch on the combined process.

**What we found.** The hand-composed prediction (~0.012 relative error) is about
50 times more accurate than the direct call (~0.605), and it matches or beats a
from-scratch network at every training-data budget, using zero extra training. When
the mathematics says the composition is exact, the learned pieces compose almost
perfectly: direct evidence that the building-block idea of Pillar 1 holds. This is a
best-case setting by construction (uniform current, one spatial dimension), which is
what makes it the right first claim to establish.

**The figure.** A log-scale plot of relative error against training-set size: the
from-scratch curve sits well above a flat horizontal line marking the hand-composed
result, which uses no training data of its own.

![E2.b: Strang composition at zero shear](reports/e2b/e2b_strang_commuting.png)

**Which pillar (and which form).** Pillar 1, the *deployment-time* form: compose the
learned elementary moves at use-time. This is the operational core of Conjecture 1
(the generator-learning hypothesis), demonstrated in the regime where the splitting
algebra is exact.

**Compute.**
- Module / CLI: `python commutator/run_e2b.py --device cuda --out_dir results_e2b`
- Resolved settings: `--n_pretrain 10000 --n_pretrain_epochs 800 --ns 200 500 1000 2000 5000 --n_finetune_epochs 500 --n_seeds 3 --nx 128 --width 96 --n_modes 32 --n_layers 4 --batch_size 128`
- Hardware / time: GPU (CUDA), 2026-05-28; about 30 minutes.

**The network.** One shared spectral network (a 1D Fourier neural operator),
conditioned on the step size and the two physical coefficients. **1,229,921
trainable parameters.**

```
FNO1d  (1,229,921 params)
  lift     : Conv1d(4 -> 96, kernel 1)          # input = field + 3 conditioning channels
  spectral : 4 x SpectralConv1d(width 96, 32 modes)
  local    : 4 x Conv1d(96 -> 96, kernel 1)
  proj     : Conv1d(96 -> 128) -> GELU -> Conv1d(128 -> 1)
```

## E2e: composition wins on the harder, nonlinear problems

> **Takeaway.** On reaction-diffusion (diffusion plus a nonlinear chemical-style
> reaction), composing two specialised pieces beats a single big model trained on
> the whole problem, and the advantage *grows* as the reaction gets stiffer (up to
> ~2x fewer training examples and ~2-3x more accurate).

**What we tested, in plain words.** Real industrial equations are usually nonlinear,
where the clean textbook theory of Pillar 1 weakens to a softer guarantee (the
Crandall-Liggett-Brezis-Pazy theorem). We test exactly that regime:
diffusion plus a nonlinear reaction term whose strength we dial up with a knob
(gamma). We compare a two-piece composition (a diffusion piece and a small
reaction piece) against one monolithic model trained on the full problem, and ask
which needs fewer examples to reach a given accuracy.

**What we found.** For any non-trivial reaction strength the composition wins on
both counts (data needed and accuracy), and the margin *grows* with stiffness.
The mechanism is instructive: factorising the problem keeps the hard, stiff part in
a small dedicated piece, while a single model trying to learn the entangled whole
degrades faster as the problem hardens. This is the first data-efficiency win for
Pillar 1 that is not confined to tiny training budgets, and it lands in the
nonlinear regime that matters for applications.

**The figure.** Left: the data-efficiency gain crossing the break-even line and
rising with reaction strength. Right: absolute errors, where the composed curve
stays low while the monolithic curve climbs. The run used a single random seed, so
the exact numbers carry some noise.

![E2e: semilinear regime, composition vs monolithic](reports/e2e/e2e_semilinear_clbp.png)

**Which pillar (and which form).** Pillar 1, the *nonlinear* form: the proposal's
fallback theory for nonlinear PDEs, shown to still favour composition.

**Compute.**
- Module / CLI: `python commutator/run_e2e.py --device cuda --out_dir results_e2e`
- Resolved settings: `--gammas 0.5 2.0 5.0 --n_brick 4000 --n_joint 8000 --budgets 1000 2000 4000 8000 16000 --n_epochs 250 --nx 128 --width 64 --n_modes 32 --n_layers 4 --batch 128 --mlp_hidden 32 --mlp_layers 3`
- Hardware / time: NVIDIA A40, 1,196.7 s (~20 min).

**The networks.** A diffusion piece (Fourier operator, **549,633 params**), a tiny
per-point reaction piece (**2,273 params**), and the monolithic baseline (Fourier
operator, **549,697 params**).

```
diffusion FNO1d (549,633)        reaction MLP (2,273)            monolithic FNO1d (549,697)
  lift  : Conv1d(3 -> 64)          net: Conv1d(3 -> 32) -> GELU     lift : Conv1d(4 -> 64)
  4 x SpectralConv1d(64, 32)            -> Conv1d(32 -> 32) -> GELU  4 x SpectralConv1d(64,32)
  4 x Conv1d(64 -> 64)                  -> Conv1d(32 -> 32) -> GELU  4 x Conv1d(64 -> 64)
  proj  : 64 -> 128 -> 1                -> Conv1d(32 -> 1)           proj : 64 -> 128 -> 1
```

## E2d: cover the right "moves" and transfer improves

> **Takeaway.** Adding a third elementary move (a reaction piece) to the training
> menu makes a single model transfer to a new combined equation 2.6x better with no
> fine-tuning, and consistently better at almost every training budget.

**What we tested, in plain words.** Pillar 1 reframes "what should we train on?" as a
coverage question: include the elementary moves the target problem is built from. We
train one model on two moves (diffusion, drift) versus three moves (diffusion, drift,
reaction), then test both on a new equation (Allen-Cahn with drift) that genuinely
contains all three. The question is simply whether covering the missing move helps.

**What we found.** It helps, most cleanly with no fine-tuning at all: the three-move
model reproduces the unseen target at 2.6x lower error than the two-move model, and
once a little fine-tuning data is added the three-move version is ahead at four of
five budgets. This is direct evidence for the mechanism the proposal names:
downstream performance tracks how well the training menu covers the target's
ingredients. The caveat: on this easy 1D problem, the advantage over a from-scratch
model is concentrated at small data budgets.

**The figure.** Left: the from-scratch and two/three-move fine-tuning curves, plus
the two horizontal "no fine-tuning" markers (the three-move marker sits far below the
two-move one). Right: the per-budget improvement.

![E2d: three-brick curriculum on Allen-Cahn-with-drift](reports/e2d/e2d_three_brick_curriculum.png)

**Which pillar (and which form).** Pillar 1, the *generator-coverage / curriculum*
form of Conjecture 1.

**Compute.**
- Module / CLI: `python commutator/run_e2d.py --device cuda --out_dir results_e2d`
- Resolved settings: `--n_pretrain 10000 --n_pretrain_epochs 800 --ns 200 500 1000 2000 5000 --n_finetune_epochs 500 --n_seeds 3 --shear 0.5 --nx 128 --width 96 --n_modes 32 --n_layers 4 --batch_size 128`
- Hardware / time: NVIDIA A40, 3,745.8 s (~62 min).

**The network.** A single shape-of-E2.b Fourier operator conditioned on the step size
and physical coefficients. **1,229,921 trainable parameters** (same architecture as
E2.b above).

---

# Pillar 2: Geometry-adaptive spectral basis

This is the strongest pillar in the set. The unifying claim across these five
experiments: a small model can predict a shape's natural frequencies from a compact
description of the shape, and those predicted frequencies make a far better
coordinate system for a PDE solver than a fixed grid. The experiments stress this
across new sizes (E3i), genuinely new shape families (E3g triangles), shapes with
holes (E3h), shapes with sharp re-entrant corners (E3d), and curved shapes where a
rectangular grid fundamentally struggles (E3f).

A shared, important nuance: the encoder predicts frequencies *within the family it
has been shown a little of*. None of these claim free transfer to a brand-new family
with zero examples (that limit is real and is recorded elsewhere). What they show is
that, given a sensible description of the shape and a small set of examples, the
approach works, and that where it learns a genuine scaling *law* it extrapolates well
beyond what it was trained on.

## E3i: the idea scales to three dimensions  (core Pillar 2 result)

> **Takeaway.** The frequency-predicting model extends from rectangles to 3D boxes
> and predicts the frequencies of boxes *larger than any it was trained on* to within
> ~2% (all 10 leading modes under 5%).

**What we tested, in plain words.** For a box, the natural frequencies follow an exact
law in terms of the side lengths. We train the small model on boxes of moderate size
and then ask it for boxes with one side stretched beyond the training range. Because
the model is built to learn the *law* (it works in logarithmic coordinates, where the
law becomes a straight line), the real question is whether it has internalised the law
rather than memorising a table.

**What we found.** It extrapolates: every one of the ten leading frequencies is
predicted to under 5% error on out-of-range boxes (mean ~2%), and a 2D slice of the
accuracy map shows the "under 5%" region extending well beyond the training box. This
is the smallest honest test of "the geometry idea scales to 3D," and it passes. The
caveats: only one side was stretched at a time, the box is the easiest (separable)
shape, and the model's design makes this particular law straightforward to extrapolate
by construction.

**The figure.** Left: a bar chart with all ten modes under the 5% line. Right: a
heat-map of accuracy with a white "training box" and the 5% contour reaching far
outside it: trained inside the box, still accurate far outside it.

![E3i: 3D box eigenvalue prediction and extrapolation](reports/e3i/e3i_box_eigenvalues.png)

**Which pillar (and which form).** Pillar 2, the *scaling-law encoder* form, extended
to 3D (the same class of result as the rectangle/size experiments E3a/E3b).

**Compute.**
- Module / CLI: `python geometry/run_e3i.py --device cuda --out_dir results_e3i`
- Resolved settings: `--K 10 --n_train 10000 --n_epochs 1500 --batch 256`
- Hardware / time: NVIDIA A40, 187.4 s (~3 min).

**The network.** A small fully-connected network mapping the three side lengths to the
ten leading frequencies. **34,826 trainable parameters.**

```
EigenvalueEncoder  (34,826 params)
  Linear(3 -> 128) -> GELU
  Linear(128 -> 128) -> GELU
  Linear(128 -> 128) -> GELU
  Linear(128 -> 10)             # works in log-coordinates so the scaling law is affine
```

## E3g: learning a genuinely new shape family (triangles)

> **Takeaway.** Triangles are not "stretched rectangles," and a rectangle-only model
> fails on them by ~54%. After a short exposure to a small library of triangles, the
> model predicts every one of the ten leading frequencies to under 5% across the whole
> family.

**What we tested, in plain words.** A triangle's frequencies cannot be guessed from its
bounding rectangle (the slanted side changes everything). We first confirm a
rectangle-trained model is uniformly wrong on triangles (~54% error, and notably *no*
better for the "half-a-square" right triangle, dispelling a natural intuition). Then we
let the model see a small library of triangles and re-check.

**What we found.** After the short exposure, the model reaches 5% accuracy on every
mode and at every point of the triangle family (the cleanest pass among the in-family
geometry tests). The encoder can absorb a genuinely non-rectangular shape family. This
is learning within the family from a small sample, not zero-shot transfer, and it was
run on a single seed.

**The figure.** Top: two bar charts, the rectangle-only model all far above 5%, the
trained model all below. Bottom: two accuracy heat-maps over the triangle family,
before and after the exposure.

![E3g: triangle eigenvalues, before and after](reports/e3g/e3g_triangle_eigenvalues.png)

**Which pillar (and which form).** Pillar 2, encoder transfer to a *non-product* shape
family.

**Compute.**
- Module / CLI: `python geometry/run_e3g.py --device cuda --out_dir results_e3g`
- Resolved settings: `--K 10 --n_rect_train 5000 --n_rect_epochs 600 --n_triangle_train 400 --n_finetune_epochs 200 --fem_nx 40 --fem_ny 40`
- Hardware / time: NVIDIA A40, 91.5 s (~1.5 min).

**The network.** The same small frequency-predicting network, here mapping a
two-number triangle descriptor to the ten leading frequencies. **34,698 trainable
parameters.**

```
EigenvalueEncoder  (34,698 params)
  Linear(2 -> 128) -> GELU -> Linear(128 -> 128) -> GELU -> Linear(128 -> 128) -> GELU -> Linear(128 -> 10)
```

## E3h: learning how a hole shifts the frequencies

> **Takeaway.** Punch a circular hole in a rectangle and its frequencies shift upward.
> A model blind to the hole is wrong by an amount that grows with the hole size
> (exactly as theory predicts), while a model given the hole's position and radius
> predicts every frequency to under 5%.

**What we tested, in plain words.** Removing material from a domain raises its natural
frequencies (more material vibrates lower, like a longer string). We test a model that
ignores the hole versus one that is told the hole's position and size, on rectangles
with a single circular hole of varying size.

**What we found.** Two things land together. First, the hole-blind model's error
grows smoothly with the hole's area, matching the classical perturbation prediction
(more removed area, larger shift): the physics behaves as expected. Second, the
hole-aware model learns that shift and reaches 5% accuracy on all ten frequencies
across the whole range of hole sizes, so the encoder handles a multiply-connected
shape. Caveats: learned within the family, single seed, and the largest-hole cases
are sparsely sampled.

**The figure.** Three panels: per-mode errors (hole-blind high, hole-aware low);
error against hole size (the hole-blind curve climbs, the hole-aware curve stays flat
and low); and a per-sample scatter. The middle panel shows the perturbation prediction
confirmed.

![E3h: perforated rectangles](reports/e3h/e3h_perforated_rectangles.png)

**Which pillar (and which form).** Pillar 2, encoder learns a *spectral perturbation*
(a shape with a hole).

**Compute.**
- Module / CLI: `python geometry/run_e3h.py --device cuda --out_dir results_e3h`
- Resolved settings: `--K 10 --n_rect_pretrain 2000 --n_perf_train 2000 --n_epochs 600 --fem_nx 64 --n_area_bins 6`
- Hardware / time: NVIDIA A40, 485.6 s (~8 min).

**The network.** The frequency-predicting network, taking a five-number descriptor
(width, height, hole centre x, hole centre y, hole radius). **35,082 trainable
parameters.**

```
EigenvalueEncoder  (35,082 params)
  Linear(5 -> 128) -> GELU -> Linear(128 -> 128) -> GELU -> Linear(128 -> 128) -> GELU -> Linear(128 -> 10)
```

## E3d: the sharp re-entrant corner

> **Takeaway.** An L-shape has an inward-pointing ("re-entrant") corner that creates a
> mathematical singularity. A convex-shape model fails (0 of 10 modes within 5%); a
> model given the cut reaches 6 of 10. The lowest mode, the one the corner distorts
> most, stays the hardest.

**What we tested, in plain words.** Cut a rectangular notch out of a rectangle and you
get an L-shape, whose inward corner makes the solution behave in a non-smooth way that
no convex shape exhibits. We test a model that only sees the bounding rectangle versus
one trained on a small library of L-shapes that is told where the cut is.

**What we found.** The cut-aware model goes from 0 to 6 of 10 frequencies under 5%, so
it captures most of the effect, including the bulk of the "missing material" shift. The
remaining error is concentrated in the lowest mode, which is exactly the one the corner
singularity distorts, matching the physics. The result is qualified: the pass is
in-family and single-seed, and the lowest mode (the one the corner singularity
distorts most) is the residual to close next.

**The figure.** Left: per-mode errors (convex model high, cut-aware model mostly low).
Right: two heat-maps of the lowest vibration pattern, predicted versus reference; the
true pattern is visibly pushed away from the sharp corner.

![E3d: L-shape re-entrant corner](reports/e3d/e3d_lshape_eigenvalues.png)

**Which pillar (and which form).** Pillar 2, encoder on a *non-convex, singular*
geometry.

**Compute.**
- Module / CLI: `python geometry/run_e3d.py --device cuda --out_dir results_e3d`
- Resolved settings: `--K 10 --n_rect_train 5000 --n_rect_epochs 1200 --n_lshape_train 200 --n_finetune_epochs 200 --nx_fem 64 --ny_fem 64`
- Hardware / time: NVIDIA A40, 141.9 s (~2.4 min).

**The network.** A fresh four-number-input frequency network (width, height, cut x, cut
y). **34,954 trainable parameters.**

```
EigenvalueEncoder  (34,954 params)
  Linear(4 -> 128) -> GELU -> Linear(128 -> 128) -> GELU -> Linear(128 -> 128) -> GELU -> Linear(128 -> 10)
```

## E3f: the shape-matched basis beats a fixed grid on curved shapes

> **Takeaway.** On heat flow in disks of varying radius, a solver that uses the disk's
> own natural modes (the geometry-adapted basis) represents the solution essentially
> exactly, while a standard fixed-grid model sits at 2-50% error and, crucially, does
> *not* improve when trained on more disk sizes. The bottleneck is the basis, not the
> amount of data.

**What we tested, in plain words.** The same heat problem on disks of different radii.
We compare three solvers: one built on the disk's own natural modes (the
geometry-adapted basis), and two standard fixed-grid models (one trained on a single
radius, one on the full range). The fair comparison is between the two standard
models: does feeding the fixed-grid model more shapes fix it?

**What we found.** The like-for-like finding supports the pillar: the fixed-grid model
fails on disks (2-50% error), and training it on more radii does *not* help, the
signature of a fixed-basis limitation. The geometry-adapted basis, by contrast,
represents the solution to machine precision. One caveat matters: that machine-precision
figure uses the *exact* known modes (not a learned prediction of them), so the
"millions of times better" ratio is not a like-for-like comparison. The claim this
experiment supports is the basis thesis itself: shape-matched modes succeed where a
fixed grid cannot.

**The figure.** A log-scale plot: the geometry-adapted curve flat at the bottom
(machine precision), the two fixed-grid curves an order of magnitude or more above and
not improving with more shapes. The figure illustrates the basis limitation, not a
like-for-like accuracy race (see the caveat above).

![E3f: disk heat equation, adapted basis vs fixed grid](reports/e3f/e3f_disk_heat_ood.png)

**Which pillar (and which form).** Pillar 2, the *geometry-adapted spectral basis*
(LB-FNO) architecture, on curved (non-rectangular) shapes.

**Compute.**
- Module / CLI: `python geometry/run_e3f.py --device cuda --out_dir results_e3f`
- Resolved settings: `--nx 96 --n_train 10000 --n_epochs 800 --batch 64 --width 48 --n_modes_x 24 --n_modes_y 24 --n_layers 4 --R_min 0.4 --R_max 1.5 --K 24`
- Hardware / time: NVIDIA A40, 16,313.1 s (~4.5 h).

**The network.** The geometry-adapted solver uses the exact known modes (no trainable
parameters in that path); the fixed-grid baseline it is compared against is a 2D
Fourier operator. **5,321,169 trainable parameters** for that baseline.

```
HeatFNO2d (standard fixed-grid baseline)  (5,321,169 params)
  lift     : Conv2d(2 -> 48, kernel 1x1)
  spectral : 4 x SpectralConv2d(width 48, 24 x 24 modes)
  local    : 4 x Conv2d(48 -> 48, kernel 1x1)
  proj     : Conv2d(48 -> 64) -> GELU -> Conv2d(64 -> 1)
```

---

# Pillar 3: Domain decomposition (stitching pieces together)

Pillar 3 is represented here by one clean result, stated precisely below; its
further tests (many subdomains, indefinite operators, time-dependent problems)
belong to the larger E4 package.

## E4e: the learned "stitch" obeys the textbook law  (core Pillar 3 result)

> **Takeaway.** When two pieces are stitched with a learned rule, increasing the
> overlap between them speeds up convergence in exactly the way classical theory
> predicts, and the learned rule matches an independent classical solver to within ~1%
> at every overlap.

**What we tested, in plain words.** In domain decomposition, pieces share an overlap
region; classical theory (Toselli-Widlund) says more overlap makes the
stitch-and-repeat process converge faster. We measure the convergence speed of a
*learned* stitching rule as we increase the overlap, and compare it to an independent
classical solver run on the same problem.

**What we found.** The learned stitch reproduces the classical convergence curve to
~1% at every overlap width, with the same slope. The learned component is not behaving
ad hoc; it obeys the same convergence law as the classical method. The scope is
deliberate: the learned rule is trained to imitate the classical one, and the
experiment validates convergence *speed*, not end-to-end assembled accuracy.

**The figure.** Left: convergence speed against overlap, the learned and classical
curves lying on top of each other and both decreasing. Right: the per-iteration
convergence histories. The two curves coincide, matching the theory.

![E4e: Schwarz convergence vs overlap width](reports/e4e/e4e_schwarz_overlap.png)

**Which pillar (and which form).** Pillar 3, the *learned consensus inherits the
classical Schwarz convergence theory*.

**Compute.**
- Module / CLI: `python schwarz/run_e4e.py --device cuda --out_dir results_e4e`
- Resolved settings: `--overlaps 0.0 0.05 0.1 0.2 0.4 --nx_half 64 --n_train 5000 --n_epochs 600 --batch 64 --width 64 --n_modes 32 --n_layers 4 --max_iter 30 --tol 1e-6`
- Hardware / time: NVIDIA A40, 3,452.2 s (~58 min).

**The networks.** Two subdomain Fourier operators (one per piece, **550,210 params**
each) and a small stitching network (**4,930 params**), all conditioned on a one-hot
overlap class.

```
OverlapSubdomainFNO  (550,210 params, x2)        OverlapDtN  (4,930 params)
  lift     : Conv1d(7 -> 64)                       Linear(9 -> 64) -> GELU
  4 x SpectralConv1d(64, 32)                       Linear(64 -> 64) -> GELU
  4 x Conv1d(64 -> 64)                             Linear(64 -> 2)
  proj_u   : Conv1d(64 -> 64) -> GELU -> Conv1d(64 -> 1)
  foreign  : Linear(69 -> 64) -> GELU -> Linear(64 -> 1)
```

---

# Cross-pillar bridge

The pillars are designed to combine. The bridge below couples Pillar 2's exact
diffusion building block into Pillar 1's composition.

## E5a: an exact diffusion move sharpens the composition

> **Takeaway.** Replacing the learned diffusion move with Pillar 2's *exact* diffusion
> move tips Pillar 1's hand-composition from losing to winning against a single big
> model, on the harder (high-drift) problems.

**What we tested, in plain words.** Pillar 2 gives an exact, closed-form diffusion step
(no learning error). We drop it in place of the learned diffusion piece inside Pillar
1's composition (keeping the drift piece learned) and ask whether the composition now
beats a single big model trained on the whole problem, across a range of drift
strengths.

**What we found.** At moderate-to-high drift the exact-diffusion composition wins where
the all-learned composition loses: the exact piece removes one source of error and tips
the balance. This is a concrete demonstration that the pillars reinforce each other.
Two caveats apply: the win is clearest at the middle drift setting (the strongest-drift
point rests on a sub-1% margin from a single seed), and the pre-registered prediction
about *where* the gain would appear was wrong (it appears at high drift, not at zero),
so the effect is real but differently shaped than expected.

**The figure.** The left panel shows the composition reaching break-even where the
all-learned version does not. Note a labelling error in the right panel: its title
reads "advantage shrinks" while the plotted line rises; read the line, not the title
(the label will be corrected).

![E5a: closed-form diffusion brick inside Strang](reports/e5a/e5a_closed_form_diffusion_brick.png)

**Which pillar (and which form).** The Pillar 1 x Pillar 2 bridge: Pillar 2's exact
diffusion semigroup used as the diffusion brick inside Pillar 1's Strang composition.

**Compute.**
- Module / CLI: `python commutator_geometry/run_e5a.py --device cuda --out_dir results_e5a`
- Resolved settings: `--shears 0.0 0.5 1.0 --n_brick 4000 --n_joint 8000 --budgets 1000 2000 4000 8000 --n_epochs 250 --nx 128 --width 64 --n_modes 32 --n_layers 4 --batch 128`
- Hardware / time: NVIDIA A40, 902.8 s (~15 min).

**The networks.** The exact diffusion move has no trainable parameters; the learned
drift move is a Fourier operator (**549,633 params**), compared against a monolithic
baseline (**549,697 params**), both of the same shape as in E2e.

---

# Core pillar experiments

One result per pillar carries the core of the case. Read together they make the
whole argument: splitting composes (Pillar 1), geometry generalises (Pillar 2),
and the learned stitching obeys the classical theory (Pillar 3).

### Pillar 1 (E2.b): the free 50x

![E2.b](reports/e2b/e2b_strang_commuting.png)

**Caption.** *One model, trained only on the two effects separately, predicts them
combined with about fifty times less error when it is composed by hand using a
200-year-old mathematical recipe, with no extra training. Learned building blocks
can be reassembled instead of retrained.*

**Significance.** The operational core of Pillar 1: a learned operator, reassembled
at use-time with no extra training and no new parameters, in the regime where the
splitting recipe is exact.

### Pillar 2 (E3i): learned the law, not a lookup table

![E3i](reports/e3i/e3i_box_eigenvalues.png)

**Caption.** *A small model is trained to read a 3D box's dimensions and predict its
natural vibration frequencies. The white square marks the sizes it was trained on; it
stays accurate (inside the contour) far outside that square, including for boxes larger
than any it ever saw. It has internalised the underlying law, not memorised examples,
the property a reusable surrogate needs.*

**Significance.** Genuine extrapolation in 3D: the encoder has captured the scaling
law rather than a table of examples, which is the property a reusable surrogate
requires.

### Pillar 3 (E4e): the learned method obeys the textbook

![E4e](reports/e4e/e4e_schwarz_overlap.png)

**Caption.** *When two solved pieces are stitched together with a learned rule,
increasing their overlap speeds up convergence exactly as classical mathematics
predicts: the learned curve (solid) lies on top of the independent classical curve
(dashed). The learned components are not black boxes that happen to work; they obey
the established theory.*

**Significance.** The learned component provably matches the classical convergence
law, evidence that the approach inherits, rather than bypasses, established numerical
theory.

---

*Scope note: the per-experiment narratives are the `results_<exp>.md` files next
to each script. These experiments are drawn from three larger packages (E2, E3,
E4) and the cross-pillar bridges (E5); reviewers can be granted access to the
complete packages on request.*
