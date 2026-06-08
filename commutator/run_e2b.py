"""Experiment E2.b (Phase A, Pillar 1, supportive): Strang composition of a
single pretrained network in the COMMUTING regime (s = 0).

Why E2.b after E2 and E2.a both came up short
---------------------------------------------
E2 (two separately trained brick FNOs, Strang-composed at deployment) failed at
every shear value tested *including* s = 0; the post-hoc diagnosis was
"per-brick approximation error compounding through composition", i.e. each
brick's finite learning accuracy dominated whatever the splitting theory would
have given exactly.

E2.a (single network pretrained on a mix of atomic samples, fine-tuned on the
hardest composite case s = 1.0) showed only a small win at N = 100 and
negative transfer at N >= 500. Diagnosis: at s = 1.0 the diffusion–advection
commutator is large, so the atomic pretraining encodes a prior that is not
algebraically valid for the downstream operator.

Both experiments stacked the deck against Pillar 1:
  - E2: two undertrained bricks
  - E2.a: tested non-commuting downstream operator (large BCH error)

E2.b targets the regime where Pillar 1's algebra is MOST favorable, with the
architectural fix from E2.a (one shared network instead of two separate bricks):

  Downstream task: constant-velocity advection-diffusion (s = 0).
  At s = 0 on the periodic torus, the diffusion generator A = D ∂_xx and the
  advection generator B = -v_0 ∂_x have ZERO COMMUTATOR, so

      exp(Δt(A + B))  =  exp(Δt/2 · A) ∘ exp(Δt · B) ∘ exp(Δt/2 · A)    (exactly).

  Strang is mathematically exact at s = 0. Any remaining error in the composed
  prediction is therefore *purely* due to per-brick learning error, NOT the
  splitting term Conjecture 1 names.

Architecture
------------
  ONE FNO (same arch as E2's bricks: FNO1d, width 64, n_cond=3, default 4
  spectral layers). Conditioning vector: (Δt, D, v_0). Zero-valued coefficients
  indicate "this generator is off" (pure-diffusion samples have v_0 = 0, pure-
  advection samples have D = 0).

Pretraining (identical to E2.a, with slightly extended dt range)
----------------------------------------------------------------
  - n_pretrain pure-diffusion samples:   D ~ U(0.01, 0.2),  v_0 = 0
  - n_pretrain pure-advection samples:   v_0 ~ U(0.1, 2.0), D = 0, shear = 0
  - dt_range_pretrain = (0.0025, 0.02). The lower bound 0.0025 ensures
    Strang half-steps Δt/2 stay in-distribution at test time (where the test
    dt ∈ [0.005, 0.02] ⇒ Δt/2 ∈ [0.0025, 0.01] ⊂ pretrain range).

Test set
--------
  n_test composite samples at s = 0:   D ~ U(0.01, 0.2),  v_0 ~ U(0.1, 2.0).

Conditions evaluated on the SAME pretrained model
--------------------------------------------------
  (Z)  Zero-shot, single call:
         u_hat = F_θ(u_0, (Δt, D, v_0))
       Asks: did the model learn to interpolate joint conditioning into the
       joint operator by itself? This is implicit / emergent composition.

  (S)  Strang composition, three calls, no composite-task training:
         u_1/2   = F_θ(u_0,    (Δt/2, D,  0  ))    diffusion half-step
         u_1*    = F_θ(u_1/2,  (Δt,   0,  v_0))    advection full-step
         u_hat   = F_θ(u_1*,   (Δt/2, D,  0  ))    diffusion half-step
       This is the algebraically exact decomposition at s=0; any error is the
       network's per-brick error. Explicit / hard-coded composition. HEADLINE
       CONDITION for Pillar 1.

  (A)  From scratch on N composite samples (fresh random init).
  (B)  Pretrained → fine-tuned on N composite samples.

Headline plot
-------------
  rel L2 test error vs N, log–log. Red = scratch, blue = pretrained+ft,
  green dashed = Strang (constant), gray dotted = zero-shot (constant).

Interpretation guide for the four possible outcomes
---------------------------------------------------
  - Strang (S) competitive with scratch at large N:
      Pillar 1 is operationally useful in 1D when the algebra works for it.
      Strongest possible 1D evidence for the proposal's hardcoded-composition
      story (Section IV.1.2 of the proposal).

  - Strang (S) >> scratch:
      Per-brick error still dominates even with a shared network; Pillar 1
      cannot be operationalised in 1D at all.

  - Zero-shot (Z) ≈ Strang (S) and both good:
      The model has implicitly learned compositionality through conditioning
      (MPP-style emergent composition); the explicit S-MoE may not be needed.

  - Zero-shot (Z) >> Strang (S):
      The model learned the atoms but not the implicit composition; explicit
      Strang at deployment is the operational path. Supports the proposal's
      S-MoE architecture (Section IV.1.7).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.simulators import (
    brick_diffusion, brick_advection, simulate_full,
    batch_random_ics, shear_profile,
)
from commutator.core import FNO1d


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _draw(rng, lo, hi, n):
    return torch.tensor(rng.uniform(lo, hi, n), dtype=torch.float32).unsqueeze(-1)


def generate_pretraining_dataset(n_per_family: int, nx: int,
                                 dt_range: tuple = (0.0025, 0.02),
                                 seed: int = 0) -> TensorDataset:
    """Mixed atomic dataset: pure diffusion + pure constant-v advection.

    Extended dt_range lower bound to 0.0025 so that Strang half-steps
    Δt/2 ∈ [0.0025, 0.01] (at test dt ∈ [0.005, 0.02]) stay in-distribution.
    """
    rng_d = np.random.default_rng(seed)
    u0_d  = batch_random_ics(n_per_family, nx, seed=seed)
    D_d   = _draw(rng_d, 0.01, 0.20, n_per_family)
    v0_d  = torch.zeros(n_per_family, 1, dtype=torch.float32)
    dt_d  = _draw(rng_d, *dt_range, n_per_family)
    u1_d  = brick_diffusion(u0_d, D=D_d, dt=dt_d)

    rng_a = np.random.default_rng(seed + 1)
    u0_a  = batch_random_ics(n_per_family, nx, seed=seed + 1)
    D_a   = torch.zeros(n_per_family, 1, dtype=torch.float32)
    v0_a  = _draw(rng_a, 0.10, 2.00, n_per_family)
    dt_a  = _draw(rng_a, *dt_range, n_per_family)
    u1_a  = brick_advection(u0_a, v0=v0_a, dt=dt_a, shear=0.0)

    u0 = torch.cat([u0_d, u0_a], dim=0)
    u1 = torch.cat([u1_d, u1_a], dim=0)
    cond = torch.cat([
        torch.cat([dt_d, D_d, v0_d], dim=-1),
        torch.cat([dt_a, D_a, v0_a], dim=-1),
    ], dim=0)
    return TensorDataset(u0, u1, cond)


def generate_composite_s0_dataset(n_samples: int, nx: int,
                                  dt_range: tuple = (0.005, 0.02),
                                  seed: int = 0) -> TensorDataset:
    """Constant-velocity advection-diffusion (s = 0): commuting regime.

    On the periodic torus with shear = 0, A = D ∂_xx and B = −v_0 ∂_x satisfy
    [A, B] = 0 exactly, so Lie–Trotter and Strang are algebraically EXACT.
    """
    rng = np.random.default_rng(seed)
    u0  = batch_random_ics(n_samples, nx, seed=seed)
    D   = _draw(rng, 0.01, 0.20, n_samples)
    v0  = _draw(rng, 0.10, 2.00, n_samples)
    dt  = _draw(rng, *dt_range, n_samples)
    profile = shear_profile(nx, seed=0)  # unused at shear=0, required by API
    u1  = simulate_full(u0, D=D, v0=v0, dt=dt, shear=0.0, profile=profile)
    return TensorDataset(u0, u1, torch.cat([dt, D, v0], dim=-1))


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train(model, dataset, n_epochs, batch_size=64, lr=1e-3,
          device="cpu", verbose=False):
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    final = None
    for ep in range(n_epochs):
        running = 0.0
        n_batches = 0
        for u0, u1, cond in loader:
            u0, u1, cond = u0.to(device), u1.to(device), cond.to(device)
            loss = F.mse_loss(model(u0, cond), u1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        final = running / max(n_batches, 1)
        if verbose and (ep == 0 or (ep + 1) % 50 == 0 or ep == n_epochs - 1):
            print(f"    epoch {ep+1:4d}/{n_epochs}   train_mse = {final:.2e}")
    return final


def rel_l2(pred, target):
    return ((pred - target).norm(dim=-1) /
            (target.norm(dim=-1) + 1e-8)).mean().item()


def evaluate_single_call(model, dataset, device):
    """Zero-shot / single-call evaluation: one forward pass with joint cond."""
    model.eval()
    loader = DataLoader(dataset, batch_size=128)
    errs = []
    with torch.no_grad():
        for u0, u1, cond in loader:
            u0, u1, cond = u0.to(device), u1.to(device), cond.to(device)
            errs.append(rel_l2(model(u0, cond), u1))
    return float(np.mean(errs))


def evaluate_strang(model, dataset, device):
    """Strang composition: same network, three calls per timestep.

    e^{Δt(A+B)}  ≈  e^{Δt/2 · A} ∘ e^{Δt · B} ∘ e^{Δt/2 · A}

    A is diffusion (cond = (Δt/2, D, 0)); B is advection (cond = (Δt, 0, v_0)).
    At s = 0 this is EXACT; any error is per-brick learning error.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=128)
    errs = []
    with torch.no_grad():
        for u0, u1, cond in loader:
            u0, u1, cond = u0.to(device), u1.to(device), cond.to(device)
            dt   = cond[:, 0:1]
            D    = cond[:, 1:2]
            v0   = cond[:, 2:3]
            half = dt / 2
            zero_v = torch.zeros_like(v0)
            zero_D = torch.zeros_like(D)
            cond_d = torch.cat([half, D,      zero_v], dim=-1)
            cond_a = torch.cat([dt,   zero_D, v0    ], dim=-1)
            u = model(u0, cond_d)
            u = model(u,  cond_a)
            u = model(u,  cond_d)
            errs.append(rel_l2(u, u1))
    return float(np.mean(errs))


def evaluate_lie_trotter(model, dataset, device):
    """First-order Lie–Trotter as a diagnostic side-channel.

    e^{Δt(A+B)}  ≈  e^{Δt · A} ∘ e^{Δt · B}

    Order-1; should be slightly worse than Strang even in the exact-algebra
    regime if the per-brick errors aren't perfectly symmetric.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=128)
    errs = []
    with torch.no_grad():
        for u0, u1, cond in loader:
            u0, u1, cond = u0.to(device), u1.to(device), cond.to(device)
            dt   = cond[:, 0:1]
            D    = cond[:, 1:2]
            v0   = cond[:, 2:3]
            zero_v = torch.zeros_like(v0)
            zero_D = torch.zeros_like(D)
            cond_d = torch.cat([dt, D,      zero_v], dim=-1)
            cond_a = torch.cat([dt, zero_D, v0    ], dim=-1)
            u = model(u0, cond_d)
            u = model(u,  cond_a)
            errs.append(rel_l2(u, u1))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E2.b: Strang composition of a single pretrained network "
                    "in the commuting regime (s = 0)"
    )
    # Pretraining
    parser.add_argument("--n_pretrain", type=int, default=5000,
                        help="Per-family pretraining samples (total = 2x)")
    parser.add_argument("--n_pretrain_epochs", type=int, default=500)
    parser.add_argument("--pretrain_dt_min", type=float, default=0.0025,
                        help="Pretrain dt LOWER bound (must be ≤ test_dt_min/2 "
                             "so Strang half-steps stay in-distribution)")
    parser.add_argument("--pretrain_dt_max", type=float, default=0.02)

    # Fine-tune / scratch sweep
    parser.add_argument("--ns", nargs="+", type=int,
                        default=[50, 100, 250, 500, 1000, 2000, 5000])
    parser.add_argument("--n_finetune_epochs", type=int, default=300)
    parser.add_argument("--n_seeds", type=int, default=3)

    # Test set
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--test_dt_min", type=float, default=0.005)
    parser.add_argument("--test_dt_max", type=float, default=0.02)

    # Architecture
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n_modes", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=4)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="results_e2b")
    parser.add_argument("--seed", type=int, default=0)

    # Optional reuse of a previously trained pretrained.pt (e.g. from E2.a)
    parser.add_argument("--load_pretrained", type=str, default=None,
                        help="Path to a previously saved pretrained.pt; if "
                             "supplied, skips the pretraining stage. The "
                             "saved model must have been trained with a "
                             "dt range covering pretrain_dt_min.")

    args = parser.parse_args()

    # Sanity check: Strang half-steps must be in pretraining distribution
    if args.pretrain_dt_min > args.test_dt_min / 2:
        print(f"  ⚠ pretrain_dt_min={args.pretrain_dt_min} > test_dt_min/2="
              f"{args.test_dt_min/2}; Strang half-steps will be OUT of "
              f"distribution. Lower pretrain_dt_min or raise test_dt_min.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("═" * 78)
    print("  Experiment E2.b: Pillar 1 in the commuting regime (s = 0)")
    print("═" * 78)
    print(f"  Device:           {args.device}")
    print(f"  FNO:              width={args.width}  modes={args.n_modes}  "
          f"layers={args.n_layers}")
    print(f"  Pretrain:         2 × {args.n_pretrain} atoms, "
          f"{args.n_pretrain_epochs} epochs, "
          f"dt ∈ [{args.pretrain_dt_min}, {args.pretrain_dt_max}]")
    print(f"  Test (s = 0):     {args.n_test} composite samples, "
          f"dt ∈ [{args.test_dt_min}, {args.test_dt_max}]")
    print(f"  Sweep:            N ∈ {args.ns} × {args.n_seeds} seeds, "
          f"{args.n_finetune_epochs} epochs each")
    print()

    t_total = time.time()

    # ---- [1/4] Pretraining (or load) -------------------------------------
    fno_pre = FNO1d(n_cond=3, width=args.width,
                    n_modes=args.n_modes, n_layers=args.n_layers).to(args.device)

    if args.load_pretrained is not None and Path(args.load_pretrained).exists():
        print(f"[1/4] Loading pretrained weights from {args.load_pretrained}...")
        state = torch.load(args.load_pretrained, map_location=args.device)
        fno_pre.load_state_dict(state)
        final_pretrain = None
    else:
        print("[1/4] Pretraining on atomic operators "
              "(pure diffusion + pure constant-v advection)...")
        pretrain_ds = generate_pretraining_dataset(
            args.n_pretrain, args.nx,
            dt_range=(args.pretrain_dt_min, args.pretrain_dt_max), seed=10)
        t0 = time.time()
        final_pretrain = train(
            fno_pre, pretrain_ds, args.n_pretrain_epochs,
            batch_size=args.batch_size, lr=args.lr,
            device=args.device, verbose=True)
        print(f"  Pretraining done in {time.time()-t0:.1f}s   "
              f"train_mse = {final_pretrain:.2e}")
        torch.save({k: v.clone().cpu() for k, v in fno_pre.state_dict().items()},
                   out_dir / "pretrained.pt")

    pretrained_state = {k: v.clone().cpu()
                        for k, v in fno_pre.state_dict().items()}

    # ---- [2/4] Test set + diagnostic evaluations on pretrained model -----
    print(f"\n[2/4] Building s = 0 composite test set "
          f"(N_test = {args.n_test})...")
    test_ds = generate_composite_s0_dataset(
        args.n_test, args.nx,
        dt_range=(args.test_dt_min, args.test_dt_max), seed=777)

    err_zero   = evaluate_single_call(fno_pre, test_ds, args.device)
    err_lie    = evaluate_lie_trotter(fno_pre, test_ds, args.device)
    err_strang = evaluate_strang(fno_pre, test_ds, args.device)

    print(f"  Zero-shot single call    (Z):  rel_L2 = {err_zero:.4f}")
    print(f"  Lie–Trotter, 2 calls/step (L): rel_L2 = {err_lie:.4f}")
    print(f"  Strang, 3 calls/step      (S): rel_L2 = {err_strang:.4f}")
    print("  (S) is the headline Pillar 1 condition: algebraically exact at s=0.")

    # ---- [3/4] Fine-tune & scratch sweep ---------------------------------
    print(f"\n[3/4] Fine-tune / scratch sweep on s = 0 composite "
          f"({len(args.ns)} budgets × 2 conditions × {args.n_seeds} seeds = "
          f"{len(args.ns) * 2 * args.n_seeds} trainings)...")
    rows = []

    for N in args.ns:
        # Shared composite training set so the only difference between scratch
        # and pretrained is the initial weights
        train_ds = generate_composite_s0_dataset(
            N, args.nx,
            dt_range=(args.test_dt_min, args.test_dt_max), seed=42)
        for seed in range(args.n_seeds):
            # (A) From scratch
            torch.manual_seed(seed)
            fno_scratch = FNO1d(n_cond=3, width=args.width,
                                n_modes=args.n_modes, n_layers=args.n_layers)
            train(fno_scratch, train_ds, args.n_finetune_epochs,
                  batch_size=args.batch_size, lr=args.lr, device=args.device)
            err_scratch = evaluate_single_call(fno_scratch, test_ds, args.device)
            rows.append({"N": N, "condition": "scratch",
                         "seed": seed, "err": err_scratch})

            # (B) Pretrained → fine-tuned
            torch.manual_seed(seed)
            fno_ft = FNO1d(n_cond=3, width=args.width,
                           n_modes=args.n_modes, n_layers=args.n_layers)
            fno_ft.load_state_dict(pretrained_state)
            train(fno_ft, train_ds, args.n_finetune_epochs,
                  batch_size=args.batch_size, lr=args.lr, device=args.device)
            err_ft = evaluate_single_call(fno_ft, test_ds, args.device)
            rows.append({"N": N, "condition": "pretrained",
                         "seed": seed, "err": err_ft})

            speedup = err_scratch / max(err_ft, 1e-9)
            print(f"  N={N:5d}  seed={seed}  "
                  f"scratch={err_scratch:.4f}  pretrained={err_ft:.4f}   "
                  f"speedup={speedup:.2f}×")

    # ---- [4/4] Aggregate + plot + verdict --------------------------------
    print("\n" + "═" * 78)
    print(f"  Summary, s = 0 (commuting regime)")
    print("═" * 78)
    print(f"  {'N':>6}  {'scratch (mean ± std)':>22}  "
          f"{'pretrained (mean ± std)':>24}  {'speedup':>8}")
    sm, ss, fm, fs = [], [], [], []
    for N in args.ns:
        s_vals = [r["err"] for r in rows
                  if r["N"] == N and r["condition"] == "scratch"]
        f_vals = [r["err"] for r in rows
                  if r["N"] == N and r["condition"] == "pretrained"]
        sm.append(float(np.mean(s_vals))); ss.append(float(np.std(s_vals)))
        fm.append(float(np.mean(f_vals))); fs.append(float(np.std(f_vals)))
        print(f"  {N:6d}  {np.mean(s_vals):8.4f} ± {np.std(s_vals):.4f}    "
              f"{np.mean(f_vals):8.4f} ± {np.std(f_vals):.4f}    "
              f"{np.mean(s_vals)/max(np.mean(f_vals),1e-9):>7.2f}×")

    scratch_best = float(min(sm))
    pretrained_best = float(min(fm))

    print()
    print(f"  Zero-shot single call (Z):  {err_zero:.4f}")
    print(f"  Lie–Trotter      (L):       {err_lie:.4f}")
    print(f"  Strang           (S):       {err_strang:.4f}      ← HEADLINE")
    print(f"  Best scratch      :         {scratch_best:.4f}")
    print(f"  Best pretrained+ft:         {pretrained_best:.4f}")

    # JSON dump
    with open(out_dir / "e2b_raw.json", "w") as f:
        json.dump({
            "args": vars(args),
            "Ns": list(args.ns),
            "err_scratch_mean": sm, "err_scratch_std": ss,
            "err_ft_mean":      fm, "err_ft_std":      fs,
            "err_zero_shot":    err_zero,
            "err_lie_trotter":  err_lie,
            "err_strang":       err_strang,
            "pretrain_train_mse": final_pretrain,
            "rows": rows,
            "elapsed_s": time.time() - t_total,
        }, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    Ns_arr = np.asarray(args.ns, dtype=float)
    sm_a, ss_a = np.asarray(sm), np.asarray(ss)
    fm_a, fs_a = np.asarray(fm), np.asarray(fs)

    ax.plot(Ns_arr, sm_a, "o-", color="#c0392b", linewidth=2, markersize=8,
            label="From scratch")
    ax.fill_between(Ns_arr, np.clip(sm_a - ss_a, 1e-6, None), sm_a + ss_a,
                    color="#c0392b", alpha=0.15)

    ax.plot(Ns_arr, fm_a, "s-", color="#2a6496", linewidth=2, markersize=8,
            label="Pretrained on atomic operators → fine-tuned")
    ax.fill_between(Ns_arr, np.clip(fm_a - fs_a, 1e-6, None), fm_a + fs_a,
                    color="#2a6496", alpha=0.15)

    ax.axhline(err_zero, ls=":", color="#777", linewidth=1.2,
               label=f"Pretrained zero-shot, single call: {err_zero:.3f}")
    ax.axhline(err_strang, ls="--", color="#1f8a4c", linewidth=2.0,
               label=f"Pretrained + Strang composition (3 calls): {err_strang:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fine-tuning / from-scratch sample budget $N$", fontsize=12)
    ax.set_ylabel("Relative $L^2$ test error", fontsize=12)
    ax.set_title("E2.b: Strang composition of a pretrained network in the "
                 "commuting regime\n"
                 f"(target: constant-velocity advection–diffusion, $s = 0$, "
                 f"{args.n_seeds} seeds)",
                 fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "e2b_strang_commuting.pdf"
    fig.savefig(out_path)
    print(f"\nPlot saved to {out_path}")
    print(f"Total elapsed: {(time.time()-t_total)/60:.1f} min")

    # Verdict
    print("\n" + "─" * 78)
    strang_vs_scratch = err_strang / max(scratch_best, 1e-9)
    zero_vs_strang    = err_zero   / max(err_strang,    1e-9)

    print(f"  Strang / best scratch ratio:     {strang_vs_scratch:.2f}×")
    print(f"  Zero-shot / Strang ratio:        {zero_vs_strang:.2f}×")
    print()

    if strang_vs_scratch < 1.2:
        print("  ✓ Strang composition matches (or beats) the best from-scratch model")
        print("    WITHOUT ANY composite-task training. Pillar 1 is operationally")
        print("    useful in 1D when the algebra works for it.")
        print("    → Headline result: hard-coded composition of a single network")
        print("      pretrained on atoms is competitive with the monolithic baseline.")
    elif strang_vs_scratch < 2.0:
        print("  ~ Strang composition is in the same order of magnitude as scratch")
        print("    but doesn't match it. Atomic pretraining is helpful but per-brick")
        print("    error still dominates the splitting advantage in 1D.")
    else:
        print("  ✗ Strang composition is materially worse than from-scratch.")
        print("    Per-brick learning error compounds even when the algebra is exact.")
        print("    Pillar 1 fails in 1D even in the commuting regime; the 1D")
        print("    joint problem is simply too easy for the single-call baseline.")

    print()
    if zero_vs_strang < 1.2:
        print("  ✓ Zero-shot (single call) ≈ Strang (3 calls): the network has")
        print("    learned implicit compositionality via conditioning. The explicit")
        print("    S-MoE may be unnecessary; MPP-style emergent composition")
        print("    suffices on this problem.")
    elif zero_vs_strang > 2.0:
        print("  → Zero-shot >> Strang: the network learned the atoms but did NOT")
        print("    learn to combine them via conditioning. Explicit Strang at")
        print("    deployment is the operational path. This supports the proposal's")
        print("    S-MoE architecture (Section IV.1.7).")
    else:
        print("  ~ Zero-shot and Strang are comparable; partial evidence for")
        print("    implicit composition.")
    print("─" * 78)


if __name__ == "__main__":
    main()
