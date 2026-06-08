"""Experiment E2d (Pillar 1): three-brick pretraining curriculum.

Tests whether adding a third elementary brick (cubic reaction) to the
pretraining library of E2.a (diffusion + advection) lets a single FNO transfer
to Allen-Cahn-with-drift with strictly fewer fine-tuning samples than the
two-brick curriculum. This is a direct test of the generator-coverage view of
Pillar 1 + Conjecture 1: the downstream PDE is the sum of the three atomic
generators, so a curriculum that covers all three should out-transfer one that
covers only two.

Pretraining libraries (single multi-task FNO, conditioned on (dt, D, v0, gamma)):
  - two-brick : pure diffusion (D ~ U(0.01, 0.20), v0 = gamma = 0)
                pure advection (v0 ~ U(0.10, 2.00), D = gamma = 0, s = 0)
  - three-brick: above plus pure cubic reaction
                (gamma ~ U(0.5, 5.0), D = v0 = 0)

Downstream target (single composite family):
  ∂_t u = D ∂_xx u - v(x) ∂_x u - gamma u (1 - u^2)
with D ~ U(0.01, 0.20), v0 ~ U(0.10, 2.00), gamma ~ U(0.5, 5.0), shear s = 0.5.

For each fine-tuning budget N and a few seeds, three conditions are trained:
  (A) random-init from scratch
  (B) two-brick pretrained, then fine-tuned
  (C) three-brick pretrained, then fine-tuned

Headline metric: speedup curves err_scratch(N) / err_pretrained(N) for both
pretrained conditions. Pre-registered prediction: three-brick speedup is
strictly greater than two-brick speedup at every N.

GPU-target runtime: about 2 hours on a single GPU at default flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commutator.core import FNO1d
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary,
    save_figure, write_run_json,
)
from shared.simulators import (
    batch_random_ics, brick_advection, brick_diffusion, brick_reaction,
    shear_profile, simulate_full_reaction,
)


# ---------------------------------------------------------------------------
# Dataset generation (cond = [dt, D, v0, gamma] across all bricks and target)
# ---------------------------------------------------------------------------

def _draw(rng, lo, hi, n):
    return torch.tensor(rng.uniform(lo, hi, n), dtype=torch.float32).unsqueeze(-1)


def _make_pretrain_block(n: int, nx: int, brick: str,
                         dt_range: tuple, seed: int) -> TensorDataset:
    """Generate one atomic-brick block of the pretraining library.

    Each block draws one PDE parameter from its physical range; the other
    two parameters are zero (so the FNO sees the operator type directly via
    its conditioning channels).
    """
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n, nx, seed=seed)
    dt = _draw(rng, *dt_range, n)
    zero = torch.zeros(n, 1, dtype=torch.float32)
    if brick == "diffusion":
        D = _draw(rng, 0.01, 0.20, n)
        v0 = zero
        gamma = zero
        u1 = brick_diffusion(u0, D=D, dt=dt)
    elif brick == "advection":
        D = zero
        v0 = _draw(rng, 0.10, 2.00, n)
        gamma = zero
        u1 = brick_advection(u0, v0=v0, dt=dt, shear=0.0)
    elif brick == "reaction":
        D = zero
        v0 = zero
        gamma = _draw(rng, 0.5, 5.0, n)
        u1 = brick_reaction(u0, gamma=gamma, dt=dt)
    else:
        raise ValueError(brick)
    cond = torch.cat([dt, D, v0, gamma], dim=-1)
    return TensorDataset(u0, u1, cond)


def generate_pretraining_dataset(n_per_family: int, nx: int,
                                 families: list[str],
                                 dt_range: tuple = (0.005, 0.02),
                                 seed: int = 0) -> TensorDataset:
    """Concatenate ``n_per_family`` samples for each requested atomic family.

    Supports any subset of {"diffusion", "advection", "reaction"} so the same
    routine emits both the two-brick (E2.a-style) and three-brick libraries.
    """
    blocks = []
    for i, fam in enumerate(families):
        blocks.append(_make_pretrain_block(
            n_per_family, nx, fam, dt_range, seed=seed + 10 * (i + 1)))
    u0 = torch.cat([b.tensors[0] for b in blocks], dim=0)
    u1 = torch.cat([b.tensors[1] for b in blocks], dim=0)
    cond = torch.cat([b.tensors[2] for b in blocks], dim=0)
    return TensorDataset(u0, u1, cond)


def generate_composite_dataset(n_samples: int, nx: int, shear: float,
                               profile: torch.Tensor,
                               dt_range: tuple = (0.005, 0.02),
                               seed: int = 0) -> TensorDataset:
    """Allen-Cahn-with-drift target. cond = [dt, D, v0, gamma]."""
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n_samples, nx, seed=seed)
    D = _draw(rng, 0.01, 0.20, n_samples)
    v0 = _draw(rng, 0.10, 2.00, n_samples)
    gamma = _draw(rng, 0.5, 5.0, n_samples)
    dt = _draw(rng, *dt_range, n_samples)
    u1 = simulate_full_reaction(u0, D=D, v0=v0, gamma=gamma, dt=dt,
                                shear=shear, profile=profile)
    cond = torch.cat([dt, D, v0, gamma], dim=-1)
    return TensorDataset(u0, u1, cond)


# ---------------------------------------------------------------------------
# Training / evaluation (4-channel conditioning across all conditions)
# ---------------------------------------------------------------------------

def train(model, dataset, n_epochs, batch_size=64, lr=1e-3, device="cpu"):
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = float("nan")
    for _ in range(n_epochs):
        running = 0.0
        nb = 0
        for u0, u1, cond in loader:
            u0, u1, cond = u0.to(device), u1.to(device), cond.to(device)
            loss = F.mse_loss(model(u0, cond), u1)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); nb += 1
        last = running / max(nb, 1)
    return last


def rel_l2(pred, target):
    return ((pred - target).norm(dim=-1) /
            (target.norm(dim=-1) + 1e-8)).mean().item()


def evaluate(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=128)
    errs = []
    with torch.no_grad():
        for u0, u1, cond in loader:
            u0, u1, cond = u0.to(device), u1.to(device), cond.to(device)
            errs.append(rel_l2(model(u0, cond), u1))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--n_pretrain", type=int, default=10000,
                    help="Pretraining samples per atomic family.")
    ap.add_argument("--n_pretrain_epochs", type=int, default=800)
    ap.add_argument("--ns", nargs="+", type=int,
                    default=[200, 500, 1000, 2000, 5000],
                    help="Fine-tuning sample budgets to sweep.")
    ap.add_argument("--n_finetune_epochs", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="Seeds per (N, condition) pair.")
    ap.add_argument("--shear", type=float, default=0.5,
                    help="Velocity-shear amplitude of the downstream target.")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--n_modes", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_test", type=int, default=400,
                    help="Held-out composite test set size.")
    ap.add_argument("--profile_seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results_e2d")
    args = ap.parse_args()

    if args.smoke:
        args.n_pretrain = 200
        args.n_pretrain_epochs = 8
        args.ns = [100, 400]
        args.n_finetune_epochs = 8
        args.n_seeds = 1
        args.nx = 64
        args.width = 16
        args.n_modes = 8
        args.n_layers = 2
        args.batch_size = 32
        args.n_test = 40

    device = auto_device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    profile = shear_profile(args.nx, seed=args.profile_seed)
    print(f"Device: {device}   seed: {args.seed}   shear: {args.shear}")

    timer = Timer()

    # ---- [1/4] Two-brick pretraining ----
    print("\n[1/4] Two-brick pretraining (diffusion + advection) ...")
    with timer("pretrain_2brick"):
        pre2_ds = generate_pretraining_dataset(
            args.n_pretrain, args.nx,
            families=["diffusion", "advection"], seed=10)
        fno_pre2 = FNO1d(n_cond=4, width=args.width,
                         n_modes=args.n_modes, n_layers=args.n_layers)
        mse2 = train(fno_pre2, pre2_ds, n_epochs=args.n_pretrain_epochs,
                     batch_size=args.batch_size, lr=args.lr, device=device)
    state_pre2 = {k: v.clone().cpu() for k, v in fno_pre2.state_dict().items()}
    torch.save(state_pre2, out_dir / "pretrained_2brick.pt")
    print(f"  two-brick pretrain mse = {mse2:.2e}")

    # ---- [2/4] Three-brick pretraining ----
    print("\n[2/4] Three-brick pretraining (diffusion + advection + reaction) ...")
    with timer("pretrain_3brick"):
        pre3_ds = generate_pretraining_dataset(
            args.n_pretrain, args.nx,
            families=["diffusion", "advection", "reaction"], seed=20)
        fno_pre3 = FNO1d(n_cond=4, width=args.width,
                         n_modes=args.n_modes, n_layers=args.n_layers)
        mse3 = train(fno_pre3, pre3_ds, n_epochs=args.n_pretrain_epochs,
                     batch_size=args.batch_size, lr=args.lr, device=device)
    state_pre3 = {k: v.clone().cpu() for k, v in fno_pre3.state_dict().items()}
    torch.save(state_pre3, out_dir / "pretrained_3brick.pt")
    print(f"  three-brick pretrain mse = {mse3:.2e}")

    # ---- [3/4] Test set + zero-shot diagnostic ----
    print(f"\n[3/4] Composite test set (N_test = {args.n_test}, s = {args.shear}) ...")
    with timer("build_test"):
        test_ds = generate_composite_dataset(
            args.n_test, args.nx, args.shear, profile, seed=777)
    err_zero_2 = evaluate(fno_pre2.to(device), test_ds, device)
    err_zero_3 = evaluate(fno_pre3.to(device), test_ds, device)
    print(f"  two-brick zero-shot   rel_L2 = {err_zero_2:.4f}")
    print(f"  three-brick zero-shot rel_L2 = {err_zero_3:.4f}")

    # ---- [4/4] Fine-tune sweep ----
    n_runs = len(args.ns) * 3 * args.n_seeds
    print(f"\n[4/4] Fine-tune sweep: {len(args.ns)} budgets x 3 conditions "
          f"x {args.n_seeds} seeds = {n_runs} trainings")
    rows: list[dict] = []

    with timer("finetune_sweep"):
        for N in args.ns:
            train_ds = generate_composite_dataset(
                N, args.nx, args.shear, profile, seed=42)
            for seed in range(args.n_seeds):
                # (A) From scratch
                torch.manual_seed(seed)
                fno_a = FNO1d(n_cond=4, width=args.width,
                              n_modes=args.n_modes, n_layers=args.n_layers)
                train(fno_a, train_ds, n_epochs=args.n_finetune_epochs,
                      batch_size=args.batch_size, lr=args.lr, device=device)
                err_a = evaluate(fno_a, test_ds, device)

                # (B) Two-brick pretrained
                torch.manual_seed(seed)
                fno_b = FNO1d(n_cond=4, width=args.width,
                              n_modes=args.n_modes, n_layers=args.n_layers)
                fno_b.load_state_dict(state_pre2)
                train(fno_b, train_ds, n_epochs=args.n_finetune_epochs,
                      batch_size=args.batch_size, lr=args.lr, device=device)
                err_b = evaluate(fno_b, test_ds, device)

                # (C) Three-brick pretrained
                torch.manual_seed(seed)
                fno_c = FNO1d(n_cond=4, width=args.width,
                              n_modes=args.n_modes, n_layers=args.n_layers)
                fno_c.load_state_dict(state_pre3)
                train(fno_c, train_ds, n_epochs=args.n_finetune_epochs,
                      batch_size=args.batch_size, lr=args.lr, device=device)
                err_c = evaluate(fno_c, test_ds, device)

                rows.append({"N": N, "seed": seed,
                             "scratch": err_a, "two_brick": err_b,
                             "three_brick": err_c})
                sp2 = err_a / max(err_b, 1e-9)
                sp3 = err_a / max(err_c, 1e-9)
                print(f"  N={N:5d} seed={seed}  scratch={err_a:.4f}  "
                      f"two_brick={err_b:.4f} (x{sp2:.2f})  "
                      f"three_brick={err_c:.4f} (x{sp3:.2f})")

    # ---- Aggregate ----
    Ns = list(args.ns)
    agg = {"N": Ns}
    for key in ("scratch", "two_brick", "three_brick"):
        means = [float(np.mean([r[key] for r in rows if r["N"] == N]))
                 for N in Ns]
        stds = [float(np.std([r[key] for r in rows if r["N"] == N]))
                for N in Ns]
        agg[f"err_{key}_mean"] = means
        agg[f"err_{key}_std"] = stds
    sm = np.asarray(agg["err_scratch_mean"])
    b2 = np.asarray(agg["err_two_brick_mean"])
    b3 = np.asarray(agg["err_three_brick_mean"])
    speedup_2 = (sm / np.clip(b2, 1e-9, None)).tolist()
    speedup_3 = (sm / np.clip(b3, 1e-9, None)).tolist()
    agg["speedup_two_brick"] = speedup_2
    agg["speedup_three_brick"] = speedup_3

    print("\n  Speedup err_scratch / err_pretrained at each N:")
    for N, s2, s3 in zip(Ns, speedup_2, speedup_3):
        winner = "three" if s3 > s2 else "two"
        print(f"    N={N:5d}  two_brick={s2:.2f}x  three_brick={s3:.2f}x  "
              f"(winner: {winner})")
    three_wins = sum(1 for s2, s3 in zip(speedup_2, speedup_3) if s3 > s2)
    print(f"  three-brick beats two-brick at {three_wins} / {len(Ns)} budgets")

    timing = timer.dump()

    # ---- Plot: two panels (error curves; speedup curves) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    Ns_arr = np.asarray(Ns, dtype=float)
    ss = np.asarray(agg["err_scratch_std"])
    s2s = np.asarray(agg["err_two_brick_std"])
    s3s = np.asarray(agg["err_three_brick_std"])

    ax = axes[0]
    ax.plot(Ns_arr, sm, "o-", color="#c0392b", lw=2, ms=7, label="From scratch")
    ax.fill_between(Ns_arr, np.clip(sm - ss, 1e-6, None), sm + ss,
                    color="#c0392b", alpha=0.15)
    ax.plot(Ns_arr, b2, "s-", color="#2a6496", lw=2, ms=7,
            label="Two-brick pretrained (E2.a)")
    ax.fill_between(Ns_arr, np.clip(b2 - s2s, 1e-6, None), b2 + s2s,
                    color="#2a6496", alpha=0.15)
    ax.plot(Ns_arr, b3, "D-", color="#27ae60", lw=2, ms=7,
            label="Three-brick pretrained (E2d)")
    ax.fill_between(Ns_arr, np.clip(b3 - s3s, 1e-6, None), b3 + s3s,
                    color="#27ae60", alpha=0.15)
    ax.axhline(err_zero_2, ls=":", color="#7f8c8d", lw=1.0,
               label=f"two-brick zero-shot ({err_zero_2:.3f})")
    ax.axhline(err_zero_3, ls=":", color="#16a085", lw=1.0,
               label=f"three-brick zero-shot ({err_zero_3:.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Fine-tuning sample budget $N$")
    ax.set_ylabel(r"Relative $L^2$ test error")
    ax.set_title("E2d: Allen-Cahn-with-drift fine-tuning")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(Ns_arr, speedup_2, "s-", color="#2a6496", lw=2, ms=7,
            label="Two-brick speedup")
    ax.plot(Ns_arr, speedup_3, "D-", color="#27ae60", lw=2, ms=7,
            label="Three-brick speedup")
    ax.axhline(1.0, color="gray", ls="--", lw=1.0, label="break-even")
    ax.set_xscale("log")
    ax.set_xlabel(r"Fine-tuning sample budget $N$")
    ax.set_ylabel(r"$\mathrm{err}_{\mathrm{scratch}} / \mathrm{err}_{\mathrm{pretrained}}$")
    ax.set_title("Pretraining speedup (higher is better)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir, "e2d_three_brick_curriculum")

    # ---- Headline numbers ----
    headline = {
        "err_zero_shot_two_brick": err_zero_2,
        "err_zero_shot_three_brick": err_zero_3,
        "pretrain_mse_two_brick": mse2,
        "pretrain_mse_three_brick": mse3,
        "three_brick_wins_count": int(three_wins),
        "n_budgets": len(Ns),
    }
    for N, s2, s3 in zip(Ns, speedup_2, speedup_3):
        headline[f"speedup_2brick_N{N}"] = float(s2)
        headline[f"speedup_3brick_N{N}"] = float(s3)

    write_run_json(
        out_dir,
        experiment="e2d",
        pillar="Pillar 1 (operator splitting)",
        hypothesis=("Adding a cubic-reaction brick to the pretraining library "
                    "of E2.a lets a single FNO transfer to Allen-Cahn-with-drift "
                    "with strictly fewer fine-tuning samples than the two-brick "
                    "curriculum, at every N (generator-coverage view of "
                    "Conjecture 1)."),
        parameters=vars(args),
        headline=headline,
        timing=timing,
        device=device,
        extra={"aggregate": agg, "rows": rows},
    )

    print_summary("E2d", headline, timing)


if __name__ == "__main__":
    main()
