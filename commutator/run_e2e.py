"""Experiment E2e (Pillar 1): semilinear (Crandall-Liggett-Brezis-Pazy) regime.

In the semilinear regime  ``du/dt = D u_xx - gamma u (1 - u^2)``  the linear
diffusion generator ``A = D d_xx`` is paired with a Lipschitz cubic reaction
``B(u) = -gamma u (1 - u^2)``. The pure BCH expansion no longer applies (B is
nonlinear), but the Crandall-Liggett-Brezis-Pazy theorem still guarantees that
Lie-Trotter converges at first order, with a constant proportional to the
Lipschitz constant of B.

Pre-registered prediction: the composed two-brick model still beats the
monolithic baseline on data efficiency in exactly the regime where the proposal
claims the weaker nonlinear semigroup theory holds, with the gain decreasing in
``gamma`` (which controls the reaction Lipschitz constant).

Sweep: ``gamma in {0.5, 2.0, 5.0}``. For each gamma, two bricks are trained
once (a parameter-conditioned diffusion FNO from E2 and a small pointwise MLP
reaction brick) and then plugged into Lie-Trotter and Strang. The monolithic
baseline is a single FNO conditioned on ``(dt, D, gamma)`` and trained on the
joint semilinear PDE at matched compute.

Headline metric: ``samples-to-match`` data-efficiency gain (how many joint
samples the monolithic baseline needs to reach the composed model's accuracy,
divided by the composed model's training samples). The plot reports gain vs
gamma on a log-y axis.

GPU-target runtime: about 1 hour on a single GPU at default flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commutator.core import (
    FNO1d, rel_l2, train,
)
from commutator.semilinear import brick_reaction, simulate_semilinear
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary,
    save_figure, write_run_json,
)
from shared.simulators import batch_random_ics, brick_diffusion


# ---------------------------------------------------------------------------
# Reaction brick: small pointwise MLP conditioned on (dt, gamma)
# ---------------------------------------------------------------------------

class PointwiseReactionMLP(nn.Module):
    """Per-pixel MLP that approximates  ``u(t+dt) = phi(u; gamma, dt)``.

    The reaction ODE is pointwise (no spatial coupling), so a 1x1 conv stack
    on the (u, dt, gamma) channels is the natural architecture and far cheaper
    than an FNO. Output shape matches the input shape ``(B, nx)``.
    """

    def __init__(self, hidden: int = 32, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv1d(3, hidden, 1), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv1d(hidden, hidden, 1), nn.GELU()]
        layers += [nn.Conv1d(hidden, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # u: (B, nx);   cond: (B, 2)  with columns (dt, gamma)
        cond_ch = cond.view(cond.size(0), cond.size(1), 1).expand(-1, -1, u.size(-1))
        x = torch.cat([u.unsqueeze(1), cond_ch.to(u.dtype)], dim=1)  # (B, 3, nx)
        return self.net(x).squeeze(1)


# ---------------------------------------------------------------------------
# Brick adapters for the splitting helpers
# ---------------------------------------------------------------------------

def make_diff_brick(model: nn.Module, D: torch.Tensor):
    """Return a callable  ``brick(u, dt) -> u``  with D frozen per-sample."""
    def brick(u: torch.Tensor, dt) -> torch.Tensor:
        dt_col = torch.full((u.size(0),), float(dt), device=u.device, dtype=u.dtype)
        cond = torch.stack([dt_col, D.to(u.device).to(u.dtype)], dim=-1)
        return model(u, cond)
    return brick


def make_reaction_brick(model: nn.Module, gamma: torch.Tensor):
    """Return a callable  ``brick(u, dt) -> u``  with gamma frozen per-sample."""
    def brick(u: torch.Tensor, dt) -> torch.Tensor:
        dt_col = torch.full((u.size(0),), float(dt), device=u.device, dtype=u.dtype)
        cond = torch.stack([dt_col, gamma.to(u.device).to(u.dtype)], dim=-1)
        return model(u, cond)
    return brick


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _draw(rng, lo, hi, n):
    return torch.tensor(rng.uniform(lo, hi, n), dtype=torch.float32).unsqueeze(-1)


def generate_diffusion_dataset(n_samples: int, nx: int,
                               dt_range=(0.005, 0.02),
                               seed: int = 1) -> TensorDataset:
    """Diffusion brick dataset; cond = (dt, D)."""
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n_samples, nx, seed=seed)
    D = _draw(rng, 0.01, 0.20, n_samples)
    dt = _draw(rng, *dt_range, n_samples)
    u1 = brick_diffusion(u0, D=D, dt=dt)
    cond = torch.cat([dt, D], dim=-1)
    return TensorDataset(u0, u1, cond)


def generate_reaction_dataset(n_samples: int, nx: int, gamma_range,
                              dt_range=(0.005, 0.02),
                              seed: int = 2) -> TensorDataset:
    """Reaction brick dataset; cond = (dt, gamma).

    ``gamma_range`` is the (lo, hi) interval to sample from; the trained MLP
    has to cover the whole sweep ``{0.5, 2.0, 5.0}`` so we draw uniformly
    across it.
    """
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n_samples, nx, seed=seed)
    gamma = _draw(rng, *gamma_range, n_samples)
    dt = _draw(rng, *dt_range, n_samples)
    u1 = brick_reaction(u0, gamma=gamma, dt=dt)
    cond = torch.cat([dt, gamma], dim=-1)
    return TensorDataset(u0, u1, cond)


def generate_joint_dataset(n_samples: int, nx: int, gamma_range,
                           dt_range=(0.005, 0.02),
                           seed: int = 3) -> TensorDataset:
    """Joint semilinear-PDE dataset; cond = (dt, D, gamma)."""
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n_samples, nx, seed=seed)
    D = _draw(rng, 0.01, 0.20, n_samples)
    gamma = _draw(rng, *gamma_range, n_samples)
    dt = _draw(rng, *dt_range, n_samples)
    u1 = simulate_semilinear(u0, D=D, gamma=gamma, dt=dt)
    cond = torch.cat([dt, D, gamma], dim=-1)
    return TensorDataset(u0, u1, cond)


def generate_test_dataset(n_samples: int, nx: int, gamma_val: float,
                          eval_dt: float, seed: int = 7) -> TensorDataset:
    """Test set at fixed ``gamma_val`` and ``eval_dt``; stores (u0, u1, D, gamma)."""
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n_samples, nx, seed=seed)
    D = _draw(rng, 0.01, 0.20, n_samples)
    gamma = torch.full((n_samples, 1), float(gamma_val))
    u1 = simulate_semilinear(u0, D=D, gamma=gamma, dt=eval_dt)
    return TensorDataset(u0, u1, D.squeeze(-1), gamma.squeeze(-1))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_composed(fno_diff: nn.Module, mlp_reaction: nn.Module,
                      test_ds: TensorDataset, eval_dt: float,
                      device: str, scheme: str) -> float:
    """Run one composition scheme over the whole test set and return rel-L2."""
    fno_diff.eval(); mlp_reaction.eval()
    loader = DataLoader(test_ds, batch_size=128)
    errors = []
    with torch.no_grad():
        for u0, u1, D, gamma in loader:
            u0 = u0.to(device); u1 = u1.to(device)
            D = D.to(device); gamma = gamma.to(device)
            brick_d = make_diff_brick(fno_diff, D)
            brick_r = make_reaction_brick(mlp_reaction, gamma)
            if scheme == "LT":
                # Lie-Trotter: diffusion then reaction
                u = brick_d(u0, eval_dt)
                pred = brick_r(u, eval_dt)
            elif scheme == "ST":
                # Strang: diffusion(dt/2), reaction(dt), diffusion(dt/2)
                u = brick_d(u0, eval_dt / 2.0)
                u = brick_r(u, eval_dt)
                pred = brick_d(u, eval_dt / 2.0)
            else:
                raise ValueError(scheme)
            errors.append(rel_l2(pred, u1))
    return float(np.mean(errors))


def evaluate_monolithic(fno_joint: nn.Module, test_ds: TensorDataset,
                        eval_dt: float, device: str) -> float:
    fno_joint.eval()
    loader = DataLoader(test_ds, batch_size=128)
    errors = []
    with torch.no_grad():
        for u0, u1, D, gamma in loader:
            u0 = u0.to(device); u1 = u1.to(device)
            D = D.to(device); gamma = gamma.to(device)
            cond = torch.stack(
                [torch.full_like(D, eval_dt), D, gamma], dim=-1)
            errors.append(rel_l2(fno_joint(u0, cond), u1))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Data-efficiency gain
# ---------------------------------------------------------------------------

def data_efficiency_gain(budgets, joint_errs, composed_err, n_composed):
    """Samples-to-match efficiency gain (log-log interpolation of the crossing).

    Returns (qualifier, gain) where qualifier is "=", ">=" or "<": "=" if the
    crossing falls inside the budget grid, ">=" if the joint baseline never
    matches the composed error (gain is a lower bound), "<" if the joint
    already matches at the smallest budget.
    """
    b = np.asarray(budgets, dtype=float)
    e = np.minimum.accumulate(np.asarray(joint_errs, dtype=float))
    below = e <= composed_err
    if not below.any():
        return ">=", b[-1] / n_composed
    idx = int(np.argmax(below))
    if idx == 0:
        return "<", b[0] / n_composed
    b0, b1, e0, e1 = b[idx - 1], b[idx], e[idx - 1], e[idx]
    t = (np.log(composed_err) - np.log(e0)) / (np.log(e1) - np.log(e0) + 1e-12)
    b_star = float(np.exp(np.log(b0) + t * (np.log(b1) - np.log(b0))))
    return "=", b_star / n_composed


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--gammas", nargs="+", type=float,
                    default=[0.5, 2.0, 5.0])
    ap.add_argument("--n_brick", type=int, default=4000,
                    help="Training samples per brick.")
    ap.add_argument("--n_joint", type=int, default=8000,
                    help="Training samples for the monolithic baseline at "
                         "matched compute.")
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[1000, 2000, 4000, 8000, 16000],
                    help="Joint-FNO sample budgets for the data-efficiency "
                         "sweep (used to estimate samples-to-match gain).")
    ap.add_argument("--n_epochs", type=int, default=250)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_modes", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mlp_hidden", type=int, default=32,
                    help="Hidden width of the pointwise reaction MLP.")
    ap.add_argument("--mlp_layers", type=int, default=3)
    ap.add_argument("--eval_dt", type=float, default=0.01)
    ap.add_argument("--out_dir", default="results_e2e")
    args = ap.parse_args()

    if args.smoke:
        args.gammas = [0.5, 5.0]
        args.n_brick = 200
        args.n_joint = 200
        args.budgets = [100, 200, 400]
        args.n_epochs = 8
        args.n_test = 40
        args.nx = 64
        args.width = 16
        args.n_modes = 8
        args.n_layers = 2
        args.batch = 32
        args.mlp_hidden = 16
        args.mlp_layers = 2

    device = auto_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Gammas: {args.gammas}   n_brick: {args.n_brick}   "
          f"n_joint: {args.n_joint}   n_epochs: {args.n_epochs}")

    timer = Timer()

    # The brick bricks see gammas only across the swept interval; padding the
    # range slightly avoids edge effects at the smallest / largest gamma.
    gamma_lo = min(args.gammas) * 0.5
    gamma_hi = max(args.gammas) * 1.1
    gamma_range = (gamma_lo, gamma_hi)

    # ---- Brick training (shared across gammas) ----
    with timer("train_bricks"):
        print("\nTraining diffusion brick ...")
        diff_ds = generate_diffusion_dataset(
            args.n_brick, args.nx, seed=10)
        fno_diff = FNO1d(n_cond=2, width=args.width,
                         n_layers=args.n_layers, n_modes=args.n_modes)
        train(fno_diff, diff_ds, n_epochs=args.n_epochs,
              batch_size=args.batch, device=device)

        print("Training reaction brick (pointwise MLP) ...")
        react_ds = generate_reaction_dataset(
            args.n_brick, args.nx, gamma_range, seed=20)
        mlp_reaction = PointwiseReactionMLP(
            hidden=args.mlp_hidden, n_layers=args.mlp_layers)
        train(mlp_reaction, react_ds, n_epochs=args.n_epochs,
              batch_size=args.batch, device=device)

    # The composed model effectively uses ``2 * n_brick`` training samples
    # (one set per brick); the joint baseline at matched compute consumes
    # ``n_joint`` samples directly on the semilinear PDE.
    n_composed = 2 * args.n_brick

    # ---- Per-gamma evaluation and joint sample-budget sweep ----
    results: dict[float, dict[str, float]] = {}
    for gamma_val in args.gammas:
        print(f"\n--- gamma = {gamma_val} ---")
        with timer(f"eval_g{gamma_val}"):
            test_ds = generate_test_dataset(
                args.n_test, args.nx, gamma_val, args.eval_dt,
                seed=700 + int(100 * gamma_val))
            err_lt = evaluate_composed(
                fno_diff, mlp_reaction, test_ds, args.eval_dt, device, "LT")
            err_st = evaluate_composed(
                fno_diff, mlp_reaction, test_ds, args.eval_dt, device, "ST")
            composed_err = min(err_lt, err_st)

        # Joint FNO budget sweep at this gamma (a fresh joint dataset per
        # budget; the gamma channel of the joint conditioning is fixed to
        # the swept value so the baseline is matched at each gamma).
        budgets = sorted(set(int(b) for b in args.budgets) | {args.n_joint})
        joint_errs = []
        for b in budgets:
            with timer(f"train_joint_b{b}_g{gamma_val}"):
                joint_ds = generate_joint_dataset(
                    b, args.nx, (gamma_val, gamma_val),
                    seed=4000 + b + int(100 * gamma_val))
                fno_joint = FNO1d(n_cond=3, width=args.width,
                                  n_layers=args.n_layers, n_modes=args.n_modes)
                train(fno_joint, joint_ds, n_epochs=args.n_epochs,
                      batch_size=args.batch, device=device)
                e = evaluate_monolithic(
                    fno_joint, test_ds, args.eval_dt, device)
                joint_errs.append(e)
                print(f"  joint FNO  n={b:>6}   rel-L2 = {e:.4f}")

        err_joint_matched = joint_errs[budgets.index(args.n_joint)]
        accuracy_ratio = err_joint_matched / (composed_err + 1e-12)
        qual, gain = data_efficiency_gain(
            budgets, joint_errs, composed_err, n_composed)

        results[gamma_val] = {
            "err_lt": err_lt,
            "err_st": err_st,
            "composed_err": composed_err,
            "err_joint_matched": err_joint_matched,
            "accuracy_ratio": accuracy_ratio,
            "gain": gain,
            "gain_qual": qual,
            "budgets": budgets,
            "joint_errs": joint_errs,
        }
        print(f"  Lie-Trotter            rel-L2 = {err_lt:.4f}")
        print(f"  Strang                 rel-L2 = {err_st:.4f}")
        print(f"  Joint (matched, n={args.n_joint})  rel-L2 = "
              f"{err_joint_matched:.4f}")
        print(f"  Accuracy ratio (joint / composed) = {accuracy_ratio:.2f}x")
        print(f"  Data-efficiency gain (samples-to-match) = "
              f"{qual}{gain:.2f}x")

    timing = timer.dump()

    # ---- Plot: gain vs gamma (log y) + absolute errors panel ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    gs = list(args.gammas)
    gains = [results[g]["gain"] for g in gs]
    axes[0].plot(gs, gains, "o-", color="#2a6496", lw=2, ms=8,
                 label="data-efficiency gain")
    axes[0].axhline(1.0, color="gray", linestyle="--", label="break-even")
    axes[0].set_xlabel(r"$\gamma$  (reaction Lipschitz scale)")
    axes[0].set_ylabel("data-efficiency gain (x)")
    axes[0].set_yscale("log")
    axes[0].set_title("E2e: composed vs monolithic data-efficiency gain")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(gs, [results[g]["err_lt"] for g in gs], "s--",
                 color="#27ae60", label="Lie-Trotter composed")
    axes[1].plot(gs, [results[g]["err_st"] for g in gs], "^-.",
                 color="#8e44ad", label="Strang composed")
    axes[1].plot(gs, [results[g]["err_joint_matched"] for g in gs], "o-",
                 color="#c0392b", label=f"Joint FNO (n={args.n_joint})")
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel(r"relative $L^2$ error")
    axes[1].set_yscale("log")
    axes[1].set_title("Absolute errors vs gamma")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, out_dir, "e2e_semilinear_clbp")

    # ---- Headline payload ----
    headline: dict[str, float] = {}
    for g in gs:
        r = results[g]
        headline[f"err_lt_g{g}"] = r["err_lt"]
        headline[f"err_st_g{g}"] = r["err_st"]
        headline[f"err_joint_g{g}"] = r["err_joint_matched"]
        headline[f"gain_g{g}"] = r["gain"]

    write_run_json(
        out_dir,
        experiment="e2e",
        pillar="Pillar 1 (operator splitting)",
        hypothesis=(
            "In the semilinear regime (linear diffusion plus Lipschitz cubic "
            "reaction) Lie-Trotter still converges at first order by the "
            "Crandall-Liggett-Brezis-Pazy theorem, so the two-brick composed "
            "model should retain a data-efficiency gain over the monolithic "
            "FNO, with the gain shrinking as gamma (reaction Lipschitz "
            "constant) grows."),
        parameters=vars(args),
        headline=headline,
        timing=timing,
        device=device,
        extra={"results_per_gamma": results,
               "n_composed": n_composed,
               "gamma_train_range": list(gamma_range)},
    )

    print_summary("E2e", headline, timing)


if __name__ == "__main__":
    main()
