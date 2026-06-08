"""Experiment E5a (cross-pillar P1 cap P2): closed-form LB diffusion brick inside Strang.

Hypothesis. Replacing the neural diffusion brick of E2 with the closed-form
LB-spectral diffusion semigroup (Pillar 2) sharpens the commutator signal:
the data-efficiency gain at s = 0 jumps because the diffusion contribution
to per-brick error vanishes (the closed-form heat semigroup is exact in
Fourier space on the periodic torus). As the shear amplitude s grows, the
splitting error itself grows by Baker-Campbell-Hausdorff, so the closed-form
brick advantage is expected to shrink.

PDE family (identical to E2):
    A (diffusion):  d_t u = D d_xx u
    B (advection):  d_t u = -v(x) d_x u,    v(x) = v0 (1 + s p(x))

Method. Re-run the E2 data-efficiency sweep at s in {0.0, 0.5, 1.0}:
  (CF)   Strang(brick_diffusion_closed_form, FNO_advection)
         where brick_diffusion_closed_form is shared.simulators.brick_diffusion,
         which evaluates IFFT(exp(-D (2 pi k)^2 dt) FFT(u)); the
         "LB" eigenvalues on the periodic torus are lambda_k = (2 pi k)^2.
         Only the advection brick is learned.
  (NN)   Strang(FNO_diffusion, FNO_advection): the all-neural Strang of E2.
  (MONO) Monolithic FNO trained on the joint operator at matched compute.

The headline plot reports the closed-form-vs-all-neural data-efficiency gain
ratio at each shear; the pre-registered prediction is that the gain ratio
is largest at s = 0 and shrinks monotonically with s.

Reuses commutator.core (FNO1d, dataset generators, training, evaluation
helpers, data_efficiency_gain) and shared.splitting.strang verbatim. The
"closed-form brick" is just shared.simulators.brick_diffusion wrapped to the
splitting (u, dt) -> u signature, conditioned per-sample on D.

GPU-target runtime: about 45 minutes on a single GPU at default flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commutator.core import (
    FNO1d, data_efficiency_gain, evaluate_joint, generate_brick_dataset,
    generate_joint_dataset, generate_test_dataset, make_brick, rel_l2, train,
)
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary,
    save_figure, write_run_json,
)
from shared.simulators import brick_diffusion, shear_profile
from shared.splitting import strang


# ---------------------------------------------------------------------------
# Closed-form diffusion brick (LB spectral semigroup on periodic [0, 1))
# ---------------------------------------------------------------------------

def make_closed_form_diffusion_brick(D: torch.Tensor):
    """Wrap shared.simulators.brick_diffusion as a splitting brick (u, dt) -> u.

    D is the per-sample diffusion coefficient (shape (B,) or (B, 1)). On the
    periodic torus this is the exact heat semigroup
        u(t + dt) = IFFT(exp(-D (2 pi k)^2 dt) FFT(u)),
    which is the Laplace-Beltrami spectral form on [0, 1) with eigenvalues
    lambda_k = (2 pi k)^2 and eigenfunctions exp(i 2 pi k x). No training.
    """
    def brick(u: torch.Tensor, dt) -> torch.Tensor:
        D_col = D.to(u.device).to(u.dtype)
        if D_col.dim() == 1:
            D_col = D_col.view(-1, 1)
        dt_t = torch.as_tensor(float(dt), device=u.device, dtype=u.dtype)
        return brick_diffusion(u, D=D_col, dt=dt_t)
    return brick


# ---------------------------------------------------------------------------
# Strang evaluation with mixed (closed-form diffusion + neural advection) bricks
# ---------------------------------------------------------------------------

def evaluate_strang_closed_form(fno_adv, test_ds, eval_dt: float,
                                device: str) -> float:
    """Strang(brick_diff_closed_form, fno_adv) on the test set."""
    fno_adv.eval()
    loader = DataLoader(test_ds, batch_size=128)
    errors = []
    with torch.no_grad():
        for u0, u1, D, v0 in loader:
            u0 = u0.to(device); u1 = u1.to(device)
            brick_d = make_closed_form_diffusion_brick(D)
            brick_a = make_brick(fno_adv, v0)
            pred = strang(brick_d, brick_a, u0, eval_dt)
            errors.append(rel_l2(pred, u1))
    return float(np.mean(errors))


def evaluate_strang_all_neural(fno_diff, fno_adv, test_ds, eval_dt: float,
                               device: str) -> float:
    """Strang(fno_diff, fno_adv) on the test set (the E2 baseline path)."""
    fno_diff.eval(); fno_adv.eval()
    loader = DataLoader(test_ds, batch_size=128)
    errors = []
    with torch.no_grad():
        for u0, u1, D, v0 in loader:
            u0 = u0.to(device); u1 = u1.to(device)
            brick_d = make_brick(fno_diff, D)
            brick_a = make_brick(fno_adv, v0)
            pred = strang(brick_d, brick_a, u0, eval_dt)
            errors.append(rel_l2(pred, u1))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# One shear-level run
# ---------------------------------------------------------------------------

def run_one_shear(shear: float, args, profile: torch.Tensor,
                  device: str, timer: Timer) -> dict:
    """Train the bricks and the monolithic-baseline sweep at one shear value."""
    n_brick = args.n_brick
    n_composed = 2 * n_brick
    budgets = sorted(set(int(b) for b in args.budgets) | {n_composed})
    print(f"\n{'-' * 64}")
    print(f"  s = {shear}   n_brick = {n_brick}   matched-compute joint = {n_composed}")
    print(f"{'-' * 64}")

    seed_offset = int(round(1000.0 * shear))

    with timer(f"train_adv_s{shear}"):
        adv_ds = generate_brick_dataset(
            "advection", n_brick, args.nx, shear, profile,
            seed=20 + seed_offset)
        fno_adv = FNO1d(n_cond=2, width=args.width,
                        n_layers=args.n_layers, n_modes=args.n_modes)
        train(fno_adv, adv_ds, n_epochs=args.n_epochs,
              batch_size=args.batch, device=device)

    with timer(f"train_diff_s{shear}"):
        diff_ds = generate_brick_dataset(
            "diffusion", n_brick, args.nx, shear, profile,
            seed=10 + seed_offset)
        fno_diff = FNO1d(n_cond=2, width=args.width,
                         n_layers=args.n_layers, n_modes=args.n_modes)
        train(fno_diff, diff_ds, n_epochs=args.n_epochs,
              batch_size=args.batch, device=device)

    with timer(f"eval_strang_s{shear}"):
        test_ds = generate_test_dataset(
            args.n_test, args.nx, shear, profile, args.eval_dt,
            seed=999 + seed_offset)
        err_cf = evaluate_strang_closed_form(fno_adv, test_ds,
                                             args.eval_dt, device)
        err_nn = evaluate_strang_all_neural(fno_diff, fno_adv, test_ds,
                                            args.eval_dt, device)

    print(f"  Strang (CF diffusion + neural adv) rel-L2 = {err_cf:.4f}")
    print(f"  Strang (all neural)                rel-L2 = {err_nn:.4f}")

    joint_errs = []
    with timer(f"train_joint_sweep_s{shear}"):
        for b in budgets:
            joint_ds = generate_joint_dataset(
                b, args.nx, shear, profile,
                seed=40 + seed_offset + b)
            fno_joint = FNO1d(n_cond=3, width=args.width,
                              n_layers=args.n_layers, n_modes=args.n_modes)
            train(fno_joint, joint_ds, n_epochs=args.n_epochs,
                  batch_size=args.batch, device=device)
            e = evaluate_joint(fno_joint, test_ds, args.eval_dt, device)
            joint_errs.append(e)
            print(f"  joint FNO n={b:>5}  rel-L2 = {e:.4f}")

    qual_cf, gain_cf = data_efficiency_gain(budgets, joint_errs, err_cf, n_composed)
    qual_nn, gain_nn = data_efficiency_gain(budgets, joint_errs, err_nn, n_composed)
    ratio = gain_cf / (gain_nn + 1e-12)

    print(f"  Data-efficiency gain  CF: {qual_cf}{gain_cf:.2f}x  "
          f"NN: {qual_nn}{gain_nn:.2f}x  ratio CF/NN = {ratio:.2f}")

    return {
        "shear": shear,
        "n_composed": n_composed,
        "budgets": budgets,
        "joint_errs": joint_errs,
        "err_cf": err_cf,
        "err_nn": err_nn,
        "gain_cf": gain_cf,
        "gain_cf_qual": qual_cf,
        "gain_nn": gain_nn,
        "gain_nn_qual": qual_nn,
        "gain_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--shears", nargs="+", type=float,
                    default=[0.0, 0.5, 1.0])
    ap.add_argument("--n_brick", type=int, default=4000,
                    help="Training samples per brick (advection brick only learned).")
    ap.add_argument("--n_joint", type=int, default=8000,
                    help="Maximum monolithic-baseline sample budget.")
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[1000, 2000, 4000, 8000],
                    help="Joint-FNO sample budgets for the data-efficiency sweep.")
    ap.add_argument("--n_epochs", type=int, default=250)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_modes", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval_dt", type=float, default=0.01)
    ap.add_argument("--profile_seed", type=int, default=12345)
    ap.add_argument("--out_dir", default="results_e5a")
    args = ap.parse_args()

    if args.smoke:
        args.shears = [0.0, 1.0]
        args.n_brick = 200
        args.n_joint = 400
        args.budgets = [100, 200, 400]
        args.n_epochs = 6
        args.n_test = 40
        args.nx = 64
        args.width = 16
        args.n_modes = 8
        args.n_layers = 2
        args.batch = 32

    device = auto_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}   seed: {args.seed}")
    print(f"Shears: {args.shears}   n_brick: {args.n_brick}   "
          f"budgets: {args.budgets}   n_epochs: {args.n_epochs}")

    timer = Timer()
    profile = shear_profile(nx=args.nx, n_modes=3, seed=args.profile_seed)

    results = [run_one_shear(s, args, profile, device, timer)
               for s in args.shears]

    timing = timer.dump()

    shears = [r["shear"] for r in results]
    gain_cf = [r["gain_cf"] for r in results]
    gain_nn = [r["gain_nn"] for r in results]
    err_cf = [r["err_cf"] for r in results]
    err_nn = [r["err_nn"] for r in results]
    ratios = [r["gain_ratio"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(shears, gain_cf, "o-", color="#27ae60", lw=2,
                 label="Strang (closed-form diffusion + neural advection)")
    axes[0].plot(shears, gain_nn, "s--", color="#c0392b", lw=2,
                 label="Strang (all neural, E2 baseline)")
    axes[0].axhline(1.0, color="gray", linestyle=":", label="break-even")
    axes[0].set_xlabel(r"Shear amplitude $s$ (proxy for $\|[A, B]\|$)")
    axes[0].set_ylabel("Data-efficiency gain (x)")
    axes[0].set_yscale("log")
    axes[0].set_title("E5a: closed-form LB diffusion brick vs all-neural Strang")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(shears, ratios, "D-", color="#8e44ad", lw=2,
                 label="Gain ratio CF / NN")
    axes[1].axhline(1.0, color="gray", linestyle=":", label="no advantage")
    axes[1].set_xlabel(r"Shear amplitude $s$")
    axes[1].set_ylabel("Gain ratio (CF gain / NN gain)")
    axes[1].set_title("Closed-form advantage shrinks as commutator grows")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir, "e5a_closed_form_diffusion_brick")

    headline = {}
    for r in results:
        s = r["shear"]
        headline[f"err_cf_s{s}"] = r["err_cf"]
        headline[f"err_nn_s{s}"] = r["err_nn"]
        headline[f"gain_cf_s{s}"] = r["gain_cf"]
        headline[f"gain_nn_s{s}"] = r["gain_nn"]
        headline[f"gain_ratio_s{s}"] = r["gain_ratio"]

    write_run_json(out_dir,
                   experiment="e5a",
                   pillar="Cross-pillar P1 cap P2 (operator splitting with geometry brick)",
                   hypothesis=("Replacing the neural diffusion brick of E2 with the "
                               "closed-form LB-spectral diffusion semigroup sharpens "
                               "the commutator signal: the data-efficiency gain at "
                               "s = 0 jumps because the diffusion contribution to "
                               "per-brick error vanishes, and the closed-form "
                               "advantage shrinks monotonically as the shear grows."),
                   parameters=vars(args),
                   headline=headline,
                   timing=timing,
                   device=device,
                   extra={"results_per_shear": results,
                          "shears": shears,
                          "err_cf": err_cf,
                          "err_nn": err_nn,
                          "gain_cf": gain_cf,
                          "gain_nn": gain_nn,
                          "gain_ratio": ratios})

    print_summary("E5a", headline, timing)


if __name__ == "__main__":
    main()
