"""Experiment E3f (Pillar 2): heat equation on disks, LB-FNO vs flat 2D FNO.

Hypothesis:
  The Pillar 2 win demonstrated on rectangles in E3b 2D also holds on disks of
  varying radius. The LB-FNO (with Bessel-based eigenpairs computed in closed
  form from the disk radius R) should beat a standard 2D FNO by at least one
  order of magnitude across R in [0.4, 1.5].

PDE: partial_t u = D Laplacian u on a disk of radius R, Dirichlet BC.
Closed-form solution via the Bessel expansion
  u(r, theta, t) = sum_{m, n} c_{mn} J_m(z_{mn} r / R)
                              (cos / sin)(m theta) exp(-D (z_{mn} / R)^2 t)
where z_{mn} is the n-th positive zero of J_m. Implemented in
``geometry/disk_modes.py``.

Three methods compared, in direct analogy to E3b (2D):
  (A) LB-FNO (closed-form Bessel expansion), using the true R.
  (B) Standard 2D FNO trained on a single fixed radius R = 1.0; applied
      zero-shot to R != 1.
  (C) Standard 2D FNO trained on R in [R_min, R_max]; matched compute.

Output: relative L2 error vs R, with the training band marked. Pre-registered
prediction: curve A is one to two orders of magnitude below curves B, C on the
OOD radii.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry.disk_modes import (
    disk_grid, disk_mode_list, heat_dirichlet_disk, random_ic_disk,
)
from geometry.run_e3b_2d import HeatFNO2d, train_fno
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary,
    save_figure, write_run_json,
)


# ---------------------------------------------------------------------------
# Dataset generation: disk heat (u0, u1, dt) on a Cartesian bounding box
# ---------------------------------------------------------------------------

def make_disk_heat_dataset(R_list, D_range: tuple, dt_range: tuple,
                            n_samples: int, nx: int = 96,
                            K: int = 24, seed: int = 0) -> TensorDataset:
    """Generate (u0, u1, dt) triples for the disk heat equation.

    The fields are sampled on the Cartesian bounding box [-R, R] x [-R, R] at
    ``nx`` points per axis with values set to zero outside the disk; the
    sampling box scales with R so the disk fills the same fraction of the
    image (this is intentional, mirroring how E3b 2D scales the rectangle box).
    The ground-truth evolution uses ``heat_dirichlet_disk`` with K Bessel modes.
    """
    rng = np.random.default_rng(seed)
    R_list = list(R_list)
    cache: dict = {}
    u0s, u1s, dts = [], [], []
    for i in range(n_samples):
        R = float(R_list[int(rng.integers(0, len(R_list)))])
        D = float(rng.uniform(*D_range))
        dt = float(rng.uniform(*dt_range))
        u0 = random_ic_disk(R=R, nx=nx, seed=seed * 1_000_000 + i)
        u1 = heat_dirichlet_disk(u0, R=R, D=D, dt=dt, K=K, basis_cache=cache)
        u0s.append(u0)
        u1s.append(u1)
        dts.append(dt)
    return TensorDataset(
        torch.tensor(np.stack(u0s)),
        torch.tensor(np.stack(u1s)),
        torch.tensor(np.array(dts, dtype=np.float32)),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def rel_l2_2d(pred: torch.Tensor, target: torch.Tensor) -> float:
    flat = lambda t: t.reshape(t.size(0), -1)
    p, q = flat(pred), flat(target)
    return float((((p - q).norm(dim=-1)) / (q.norm(dim=-1) + 1e-8)).mean())


def eval_fno_at_R(model: HeatFNO2d, R: float, D_range, dt_eval: float,
                  nx: int, n_test: int, K: int, seed: int = 999) -> float:
    """Evaluate a standard 2D FNO at a specific disk radius (zero-shot OOD)."""
    model.eval()
    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    cache: dict = {}
    errs = []
    with torch.no_grad():
        for i in range(n_test):
            D = float(rng.uniform(*D_range))
            u0 = random_ic_disk(R=R, nx=nx, seed=seed * 1_000_000 + i)
            u1 = heat_dirichlet_disk(u0, R=R, D=D, dt=dt_eval, K=K,
                                     basis_cache=cache)
            u0t = torch.tensor(u0, device=dev).unsqueeze(0)
            u1t = torch.tensor(u1, device=dev).unsqueeze(0)
            pred = model(u0t, dt_eval)
            errs.append(rel_l2_2d(pred, u1t))
    return float(np.mean(errs))


def eval_lb_fno_at_R(R: float, D_range, dt_eval: float,
                     nx: int, n_test: int, K: int,
                     seed: int = 999) -> float:
    """Closed-form LB-FNO error: the Bessel expansion truncated to K modes.

    Ground truth uses a much richer Bessel expansion (K_ref >> K); the LB-FNO
    decodes with only K modes, so this curve measures the basis-truncation
    error rather than zero, which is the honest comparison against the trained
    standard FNO baselines.
    """
    K_ref = max(K * 3, K + 24)
    rng = np.random.default_rng(seed)
    cache_ref: dict = {}
    cache_lb: dict = {}
    errs = []
    for i in range(n_test):
        D = float(rng.uniform(*D_range))
        u0 = random_ic_disk(R=R, nx=nx, seed=seed * 1_000_000 + i)
        u_truth = heat_dirichlet_disk(u0, R=R, D=D, dt=dt_eval,
                                      K=K_ref, basis_cache=cache_ref)
        u_lb = heat_dirichlet_disk(u0, R=R, D=D, dt=dt_eval,
                                   K=K, basis_cache=cache_lb)
        num = float(np.linalg.norm(u_lb - u_truth))
        den = float(np.linalg.norm(u_truth)) + 1e-8
        errs.append(num / den)
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(R_list, errs_A, errs_B, errs_C, R_min, R_max, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(R_list, errs_A, "o-", color="#27ae60", lw=2,
            label="(A) LB-FNO (closed-form Bessel, K modes)")
    ax.plot(R_list, errs_B, "s--", color="#c0392b", lw=2,
            label="(B) Standard FNO trained on R = 1 only")
    ax.plot(R_list, errs_C, "^-.", color="#2a6496", lw=2,
            label="(C) Standard FNO trained on full R range (matched)")
    ax.axvspan(R_min, R_max, color="gray", alpha=0.1,
               label="training range")
    ax.set_xlabel(r"Disk radius $R$", fontsize=12)
    ax.set_ylabel(r"Relative $L^2$ error at eval_dt", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("E3f: disk heat equation, LB-FNO vs standard 2D FNO "
                 "(Phase A, Pillar 2)")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir, "e3f_disk_heat_ood")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--nx", type=int, default=96,
                    help="Grid resolution per axis on the bounding box.")
    ap.add_argument("--n_train", type=int, default=10000,
                    help="Training samples for each standard FNO.")
    ap.add_argument("--n_epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--width", type=int, default=48,
                    help="Channel width of the standard 2D FNO.")
    ap.add_argument("--n_modes_x", type=int, default=24)
    ap.add_argument("--n_modes_y", type=int, default=24)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--R_min", type=float, default=0.4)
    ap.add_argument("--R_max", type=float, default=1.5)
    ap.add_argument("--K", type=int, default=24,
                    help="Bessel modes kept in the LB-FNO decoder.")
    ap.add_argument("--eval_dt", type=float, default=0.01)
    ap.add_argument("--out_dir", default="results_e3f")
    args = ap.parse_args()

    if args.smoke:
        args.nx = 32
        args.n_train = 80
        args.n_epochs = 4
        args.batch = 16
        args.width = 12
        args.n_modes_x = 6
        args.n_modes_y = 6
        args.n_layers = 2
        args.n_test = 4
        args.K = 8

    device = auto_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    D_range = (0.01, 0.20)
    dt_range = (0.005, 0.02)
    dt_eval = args.eval_dt

    fno_kwargs = dict(width=args.width, n_layers=args.n_layers,
                      n_modes_x=args.n_modes_x, n_modes_y=args.n_modes_y)

    timer = Timer()

    # ---- Train FNO-B: single fixed radius R = 1.0 ----
    with timer("train_fno_B"):
        print(f"Training FNO-B (R = 1.0)  width={args.width}  "
              f"modes={args.n_modes_x}x{args.n_modes_y}")
        ds_B = make_disk_heat_dataset(
            [1.0], D_range, dt_range, args.n_train,
            nx=args.nx, K=args.K, seed=0)
        fno_B = HeatFNO2d(**fno_kwargs)
        train_fno(fno_B, ds_B, n_epochs=args.n_epochs,
                  batch=args.batch, device=device)

    # ---- Train FNO-C: all R in [R_min, R_max] ----
    with timer("train_fno_C"):
        print(f"Training FNO-C (R in [{args.R_min}, {args.R_max}])")
        R_train = np.linspace(args.R_min, args.R_max, 10).tolist()
        ds_C = make_disk_heat_dataset(
            R_train, D_range, dt_range, args.n_train,
            nx=args.nx, K=args.K, seed=1)
        fno_C = HeatFNO2d(**fno_kwargs)
        train_fno(fno_C, ds_C, n_epochs=args.n_epochs,
                  batch=args.batch, device=device)

    # ---- Sweep over R, evaluate all three methods ----
    if args.smoke:
        R_list = [0.5, 1.0, 1.4]
    else:
        R_list = [0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.8, 2.2]

    errs_A, errs_B, errs_C = [], [], []
    with timer("eval_sweep"):
        for R in R_list:
            errs_A.append(eval_lb_fno_at_R(
                R, D_range, dt_eval,
                nx=args.nx, n_test=args.n_test, K=args.K))
            errs_B.append(eval_fno_at_R(
                fno_B, R, D_range, dt_eval,
                nx=args.nx, n_test=args.n_test, K=args.K))
            errs_C.append(eval_fno_at_R(
                fno_C, R, D_range, dt_eval,
                nx=args.nx, n_test=args.n_test, K=args.K))
            print(f"  R = {R:>4.2f}  A (LB-FNO) = {errs_A[-1]:.4f}  "
                  f"B (R=1) = {errs_B[-1]:.4f}  "
                  f"C (full) = {errs_C[-1]:.4f}")

    timing = timer.dump()

    # ---- Plot ----
    make_plot(R_list, errs_A, errs_B, errs_C,
              args.R_min, args.R_max, out_dir)

    # ---- Headline numbers: in-band and OOD means, plus B/A ratio ----
    def _band_mean(curve, axis_vals, lo, hi, inside: bool):
        sel = [e for v, e in zip(axis_vals, curve)
               if (lo <= v <= hi) == inside]
        if not sel:
            return float("nan")
        return float(np.mean(sel))

    in_A = _band_mean(errs_A, R_list, args.R_min, args.R_max, inside=True)
    in_B = _band_mean(errs_B, R_list, args.R_min, args.R_max, inside=True)
    in_C = _band_mean(errs_C, R_list, args.R_min, args.R_max, inside=True)
    ood_A = _band_mean(errs_A, R_list, args.R_min, args.R_max, inside=False)
    ood_B = _band_mean(errs_B, R_list, args.R_min, args.R_max, inside=False)
    ood_C = _band_mean(errs_C, R_list, args.R_min, args.R_max, inside=False)

    headline = {
        "mean_err_A_inband": in_A,
        "mean_err_B_inband": in_B,
        "mean_err_C_inband": in_C,
        "mean_err_A_ood": ood_A,
        "mean_err_B_ood": ood_B,
        "mean_err_C_ood": ood_C,
        "ratio_B_over_A_inband": (in_B / in_A) if in_A > 0 else float("inf"),
        "ratio_C_over_A_inband": (in_C / in_A) if in_A > 0 else float("inf"),
    }

    write_run_json(
        out_dir,
        experiment="e3f",
        pillar="Pillar 2 (geometry-adaptive spectral basis)",
        hypothesis=("On disks of varying radius R the LB-FNO (closed-form "
                    "Bessel expansion with eigenvalues lambda_{mn} = "
                    "(z_{mn}/R)^2) beats a standard 2D FNO by at least one "
                    "order of magnitude across R in [0.4, 1.5]."),
        parameters=vars(args),
        headline=headline,
        timing=timing,
        device=device,
        extra={
            "R_list": R_list,
            "errs_A_lb_fno": errs_A,
            "errs_B_fno_R1": errs_B,
            "errs_C_fno_full": errs_C,
        },
    )

    print_summary("E3f", headline, timing)


if __name__ == "__main__":
    main()
