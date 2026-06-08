"""Experiment E3h (Pillar 2): perforated rectangles.

Hypothesis. Adding a single circular hole to a rectangle perturbs the
Dirichlet spectrum predictably: eigenvalues shift up by an amount
proportional to the hole-area fraction (a Rayleigh-quotient perturbation).
The encoder should learn this perturbation when fine-tuned on a small
perforated-rectangle library.

Two encoders are compared:
  (i)  a baseline encoder that ignores the hole: the E3a rectangle encoder
       applied to the bounding rectangle ``[0, a] x [0, b]``.  It is trained
       on closed-form rectangle eigenvalues exactly as in E3a.
  (ii) a small dedicated encoder that consumes the full 5-vector descriptor
       ``(a, b, cx, cy, r)`` directly, trained on FEM-on-grid reference
       eigenvalues of a perforated-rectangle library.

Reference eigenvalues come from a sparse 5-point Laplacian eigensolve on a
uniform Cartesian grid that masks out the hole pixels (see
``geometry/perforated.py``).  No external mesh tools are required.

Output: per-mode relative error vs hole-area fraction for both encoders,
plus a scatter of (area-fraction, mean error over the K modes).

Runtime: pre-registered at about 40 minutes on a single GPU at default
flags; the FEM eigensolves themselves are CPU work and dominate the budget.
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

from geometry.encoder import EigenvalueEncoder
from geometry.lb_truth import rectangle_eigenvalues
from geometry.perforated import (
    PerforatedRect, perforated_eigenvalues, sample_perforated_library,
)
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary,
    save_figure, write_run_json,
)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def rectangle_pretrain_dataset(n: int, K: int,
                                a_range: tuple = (0.5, 1.5),
                                b_range: tuple = (0.5, 1.5),
                                seed: int = 0) -> TensorDataset:
    """Closed-form rectangle eigenvalues for the baseline encoder."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(*a_range, size=n).astype(np.float32)
    b = rng.uniform(*b_range, size=n).astype(np.float32)
    eigs = np.stack([rectangle_eigenvalues(float(ai), float(bi), K)
                     for ai, bi in zip(a, b)]).astype(np.float32)
    return TensorDataset(
        torch.tensor(np.stack([a, b], axis=1)),
        torch.tensor(eigs),
    )


def perforated_dataset(descs: np.ndarray, eigs: np.ndarray) -> TensorDataset:
    """Wrap precomputed (descriptor, eigenvalue) arrays as a TensorDataset.

    The descriptor's r column is shifted by a small floor so the log-space
    encoder can handle the ``r = 0`` corner without taking ``log(0)``.
    """
    x = descs.copy()
    x[:, 4] = np.maximum(x[:, 4], 1e-3)
    return TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(eigs, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: torch.nn.Module, ds: TensorDataset, *,
          n_epochs: int, batch: int, lr: float, device: str) -> list[float]:
    model.to(device)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, n_epochs))
    hist: list[float] = []
    for _ in range(n_epochs):
        running = 0.0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(torch.log(pred), torch.log(y))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        sched.step()
        hist.append(running / max(1, len(loader)))
    return hist


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _predict(model: torch.nn.Module, x_np: np.ndarray) -> np.ndarray:
    model.eval()
    dev = _device_of(model)
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)
    with torch.no_grad():
        return model(x).cpu().numpy()


def rel_err_per_mode(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean per-mode relative error, shape (K,)."""
    return np.abs(pred - true).mean(axis=0) / (true.mean(axis=0) + 1e-12)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--K", type=int, default=10,
                    help="Number of leading eigenvalues to predict.")
    ap.add_argument("--n_rect_pretrain", type=int, default=2000,
                    help="Plain-rectangle samples for the baseline encoder.")
    ap.add_argument("--n_rect_pretrain_epochs", type=int, default=600)
    ap.add_argument("--n_perf_train", type=int, default=2000,
                    help="Perforated-rectangle training samples.")
    ap.add_argument("--n_perf_test", type=int, default=200)
    ap.add_argument("--n_epochs", type=int, default=600,
                    help="Epochs for the perforated-rectangle encoder.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--fem_nx", type=int, default=64,
                    help="FEM-on-grid resolution per axis for ref eigenvalues.")
    ap.add_argument("--n_area_bins", type=int, default=6,
                    help="Hole-area-fraction bins for the per-bin error plot.")
    ap.add_argument("--out_dir", default="results_e3h")
    args = ap.parse_args()

    if args.smoke:
        args.K = 6
        args.n_rect_pretrain = 200
        args.n_rect_pretrain_epochs = 30
        args.n_perf_train = 60
        args.n_perf_test = 24
        args.n_epochs = 30
        args.batch = 16
        args.fem_nx = 28
        args.n_area_bins = 4

    device = auto_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    timer = Timer()

    # -----------------------------------------------------------------
    # 1. Baseline: pretrain the plain-rectangle encoder (ignores the hole).
    # -----------------------------------------------------------------
    with timer("baseline_pretrain"):
        ds_rect = rectangle_pretrain_dataset(
            args.n_rect_pretrain, args.K, seed=args.seed + 11,
        )
        baseline = EigenvalueEncoder(d_in=2, K=args.K)
        train(baseline, ds_rect,
              n_epochs=args.n_rect_pretrain_epochs, batch=args.batch,
              lr=args.lr, device=device)

    # -----------------------------------------------------------------
    # 2. Build the perforated-rectangle train and test libraries.
    # -----------------------------------------------------------------
    with timer("build_perf_train"):
        train_descs, train_eigs, _ = sample_perforated_library(
            n=args.n_perf_train, K=args.K,
            seed=args.seed + 101,
            nx=args.fem_nx, ny=args.fem_nx,
        )
    with timer("build_perf_test"):
        test_descs, test_eigs, test_areas = sample_perforated_library(
            n=args.n_perf_test, K=args.K,
            seed=args.seed + 202,
            nx=args.fem_nx, ny=args.fem_nx,
        )

    # Filter any degenerate (NaN) samples from the FEM eigensolves.
    valid_train = ~np.isnan(train_eigs).any(axis=1)
    valid_test = ~np.isnan(test_eigs).any(axis=1)
    train_descs = train_descs[valid_train]
    train_eigs = train_eigs[valid_train]
    test_descs = test_descs[valid_test]
    test_eigs = test_eigs[valid_test]
    test_areas = test_areas[valid_test]
    print(f"Perforated library: train={len(train_eigs)}  test={len(test_eigs)}")

    # -----------------------------------------------------------------
    # 3. Train the perforated-rectangle encoder (5-vector input).
    # -----------------------------------------------------------------
    with timer("perf_train"):
        ds_perf = perforated_dataset(train_descs, train_eigs)
        perf_model = EigenvalueEncoder(d_in=5, K=args.K)
        hist = train(perf_model, ds_perf,
                     n_epochs=args.n_epochs, batch=args.batch,
                     lr=args.lr, device=device)
        print(f"  perf encoder final log-MSE: {hist[-1]:.5f}")

    # -----------------------------------------------------------------
    # 4. Evaluate both encoders on the test library.
    # -----------------------------------------------------------------
    with timer("eval"):
        # Baseline sees only (a, b).
        pred_baseline = _predict(baseline, test_descs[:, :2])
        # Perforated encoder sees the full 5-vector (with the r-floor applied).
        x_perf = test_descs.copy()
        x_perf[:, 4] = np.maximum(x_perf[:, 4], 1e-3)
        pred_perf = _predict(perf_model, x_perf)

        rel_baseline = np.abs(pred_baseline - test_eigs) / (test_eigs + 1e-12)
        rel_perf = np.abs(pred_perf - test_eigs) / (test_eigs + 1e-12)

        err_baseline_per_mode = rel_err_per_mode(pred_baseline, test_eigs)
        err_perf_per_mode = rel_err_per_mode(pred_perf, test_eigs)

        # Per-area-bin mean error (across modes).
        area_edges = np.linspace(0.0, max(0.001, float(test_areas.max())),
                                  args.n_area_bins + 1)
        area_centres = 0.5 * (area_edges[:-1] + area_edges[1:])
        bin_baseline = np.zeros(args.n_area_bins)
        bin_perf = np.zeros(args.n_area_bins)
        bin_counts = np.zeros(args.n_area_bins, dtype=int)
        bin_idx = np.clip(
            np.digitize(test_areas, area_edges) - 1, 0, args.n_area_bins - 1)
        for i in range(args.n_area_bins):
            sel = bin_idx == i
            bin_counts[i] = int(sel.sum())
            if bin_counts[i] > 0:
                bin_baseline[i] = float(rel_baseline[sel].mean())
                bin_perf[i] = float(rel_perf[sel].mean())

    # -----------------------------------------------------------------
    # 5. Plot: per-mode bars, plus per-area-bin curves.
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # Per-mode bars: baseline.
    modes = np.arange(1, args.K + 1)
    width = 0.4
    ax = axes[0]
    ax.bar(modes - width / 2, err_baseline_per_mode, width,
           color="#c0392b", alpha=0.85, label="baseline (ignores hole)")
    ax.bar(modes + width / 2, err_perf_per_mode, width,
           color="#27ae60", alpha=0.85, label="fine-tuned (a, b, cx, cy, r)")
    ax.axhline(0.05, color="black", linestyle="--", lw=1.0,
               label="5 % threshold")
    ax.set_xlabel("Eigenmode k")
    ax.set_ylabel("Mean relative error")
    ax.set_title("E3h: per-mode relative error")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Per-area-bin curves.
    ax = axes[1]
    ax.plot(area_centres, bin_baseline, "o-", color="#c0392b",
            label="baseline (ignores hole)", lw=2)
    ax.plot(area_centres, bin_perf, "s-", color="#27ae60",
            label="fine-tuned encoder", lw=2)
    ax.axhline(0.05, color="black", linestyle="--", lw=1.0,
               label="5 % threshold")
    ax.set_xlabel("Hole-area fraction (pi r^2 / (a b))")
    ax.set_ylabel("Mean relative error over K modes")
    ax.set_title("E3h: error vs hole-area fraction")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Scatter of mean-over-modes error vs area fraction (per test sample).
    ax = axes[2]
    mean_rel_baseline = rel_baseline.mean(axis=1)
    mean_rel_perf = rel_perf.mean(axis=1)
    ax.scatter(test_areas, mean_rel_baseline, s=14, alpha=0.55,
               color="#c0392b", label="baseline")
    ax.scatter(test_areas, mean_rel_perf, s=14, alpha=0.55,
               color="#27ae60", label="fine-tuned")
    ax.axhline(0.05, color="black", linestyle="--", lw=1.0)
    ax.set_xlabel("Hole-area fraction")
    ax.set_ylabel("Mean rel-err over K modes (per sample)")
    ax.set_title("E3h: per-sample scatter")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Experiment E3h: perforated rectangles (Pillar 2)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "e3h_perforated_rectangles")

    # -----------------------------------------------------------------
    # 6. Headline metrics + JSON dump.
    # -----------------------------------------------------------------
    timing = timer.dump()
    headline = {
        "mean_rel_err_baseline": float(rel_baseline.mean()),
        "mean_rel_err_perf": float(rel_perf.mean()),
        "median_rel_err_baseline": float(np.median(rel_baseline)),
        "median_rel_err_perf": float(np.median(rel_perf)),
        "fraction_modes_under_5pct_baseline":
            float((err_baseline_per_mode < 0.05).mean()),
        "fraction_modes_under_5pct_perf":
            float((err_perf_per_mode < 0.05).mean()),
        "n_test_kept": int(len(test_eigs)),
    }

    extra = {
        "err_per_mode_baseline": err_baseline_per_mode.tolist(),
        "err_per_mode_perf": err_perf_per_mode.tolist(),
        "area_bin_centres": area_centres.tolist(),
        "area_bin_baseline": bin_baseline.tolist(),
        "area_bin_perf": bin_perf.tolist(),
        "area_bin_counts": bin_counts.tolist(),
    }

    write_run_json(
        out_dir,
        experiment="e3h",
        pillar="Pillar 2 (geometry-adaptive spectral basis)",
        hypothesis=(
            "A circular hole in a rectangle shifts the Dirichlet spectrum "
            "predictably (Rayleigh-quotient perturbation proportional to "
            "the hole-area fraction). A small dedicated encoder fine-tuned "
            "on the 5-vector descriptor (a, b, cx, cy, r) should recover "
            "the perturbed eigenvalues, while a plain-rectangle baseline "
            "(which sees only (a, b)) is expected to degrade as the hole "
            "area grows."
        ),
        parameters=vars(args),
        headline=headline,
        timing=timing,
        device=device,
        extra=extra,
    )

    print_summary("E3h", headline, timing)


if __name__ == "__main__":
    main()
