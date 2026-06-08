"""Experiment E3d (Pillar 2): L-shape eigenvalues with a re-entrant corner.

Hypothesis. A rectangle-pretrained encoder (E3a-style) partially recovers the
L-shape spectrum via Weyl asymptotics (which only see volume + perimeter), but
the re-entrant corner produces a r^{2/3} singularity that the lowest
eigenmodes encode and that no convex-shape encoder has seen. Fine-tuning on a
small L-shape library that exposes the cut dimensions ``(cx, cy)`` recovers
the missing structure.

Pipeline.
  1. Train a rectangle encoder from scratch on synthetic ``[0, a] x [0, b]``
     samples with analytic eigenvalues (this mirrors the E3a rectangle run,
     so the experiment is self-contained at runtime).
  2. Evaluate that encoder on an L-shape test library by feeding the
     bounding-rectangle descriptor ``(a, b)``; record per-mode relative
     error vs the FEM-on-grid reference (the "rectangle-only" baseline).
  3. Fine-tune a fresh 4-input encoder on a small L-shape training library
     ``(a, b, cx, cy)``, with FEM reference eigenvalues. Evaluate on the
     same test library; record per-mode error.
  4. Plot per-mode relative error for both encoders (the rectangle-only
     baseline and the L-aware fine-tuned encoder), plus a heatmap of the
     first eigenmode (rectangle prediction restricted to the L vs the FEM
     reference) to visualise the corner singularity.

Output. A two-panel figure (bar chart + eigenmode heatmap), plus
``e3d_raw.json`` with per-mode errors and timing.

GPU defaults:
``--K 10 --n_lshape_train 200 --n_lshape_test 40 --n_finetune_epochs 200``.
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
from geometry.lshape_modes import (
    bounding_rectangle_eigenfunction, lshape_eigenpairs, lshape_eigenvalues,
)
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary, save_figure,
    write_run_json,
)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def rectangle_dataset(n: int, K: int,
                       size_range: tuple = (0.5, 1.5),
                       aspect_range: tuple = (0.5, 3.0),
                       seed: int = 1) -> TensorDataset:
    """Synthetic rectangle library with analytic Dirichlet eigenvalues.

    Mirrors ``geometry/run_e3a.py``'s rectangle dataset; reproduced here so the
    E3d script is self-contained and the helper used in fine-tuning is
    co-located with the L-shape loaders.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(*size_range, size=n).astype(np.float32)
    rho = rng.uniform(*aspect_range, size=n).astype(np.float32)
    b = (a / rho).astype(np.float32)
    eigs = np.stack([rectangle_eigenvalues(ai, bi, K)
                     for ai, bi in zip(a, b)]).astype(np.float32)
    return TensorDataset(
        torch.tensor(np.stack([a, b], axis=1)),
        torch.tensor(eigs),
    )


def lshape_dataset(n: int, K: int, nx: int, ny: int,
                    size_range: tuple = (0.7, 1.5),
                    cut_frac_range: tuple = (0.2, 0.6),
                    seed: int = 7) -> tuple[TensorDataset, np.ndarray]:
    """Random L-shape library with FEM-on-grid reference eigenvalues.

    The descriptor is ``(a, b, cx, cy)``. The cut size in each axis is drawn
    as a fraction of the bounding-rectangle side in ``cut_frac_range``, which
    keeps the L-shape clearly non-convex without collapsing it.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(*size_range, size=n).astype(np.float32)
    rho = rng.uniform(0.7, 1.4, size=n).astype(np.float32)
    b = (a / rho).astype(np.float32)
    fx = rng.uniform(*cut_frac_range, size=n).astype(np.float32)
    fy = rng.uniform(*cut_frac_range, size=n).astype(np.float32)
    cx = (a * fx).astype(np.float32)
    cy = (b * fy).astype(np.float32)
    descriptors = np.stack([a, b, cx, cy], axis=1)
    eigs = np.zeros((n, K), dtype=np.float32)
    for i in range(n):
        eigs[i] = lshape_eigenvalues(float(a[i]), float(b[i]),
                                      float(cx[i]), float(cy[i]),
                                      K, nx=nx, ny=ny).astype(np.float32)
    ds = TensorDataset(torch.tensor(descriptors), torch.tensor(eigs))
    return ds, descriptors


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: torch.nn.Module, ds: TensorDataset,
          n_epochs: int, batch: int, lr: float, device: str) -> list:
    """Log-space MSE training, same recipe as E3a."""
    model.to(device)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(n_epochs, 1))
    history = []
    for _ in range(n_epochs):
        running = 0.0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(torch.log(pred), torch.log(y))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        sched.step()
        history.append(running / max(len(loader), 1))
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rel_err_per_mode(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean relative error per eigenmode over the test set, shape (K,)."""
    return np.abs(pred - true).mean(axis=0) / (true.mean(axis=0) + 1e-12)


def encoder_predict(model: EigenvalueEncoder, descriptors: np.ndarray,
                     device: str) -> np.ndarray:
    model.eval()
    x = torch.tensor(descriptors, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(x).cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bar_panel(ax, err_rect, err_ft, K: int):
    modes = np.arange(1, K + 1)
    width = 0.4
    ax.bar(modes - width / 2, err_rect, width,
           color="#b95c50", alpha=0.9, label="rectangle encoder (no cut info)")
    ax.bar(modes + width / 2, err_ft, width,
           color="#2a6496", alpha=0.9, label="L-aware fine-tuned encoder")
    ax.axhline(0.05, color="red", linestyle="--", lw=1.0, label="5 % threshold")
    ax.set_xlabel("Eigenmode k")
    ax.set_ylabel("Mean relative error (test set)")
    ax.set_title("E3d: per-mode error on L-shape test library")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)


def _mode_heatmap_panel(ax_pred, ax_true, a: float, b: float,
                         cx: float, cy: float, nx: int, ny: int):
    """First-eigenmode comparison: rectangle prediction (bounding-rectangle
    sine restricted to the L-shape) vs the FEM reference. The reentrant corner
    is highlighted by a marker.
    """
    _, phis = lshape_eigenpairs(a, b, cx, cy, K=1, nx=nx, ny=ny,
                                  return_vectors=True)
    phi_true = phis[0]
    sign = 1.0 if phi_true.sum() >= 0 else -1.0
    phi_true = sign * phi_true

    from geometry.lshape_modes import lshape_mask
    mask = lshape_mask(nx, ny, a, b, cx, cy)
    phi_rect = bounding_rectangle_eigenfunction(a, b, nx, ny, mask=mask)
    if phi_rect.sum() < 0:
        phi_rect = -phi_rect

    vmax = max(np.abs(phi_true).max(), np.abs(phi_rect).max())

    def _draw(ax, field, title):
        masked = np.where(mask, field, np.nan)
        im = ax.imshow(masked.T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=(0.0, a, 0.0, b), aspect="equal")
        ax.plot([a - cx], [b - cy], marker="o", color="black",
                markersize=6, label="reentrant corner")
        ax.plot([a - cx, a - cx, a], [b - cy, b, b - cy],
                marker="", linestyle="", color="black")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="lower left", fontsize=8)
        return im

    _draw(ax_pred, phi_rect, "Rectangle prediction (mode 1, restricted)")
    im = _draw(ax_true, phi_true, "FEM reference (mode 1, L-shape)")
    plt.colorbar(im, ax=ax_true, fraction=0.045, pad=0.04)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--K", type=int, default=10,
                    help="Number of eigenvalues the encoder predicts.")
    ap.add_argument("--n_rect_train", type=int, default=5000,
                    help="Rectangle pretraining library size.")
    ap.add_argument("--n_rect_epochs", type=int, default=1200,
                    help="Rectangle pretraining epoch count.")
    ap.add_argument("--n_lshape_train", type=int, default=200,
                    help="L-shape fine-tune library size.")
    ap.add_argument("--n_lshape_test", type=int, default=40,
                    help="L-shape held-out test library size.")
    ap.add_argument("--n_finetune_epochs", type=int, default=200,
                    help="Fine-tune epoch count for the L-aware encoder.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr_rect", type=float, default=3e-3)
    ap.add_argument("--lr_finetune", type=float, default=2e-3)
    ap.add_argument("--nx_fem", type=int, default=64,
                    help="Grid resolution for FEM-on-grid eigensolves.")
    ap.add_argument("--ny_fem", type=int, default=64)
    ap.add_argument("--vis_a", type=float, default=1.0,
                    help="Bounding-rectangle width for the eigenmode panel.")
    ap.add_argument("--vis_b", type=float, default=1.0,
                    help="Bounding-rectangle height for the eigenmode panel.")
    ap.add_argument("--vis_cx", type=float, default=0.5,
                    help="Cut width for the eigenmode panel.")
    ap.add_argument("--vis_cy", type=float, default=0.5,
                    help="Cut height for the eigenmode panel.")
    ap.add_argument("--out_dir", default="results_e3d")
    args = ap.parse_args()

    if args.smoke:
        args.K = 6
        args.n_rect_train = 200
        args.n_rect_epochs = 30
        args.n_lshape_train = 12
        args.n_lshape_test = 6
        args.n_finetune_epochs = 30
        args.batch = 32
        args.nx_fem = 24
        args.ny_fem = 24

    device = auto_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    timer = Timer()

    # ----- 1. Rectangle pretraining -----
    with timer("rect_pretrain"):
        rect_ds = rectangle_dataset(args.n_rect_train, args.K, seed=args.seed + 1)
        rect_encoder = EigenvalueEncoder(d_in=2, K=args.K)
        rect_hist = train(rect_encoder, rect_ds,
                           n_epochs=args.n_rect_epochs, batch=args.batch,
                           lr=args.lr_rect, device=device)
    print(f"Rectangle encoder final log-MSE: {rect_hist[-1]:.4f}")

    # ----- 2. Build L-shape libraries (FEM ground truth) -----
    with timer("lshape_fem"):
        train_ds, train_desc = lshape_dataset(
            args.n_lshape_train, args.K, args.nx_fem, args.ny_fem,
            seed=args.seed + 11)
        test_ds, test_desc = lshape_dataset(
            args.n_lshape_test, args.K, args.nx_fem, args.ny_fem,
            seed=args.seed + 23)
    test_true = test_ds.tensors[1].numpy()

    # ----- 3. Rectangle-only baseline on L-shape test set -----
    with timer("rect_eval"):
        test_bounding = test_desc[:, :2].astype(np.float32)   # (n, 2) = (a, b)
        pred_rect = encoder_predict(rect_encoder, test_bounding, device)
        err_rect = rel_err_per_mode(pred_rect, test_true)

    # ----- 4. Fine-tune a 4-input encoder on the L-shape library -----
    with timer("lshape_finetune"):
        ft_encoder = EigenvalueEncoder(d_in=4, K=args.K)
        ft_hist = train(ft_encoder, train_ds,
                         n_epochs=args.n_finetune_epochs, batch=args.batch,
                         lr=args.lr_finetune, device=device)
    print(f"L-aware encoder final log-MSE: {ft_hist[-1]:.4f}")

    with timer("lshape_eval"):
        pred_ft = encoder_predict(ft_encoder, test_desc.astype(np.float32),
                                    device)
        err_ft = rel_err_per_mode(pred_ft, test_true)

    print("\nPer-mode relative error on L-shape test library:")
    print("  k   rectangle-only   L-aware fine-tuned")
    for k in range(args.K):
        print(f"  {k+1:2d}  {err_rect[k]:>14.4f}   {err_ft[k]:>14.4f}")

    # ----- 5. Plot -----
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0], wspace=0.32)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_true = fig.add_subplot(gs[0, 2])
    _bar_panel(ax_bar, err_rect, err_ft, args.K)
    _mode_heatmap_panel(ax_pred, ax_true,
                         args.vis_a, args.vis_b, args.vis_cx, args.vis_cy,
                         args.nx_fem, args.ny_fem)
    fig.suptitle("E3d: L-shape eigenvalues; reentrant corner exposes the "
                  "rectangle encoder's blind spot", fontsize=12)
    save_figure(fig, out_dir, "e3d_lshape_eigenvalues")

    # ----- 6. Headline + JSON -----
    headline = {
        "rect_baseline_mean_relerr": float(err_rect.mean()),
        "lshape_finetune_mean_relerr": float(err_ft.mean()),
        "rect_baseline_modes_under_5pct":
            int((err_rect < 0.05).sum()),
        "lshape_finetune_modes_under_5pct":
            int((err_ft < 0.05).sum()),
        "rect_baseline_first_mode_relerr": float(err_rect[0]),
        "lshape_finetune_first_mode_relerr": float(err_ft[0]),
    }
    headline["pass_six_of_ten_under_5pct"] = bool(
        (err_ft < 0.05).sum() >= 6
    )

    timing = timer.dump()
    write_run_json(
        out_dir,
        experiment="e3d",
        pillar="Pillar 2 (geometry-adaptive spectral basis)",
        hypothesis=(
            "A rectangle-pretrained encoder only partially recovers the "
            "L-shape spectrum (Weyl asymptotics carry it part of the way), "
            "but the reentrant-corner r^(2/3) singularity in the lowest "
            "modes drives per-mode errors well above 5 % unless the encoder "
            "is exposed to the cut dimensions (cx, cy) directly."
        ),
        parameters=vars(args),
        headline=headline,
        timing=timing,
        device=device,
        extra={
            "err_rect_per_mode": err_rect.tolist(),
            "err_ft_per_mode": err_ft.tolist(),
            "rect_history_last": rect_hist[-1],
            "ft_history_last": ft_hist[-1],
        },
    )

    print_summary("E3d", headline, timing)


if __name__ == "__main__":
    main()
