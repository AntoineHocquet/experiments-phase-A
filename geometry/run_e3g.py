"""Experiment E3g (Pillar 2): triangle eigenvalues from a rectangle-pretrained
encoder, plus a small fine-tuning phase on a triangle library.

Hypothesis. Triangles are a non-product geometry but they admit closed-form
Dirichlet spectra for equilateral and right-isosceles cases, and a small set
of *almost* closed-form spectra parameterised by the three angles. A
rectangle-pretrained encoder should transfer to right triangles
(45-45-90) better than to skew triangles, with error correlated with how far
the triangle is from being a half-rectangle (i.e. how far apex angle alpha is
from pi / 4).

Method. We parametrise right triangles by ``(L, alpha)`` with ``L`` the base
and ``alpha`` the apex angle. Ground truth is obtained from a masked-grid
Dirichlet Laplacian eigensolve (see ``geometry/triangle_modes.py``), an
in-repo FEM stand-in that requires no new dependencies. The encoder input is
the "fictitious bounding rectangle" descriptor ``(L, L * tan(alpha))``, so a
rectangle-pretrained ``EigenvalueEncoder(d_in=2)`` can be evaluated as-is and
its error reported as a function of how skew the triangle is.

We then add a short fine-tuning phase on a small triangle library and report
the post-finetune error map, to show how cheaply the encoder closes the gap.

Output. Per-mode rel-err panel plus a heat-map of mean relative error over a
``(L, alpha)`` grid, for both the rectangle-only encoder and the fine-tuned
encoder. The line ``alpha = pi / 4`` (half-rectangle) is highlighted on the
heat-maps.
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
from geometry.run_e3a import rectangle_dataset, train as train_encoder
from geometry.triangle_modes import triangle_eigenvalues
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary,
    save_figure, write_run_json,
)


# ---------------------------------------------------------------------------
# Triangle dataset (FEM ground truth)
# ---------------------------------------------------------------------------

def triangle_dataset(n: int, K: int,
                      L_range: tuple = (0.6, 1.4),
                      alpha_range: tuple = (np.pi / 8, 3 * np.pi / 8),
                      nx: int = 40, ny: int = 40,
                      seed: int = 0) -> TensorDataset:
    """Random right triangles parametrised by (L, alpha).

    Descriptor is ``(L, L * tan(alpha))``, the bounding rectangle of the
    triangle. Eigenvalues are computed by a masked-grid Laplacian eigensolve.
    """
    rng = np.random.default_rng(seed)
    Ls    = rng.uniform(*L_range, size=n).astype(np.float32)
    alpha = rng.uniform(*alpha_range, size=n).astype(np.float32)
    Hs    = (Ls * np.tan(alpha)).astype(np.float32)

    eigs = np.stack([
        triangle_eigenvalues(float(Li), float(Hi), K, nx=nx, ny=ny)
        for Li, Hi in zip(Ls, Hs)
    ]).astype(np.float32)

    desc = np.stack([Ls, Hs], axis=1)            # (n, 2): fictitious bounding rect
    return TensorDataset(torch.tensor(desc), torch.tensor(eigs))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def rel_err_per_mode(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true).mean(axis=0) / (true.mean(axis=0) + 1e-12)


def eval_triangle(model: EigenvalueEncoder, descs: np.ndarray,
                   eigs_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-sample mean rel-err over K modes, per-mode rel-err over the batch).

    ``descs`` has shape (M, 2) (the bounding rectangle (L, H));
    ``eigs_true`` has shape (M, K).
    """
    model.eval()
    dev = _device_of(model)
    x = torch.tensor(descs, dtype=torch.float32, device=dev)
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    err_each = np.abs(pred - eigs_true) / (eigs_true + 1e-12)   # (M, K)
    return err_each.mean(axis=1), rel_err_per_mode(pred, eigs_true)


def triangle_error_grid(model: EigenvalueEncoder, K: int,
                         L_grid: np.ndarray, alpha_grid: np.ndarray,
                         nx: int, ny: int) -> np.ndarray:
    """Mean relative error over K leading eigenvalues on a (L, alpha) grid."""
    A, B = np.meshgrid(L_grid, alpha_grid, indexing="ij")
    L_flat = A.reshape(-1).astype(np.float32)
    alpha_flat = B.reshape(-1).astype(np.float32)
    H_flat = (L_flat * np.tan(alpha_flat)).astype(np.float32)

    descs = np.stack([L_flat, H_flat], axis=1)
    true = np.stack([
        triangle_eigenvalues(float(Li), float(Hi), K, nx=nx, ny=ny)
        for Li, Hi in zip(L_flat, H_flat)
    ]).astype(np.float32)

    dev = _device_of(model)
    x = torch.tensor(descs, dtype=torch.float32, device=dev)
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    err = np.abs(pred - true) / (true + 1e-12)
    return err.mean(axis=1).reshape(A.shape)


# ---------------------------------------------------------------------------
# Fine-tuning on a triangle library
# ---------------------------------------------------------------------------

def finetune_on_triangles(model: EigenvalueEncoder, ds: TensorDataset,
                           n_epochs: int, batch: int, lr: float,
                           device: str) -> list[float]:
    model.to(device).train()
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(n_epochs, 1))
    history: list[float] = []
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
# Plotting
# ---------------------------------------------------------------------------

def _bar_panel(ax, err, title: str):
    modes = np.arange(1, len(err) + 1)
    ax.bar(modes, err, color="#2a6496", alpha=0.85)
    ax.axhline(0.05, color="red", linestyle="--", label="5 % threshold")
    ax.set_xlabel("Eigenmode k")
    ax.set_ylabel("Mean relative error")
    ax.set_title(title)
    ax.legend(loc="upper left")


def _heatmap_panel(ax, L_grid, alpha_grid, err_map, title: str):
    pcm = ax.pcolormesh(np.degrees(alpha_grid), L_grid,
                        np.clip(err_map, 1e-4, 1.0),
                        cmap="viridis", shading="auto",
                        norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=1.0))
    cs = ax.contour(np.degrees(alpha_grid), L_grid, err_map,
                    levels=[0.05], colors="red",
                    linewidths=1.5, linestyles="--")
    if len(cs.allsegs[0]) > 0:
        ax.clabel(cs, fmt="5 %%", inline=True, fontsize=9)
    ax.axvline(45.0, color="white", linewidth=1.2, linestyle=":",
               label="half-rectangle (alpha = 45 deg)")
    ax.set_xlabel("Apex angle alpha (deg)")
    ax.set_ylabel("Base length L")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    plt.colorbar(pcm, ax=ax, label="mean rel. error (K modes)")


def make_plot(K: int, err_per_mode_rect, err_per_mode_ft,
              L_grid, alpha_grid, err_map_rect, err_map_ft, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _bar_panel(axes[0, 0], err_per_mode_rect,
                "Per-mode rel-err (rectangle-pretrained)")
    _bar_panel(axes[0, 1], err_per_mode_ft,
                "Per-mode rel-err (after triangle fine-tune)")
    _heatmap_panel(axes[1, 0], L_grid, alpha_grid, err_map_rect,
                    "Rectangle-pretrained: error over (L, alpha)")
    _heatmap_panel(axes[1, 1], L_grid, alpha_grid, err_map_ft,
                    "Fine-tuned: error over (L, alpha)")
    fig.suptitle(
        f"Experiment E3g: triangle eigenvalues, K = {K} modes (Phase A, Pillar 2)",
        fontsize=12,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, "e3g_triangle_eigenvalues")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--K", type=int, default=10,
                    help="Number of eigenvalues the encoder predicts.")
    # Rectangle pretraining (mirrors E3a defaults)
    ap.add_argument("--n_rect_train", type=int, default=5000)
    ap.add_argument("--n_rect_epochs", type=int, default=600)
    ap.add_argument("--rect_batch", type=int, default=128)
    # Triangle library
    ap.add_argument("--n_triangle_train", type=int, default=400,
                    help="Triangles in the fine-tuning library.")
    ap.add_argument("--n_triangle_test", type=int, default=200,
                    help="Held-out triangles for evaluation.")
    ap.add_argument("--n_finetune_epochs", type=int, default=200)
    ap.add_argument("--finetune_batch", type=int, default=64)
    ap.add_argument("--finetune_lr", type=float, default=1e-3)
    # Grid for the FEM eigensolve
    ap.add_argument("--fem_nx", type=int, default=40)
    ap.add_argument("--fem_ny", type=int, default=40)
    # Heat-map resolution
    ap.add_argument("--map_nL", type=int, default=12)
    ap.add_argument("--map_nalpha", type=int, default=12)
    ap.add_argument("--out_dir", default="results_e3g")
    args = ap.parse_args()

    if args.smoke:
        args.K = 6
        args.n_rect_train = 300
        args.n_rect_epochs = 30
        args.rect_batch = 64
        args.n_triangle_train = 30
        args.n_triangle_test = 20
        args.n_finetune_epochs = 20
        args.finetune_batch = 16
        args.fem_nx = 18
        args.fem_ny = 18
        args.map_nL = 5
        args.map_nalpha = 5

    device = auto_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    timer = Timer()
    K = args.K

    # 1) Pretrain a rectangle encoder (mirrors E3a's run_rectangle).
    with timer("pretrain_rectangle"):
        print(f"\nPretraining rectangle encoder  K={K}  n={args.n_rect_train}"
              f"  epochs={args.n_rect_epochs}")
        enc = EigenvalueEncoder(d_in=2, K=K)
        ds_rect = rectangle_dataset(args.n_rect_train, K)
        train_encoder(enc, ds_rect,
                       n_epochs=args.n_rect_epochs,
                       batch=args.rect_batch, device=device)

    # 2) Build a triangle test set (always the same seed for fair compare).
    with timer("triangle_test_dataset"):
        ds_tri_test = triangle_dataset(
            args.n_triangle_test, K,
            nx=args.fem_nx, ny=args.fem_ny, seed=10101,
        )
    test_descs = ds_tri_test.tensors[0].numpy()
    test_eigs  = ds_tri_test.tensors[1].numpy()
    test_alpha = np.arctan2(test_descs[:, 1], test_descs[:, 0])

    # 3) Evaluate rectangle-pretrained encoder on the triangle test set.
    with timer("eval_rect_pretrained"):
        sample_err_rect, per_mode_rect = eval_triangle(enc, test_descs, test_eigs)
    print(f"\nRectangle-pretrained encoder on triangles:")
    print(f"  mean rel-err over K modes : {sample_err_rect.mean():.4f}")
    print(f"  median rel-err            : {float(np.median(sample_err_rect)):.4f}")
    for k, e in enumerate(per_mode_rect):
        print(f"    mode k={k+1:2d}  rel-err = {e:.4f}")

    # Skew dependency: split alpha into "near 45 deg" vs "far from 45 deg".
    half_rect_mask = np.abs(test_alpha - np.pi / 4) < np.deg2rad(10.0)
    err_near = (sample_err_rect[half_rect_mask].mean()
                 if half_rect_mask.any() else float("nan"))
    err_far  = (sample_err_rect[~half_rect_mask].mean()
                 if (~half_rect_mask).any() else float("nan"))
    print(f"  rel-err on near-45-deg triangles : {err_near:.4f}")
    print(f"  rel-err on skew triangles        : {err_far:.4f}")

    # 4) Heat-map (rectangle-pretrained) on a (L, alpha) grid.
    with timer("heatmap_rect"):
        L_grid = np.linspace(0.5, 1.5, args.map_nL)
        alpha_grid = np.linspace(np.pi / 8, 3 * np.pi / 8, args.map_nalpha)
        err_map_rect = triangle_error_grid(
            enc, K, L_grid, alpha_grid, args.fem_nx, args.fem_ny,
        )

    # 5) Fine-tune on a small triangle library.
    with timer("finetune"):
        ds_tri_train = triangle_dataset(
            args.n_triangle_train, K,
            nx=args.fem_nx, ny=args.fem_ny, seed=20202,
        )
        ft_history = finetune_on_triangles(
            enc, ds_tri_train,
            n_epochs=args.n_finetune_epochs,
            batch=args.finetune_batch,
            lr=args.finetune_lr,
            device=device,
        )
    print(f"\nFine-tune final log-MSE : {ft_history[-1]:.4f}")

    # 6) Evaluate fine-tuned encoder on the held-out triangle test set.
    with timer("eval_finetuned"):
        sample_err_ft, per_mode_ft = eval_triangle(enc, test_descs, test_eigs)
    print(f"\nFine-tuned encoder on triangles:")
    print(f"  mean rel-err over K modes : {sample_err_ft.mean():.4f}")
    print(f"  median rel-err            : {float(np.median(sample_err_ft)):.4f}")
    for k, e in enumerate(per_mode_ft):
        print(f"    mode k={k+1:2d}  rel-err = {e:.4f}")

    # 7) Heat-map after fine-tuning.
    with timer("heatmap_finetuned"):
        err_map_ft = triangle_error_grid(
            enc, K, L_grid, alpha_grid, args.fem_nx, args.fem_ny,
        )

    # 8) Plot.
    with timer("plot"):
        make_plot(K, per_mode_rect, per_mode_ft,
                   L_grid, alpha_grid, err_map_rect, err_map_ft,
                   out_dir)

    timing = timer.dump()

    headline = {
        "mean_rel_err_rect_pretrained": float(sample_err_rect.mean()),
        "median_rel_err_rect_pretrained": float(np.median(sample_err_rect)),
        "mean_rel_err_near_45deg_rect": float(err_near),
        "mean_rel_err_skew_rect": float(err_far),
        "mean_rel_err_finetuned": float(sample_err_ft.mean()),
        "median_rel_err_finetuned": float(np.median(sample_err_ft)),
        "frac_grid_under_5pct_rect": float((err_map_rect < 0.05).mean()),
        "frac_grid_under_5pct_finetuned": float((err_map_ft < 0.05).mean()),
        "modes_under_5pct_rect": int((per_mode_rect < 0.05).sum()),
        "modes_under_5pct_finetuned": int((per_mode_ft < 0.05).sum()),
    }

    extra = {
        "per_mode_rel_err_rect": per_mode_rect.tolist(),
        "per_mode_rel_err_finetuned": per_mode_ft.tolist(),
        "L_grid": L_grid.tolist(),
        "alpha_grid_rad": alpha_grid.tolist(),
        "err_map_rect": err_map_rect.tolist(),
        "err_map_finetuned": err_map_ft.tolist(),
        "finetune_history": ft_history,
    }

    write_run_json(
        out_dir,
        experiment="e3g",
        pillar="Pillar 2 (geometry-adaptive spectral basis)",
        hypothesis=(
            "A rectangle-pretrained LB-eigenvalue encoder transfers to right "
            "triangles via the bounding-rectangle descriptor (L, L*tan(alpha)), "
            "with error correlated to how far alpha is from pi/4 "
            "(the half-rectangle case). A short fine-tune on a small triangle "
            "library should close the gap, especially in the skew regime."
        ),
        parameters=vars(args),
        headline=headline,
        timing=timing,
        device=device,
        extra=extra,
    )

    print_summary("E3g", headline, timing)


if __name__ == "__main__":
    main()
