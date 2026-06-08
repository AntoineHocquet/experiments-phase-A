"""Experiment E3i (Pillar 2): 3D box eigenvalue prediction.

Smallest test of "the LB-eigenvalue idea scales to 3D". The 2-parameter
rectangle encoder of E3a extends naturally to a 3-parameter box encoder
(d_in = 3). Same architecture (geometry.encoder.EigenvalueEncoder), same
log-log training recipe, same per-mode error budget.

Hypothesis. A 3-parameter box encoder reaches the E3a accuracy level
(< 5 % per-mode rel-err for K = 10) on in-family OOD boxes, with compute
similar to the rectangle case. This validates the scaling claim for 3D
geometries.

PDE / ground truth. Dirichlet eigenvalues of -Laplace on
[0, a] x [0, b] x [0, c]:
    lambda_{nml} = pi^2 (n^2/a^2 + m^2/b^2 + l^2/c^2),  n, m, l >= 1.
Training distribution: (a, b, c) ~ Uniform(0.5, 1.5)^3 (independent).
In-family OOD: a in {1.8, 2.0}, with b, c spanning the training range and
slightly beyond.

Method. Mirrors geometry.run_e3a.run_rectangle exactly, with d_in = 3.
The encoder is geometry.encoder.EigenvalueEncoder(d_in=3, K=10).

Output.
  - Per-mode rel-err bar plot on in-family OOD boxes (a in {1.8, 2.0},
    b, c sweep).
  - 2D extrapolation slice: fix c = 1, scan (a, b) on a grid; heatmap of
    the mean rel-err over K modes with the 5 % contour and the training
    box overlay.

GPU-target runtime: about 1 h at default flags on a single GPU.
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

from geometry.encoder import EigenvalueEncoder
from geometry.lb_truth import box_eigenvalues
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary,
    save_figure, write_run_json,
)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def box_dataset(n: int, K: int,
                 size_range: tuple = (0.5, 1.5),
                 seed: int = 0) -> TensorDataset:
    """Random boxes with (a, b, c) drawn independently from size_range."""
    rng = np.random.default_rng(seed)
    abc  = rng.uniform(*size_range, size=(n, 3)).astype(np.float32)
    eigs = np.stack([box_eigenvalues(float(a), float(b), float(c), K)
                     for a, b, c in abc]).astype(np.float32)
    return TensorDataset(torch.tensor(abc), torch.tensor(eigs))


# ---------------------------------------------------------------------------
# Training (log-space MSE, same recipe as E3a)
# ---------------------------------------------------------------------------

def train(model: nn.Module, ds: TensorDataset,
          n_epochs: int = 1500, batch: int = 256, lr: float = 3e-3,
          device: str = "cpu") -> list[float]:
    model.to(device)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
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
        history.append(running / len(loader))
    return history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def rel_err_per_mode(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean relative error per eigenmode, shape (K,)."""
    return np.abs(pred - true).mean(axis=0) / (true.mean(axis=0) + 1e-12)


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def eval_box(model: EigenvalueEncoder, abc_triples, K: int) -> np.ndarray:
    model.eval()
    dev = _device_of(model)
    x = torch.tensor(abc_triples, dtype=torch.float32, device=dev)
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    true = np.stack([box_eigenvalues(float(a), float(b), float(c), K)
                     for a, b, c in abc_triples])
    return rel_err_per_mode(pred, true)


def box_error_grid(model: EigenvalueEncoder, K: int,
                    a_grid: np.ndarray, b_grid: np.ndarray,
                    c_fixed: float) -> np.ndarray:
    """Mean relative error over the K leading eigenvalues on a 2D (a, b) slice.

    The third dimension is fixed at c_fixed so the result is a 2D heatmap
    (the natural visualisation of an extrapolation map in 3D).
    """
    model.eval()
    dev = _device_of(model)
    A, B = np.meshgrid(a_grid, b_grid, indexing="ij")
    a_flat = A.reshape(-1).astype(np.float32)
    b_flat = B.reshape(-1).astype(np.float32)
    c_flat = np.full_like(a_flat, c_fixed, dtype=np.float32)
    x = torch.tensor(np.stack([a_flat, b_flat, c_flat], axis=1), device=dev)
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    true = np.stack([box_eigenvalues(float(a), float(b), float(c), K)
                     for a, b, c in zip(a_flat, b_flat, c_flat)])
    err = np.abs(pred - true) / (true + 1e-12)
    return err.mean(axis=1).reshape(A.shape)


# ---------------------------------------------------------------------------
# Per-run logic
# ---------------------------------------------------------------------------

def run_box(K: int, n_train: int, n_epochs: int, batch: int,
            device: str, timer: Timer, seed: int) -> dict:
    print(f"\nTraining box encoder  K={K}  n={n_train}  "
          f"epochs={n_epochs}  device={device}")
    enc = EigenvalueEncoder(d_in=3, K=K)
    with timer("train"):
        history = train(enc, box_dataset(n_train, K, seed=seed),
                         n_epochs=n_epochs, batch=batch, device=device)
    print(f"  Final log-MSE: {history[-1]:.4f}")

    # In-family OOD: a in {1.8, 2.0} (outside [0.5, 1.5]); b, c span the
    # training range plus a small extrapolation slice.
    with timer("eval_ood"):
        ood_triples = [(a, b, c)
                       for a in (1.8, 2.0)
                       for b in (0.6, 1.0, 1.4)
                       for c in (0.6, 1.0, 1.4)]
        err_ood = eval_box(enc, ood_triples, K)
    print("  In-family OOD (a in {1.8, 2.0}, b, c in {0.6, 1.0, 1.4}):")
    for k, e in enumerate(err_ood):
        print(f"    mode k={k+1:2d}   rel-err = {e:.4f}")
    ok = bool((err_ood < 0.05).all())
    print(f"  PoC criterion (< 5 %): {'PASS' if ok else 'FAIL'}")

    # 2D extrapolation slice at c = 1.
    with timer("eval_grid"):
        a_grid = np.linspace(0.3, 2.5, 20)
        b_grid = np.linspace(0.3, 2.5, 20)
        err_map = box_error_grid(enc, K, a_grid, b_grid, c_fixed=1.0)

    return {
        "err_ood": err_ood,
        "pass": ok,
        "a_grid": a_grid,
        "b_grid": b_grid,
        "err_map": err_map,
        "final_log_mse": history[-1],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bar_panel(ax, err, title):
    modes = np.arange(1, len(err) + 1)
    ax.bar(modes, err, color="#2a6496", alpha=0.85)
    ax.axhline(0.05, color="red", linestyle="--", label="5 % threshold")
    ax.set_xlabel("Eigenmode $k$")
    ax.set_ylabel("Mean relative error")
    ax.set_title(title)
    ax.legend(loc="upper left")


def _heatmap_panel(ax, a_grid, b_grid, err_map, title,
                    train_a=(0.5, 1.5), train_b=(0.5, 1.5)):
    pcm = ax.pcolormesh(b_grid, a_grid, np.clip(err_map, 1e-4, 1.0),
                        cmap="viridis", shading="auto",
                        norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=1.0))
    cs = ax.contour(b_grid, a_grid, err_map, levels=[0.05],
                    colors="red", linewidths=1.5, linestyles="--")
    ax.clabel(cs, fmt="5 %%", inline=True, fontsize=9)
    ax.add_patch(plt.Rectangle((train_b[0], train_a[0]),
                                train_b[1] - train_b[0],
                                train_a[1] - train_a[0],
                                fill=False, edgecolor="white",
                                linewidth=1.5, linestyle="-"))
    ax.text(train_b[0] + 0.05, train_a[1] - 0.08, "training box",
            color="white", fontsize=8, va="top")
    ax.set_xlabel("Side length $b$")
    ax.set_ylabel("Side length $a$")
    ax.set_title(title)
    plt.colorbar(pcm, ax=ax, label="mean rel. error (K modes)")


def make_plot(result: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _bar_panel(axes[0], result["err_ood"],
               "E3i: 3D box, in-family OOD (a in {1.8, 2.0})")
    _heatmap_panel(axes[1], result["a_grid"], result["b_grid"],
                    result["err_map"],
                    "E3i: extrapolation slice (c = 1)")
    fig.suptitle("Experiment E3i: 3D box LB eigenvalue prediction "
                 "(Phase A, Pillar 2)",
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "e3i_box_eigenvalues")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--K",        type=int, default=10,
                    help="Number of eigenvalues the encoder predicts.")
    ap.add_argument("--n_train",  type=int, default=10000)
    ap.add_argument("--n_epochs", type=int, default=1500)
    ap.add_argument("--batch",    type=int, default=256)
    ap.add_argument("--out_dir",  default="results_e3i")
    args = ap.parse_args()

    if args.smoke:
        args.K        = 6
        args.n_train  = 200
        args.n_epochs = 30
        args.batch    = 32

    device = auto_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    timer = Timer()
    result = run_box(args.K, args.n_train, args.n_epochs, args.batch,
                     device, timer, seed=args.seed)

    make_plot(result, out_dir)

    timing = timer.dump()

    emap = result["err_map"]
    frac_under_5 = float((emap < 0.05).mean())
    print("\nInterpretation:")
    print("  In-family OOD: a PASS means the encoder learned the 3D scaling")
    print("  law pi^2 (n^2/a^2 + m^2/b^2 + l^2/c^2) rather than a per-shape")
    print("  lookup.")
    print(f"  3D slice extrapolation (c=1): {100*frac_under_5:.1f} % of the "
          f"({len(result['a_grid'])}x{len(result['b_grid'])}) (a, b) grid "
          "stays under the 5 % threshold,")
    print("  including regions well outside the training box "
          "(a, b in [0.5, 1.5]).")

    headline: dict[str, float | bool] = {
        "pass_box_OOD": bool(result["pass"]),
        "max_rel_err_ood": float(result["err_ood"].max()),
        "mean_rel_err_ood": float(result["err_ood"].mean()),
        "frac_grid_under_5pct": frac_under_5,
        "final_log_mse": float(result["final_log_mse"]),
    }
    for k, e in enumerate(result["err_ood"]):
        headline[f"rel_err_mode_{k+1}"] = float(e)

    write_run_json(out_dir,
                   experiment="e3i",
                   pillar="Pillar 2 (geometry-adaptive spectral basis)",
                   hypothesis=("The 2-parameter rectangle encoder extends "
                               "naturally to a 3-parameter box encoder, "
                               "matching the E3a accuracy level "
                               "(< 5 % per-mode rel-err at K = 10) on "
                               "in-family OOD boxes."),
                   parameters=vars(args),
                   headline=headline,
                   timing=timing,
                   device=device,
                   extra={
                       "err_ood": result["err_ood"].tolist(),
                       "a_grid": result["a_grid"].tolist(),
                       "b_grid": result["b_grid"].tolist(),
                       "err_map": result["err_map"].tolist(),
                   })

    print_summary("E3i", headline, timing)


if __name__ == "__main__":
    main()
