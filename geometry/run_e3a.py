"""Experiment E3a: eigenvalue prediction across geometry families.

Tests whether a shallow MLP encoder learns the Laplace–Beltrami eigenvalue
scaling law well enough to generalise within each geometry family, including
substantial extrapolation beyond the training box on rectangles, where the
encoder must learn the joint two-parameter law λ_{mn} = (mπ/a)² + (nπ/b)².

Two families are trained (selectable with --family), matching the proposal
(architectures.tex, Section IV.3):
  - Intervals  [0, L],        L ~ Uniform[0.5, 2.0]              (encoder d_in=1)
  - Rectangles [0,a]×[0,b],   a ~ Uniform[0.5, 1.5],
                              aspect ratio a/b ~ Uniform[0.5, 3.0]  (d_in=2)
Both have analytically known Dirichlet eigenvalues (no FEM required).

Evaluation:
  - Interval in-family OOD:        L ∈ {2.5, 3.0} (per-mode bars).
  - Rectangle in-family OOD:       sizes a ∈ {1.8, 2.2} (outside the training
                                   range), aspect ratios spanning [0.5, 3.0]
                                   (per-mode bars).
  - Rectangle extrapolation map:   mean relative error over the K leading
                                   eigenvalues, scanned across a 2D grid of
                                   (a, ρ=a/b) that extends well outside the
                                   training box.  Positive evidence: the
                                   low-error region is much larger than the
                                   training region itself.

Success criterion (Pillar 2 PoC):
  Relative eigenvalue error < 5 % for k ≤ K on held-out (in-family OOD) geometries.

Runtime: < 30 min on CPU; significantly faster on GPU with --device cuda.
Code mirrors the validation protocol in architectures.tex, Section IV.3
(Pillar 2: Geometry-adaptive spectral backbone), Experiment E3a.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry.lb_truth import (
    interval_eigenvalues, rectangle_eigenvalues,
)
from geometry.encoder import EigenvalueEncoder


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def interval_dataset(n: int, K: int,
                     L_range: tuple = (0.5, 2.0), seed: int = 0) -> TensorDataset:
    rng = np.random.default_rng(seed)
    Ls   = rng.uniform(*L_range, size=n).astype(np.float32)
    eigs = np.stack([interval_eigenvalues(L, K) for L in Ls]).astype(np.float32)
    return TensorDataset(torch.tensor(Ls[:, None]), torch.tensor(eigs))


def rectangle_dataset(n: int, K: int,
                      size_range: tuple = (0.5, 1.5),
                      aspect_range: tuple = (0.5, 3.0),
                      seed: int = 1) -> TensorDataset:
    """Rectangles parametrised by size a and aspect ratio a/b.

    Sampling by aspect ratio (rather than independent a, b) makes the ratio range
    exactly [0.5, 3.0] as stated in the proposal.
    """
    rng = np.random.default_rng(seed)
    a   = rng.uniform(*size_range, size=n).astype(np.float32)
    rho = rng.uniform(*aspect_range, size=n).astype(np.float32)   # a / b
    b   = (a / rho).astype(np.float32)
    eigs = np.stack([rectangle_eigenvalues(ai, bi, K)
                     for ai, bi in zip(a, b)]).astype(np.float32)
    return TensorDataset(
        torch.tensor(np.stack([a, b], axis=1)),   # (n, 2)
        torch.tensor(eigs),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: nn.Module, ds: TensorDataset,
          n_epochs: int = 600, batch: int = 128, lr: float = 3e-3,
          device: str = "cpu") -> list:
    model.to(device)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    history = []
    for _ in range(n_epochs):
        running = 0.0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x)
            # Log-space MSE: scale-robust, treats all eigenmodes equally
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


def eval_interval(model: EigenvalueEncoder, Ls, K: int) -> np.ndarray:
    model.eval()
    dev = _device_of(model)
    x = torch.tensor([[L] for L in Ls], dtype=torch.float32, device=dev)
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    true = np.stack([interval_eigenvalues(L, K) for L in Ls])
    return rel_err_per_mode(pred, true)   # (K,)


def eval_rectangle(model: EigenvalueEncoder, ab_pairs, K: int) -> np.ndarray:
    model.eval()
    dev = _device_of(model)
    x = torch.tensor(ab_pairs, dtype=torch.float32, device=dev)        # (M, 2)
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    true = np.stack([rectangle_eigenvalues(a, b, K) for a, b in ab_pairs])
    return rel_err_per_mode(pred, true)


def rectangle_error_grid(model: EigenvalueEncoder, K: int,
                          a_grid: np.ndarray, rho_grid: np.ndarray) -> np.ndarray:
    """Mean relative error over the K leading eigenvalues, on a grid of (a, ρ).

    Returns a 2D array of shape (len(a_grid), len(rho_grid)).  Each entry is
    the mean over the K leading eigenmodes of |λ̂_k − λ_k| / λ_k for the
    rectangle [0, a] × [0, a/ρ].
    """
    model.eval()
    dev = _device_of(model)
    A, R = np.meshgrid(a_grid, rho_grid, indexing="ij")
    a_flat = A.reshape(-1).astype(np.float32)
    b_flat = (A / R).reshape(-1).astype(np.float32)
    x = torch.tensor(np.stack([a_flat, b_flat], axis=1), device=dev)
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    true = np.stack([rectangle_eigenvalues(float(a), float(b), K)
                     for a, b in zip(a_flat, b_flat)])
    err  = np.abs(pred - true) / (true + 1e-12)        # (M, K)
    return err.mean(axis=1).reshape(A.shape)            # (len(a), len(rho))


# ---------------------------------------------------------------------------
# Per-family runs
# ---------------------------------------------------------------------------

def run_interval(K: int, n_train: int, n_epochs: int,
                  batch: int, device: str) -> dict:
    print(f"\nTraining interval encoder  K={K}  n={n_train}  epochs={n_epochs}  device={device}")
    enc = EigenvalueEncoder(d_in=1, K=K)
    history = train(enc, interval_dataset(n_train, K),
                     n_epochs=n_epochs, batch=batch, device=device)
    print(f"  Final log-MSE: {history[-1]:.4f}")

    err_ood = eval_interval(enc, [2.5, 3.0], K)

    print("  In-family OOD (L=2.5, 3.0):")
    for k, e in enumerate(err_ood):
        print(f"    mode k={k+1:2d}   rel-err = {e:.4f}")
    ok = bool((err_ood < 0.05).all())
    print(f"  PoC criterion (< 5 %): {'PASS ✓' if ok else 'FAIL ✗'}")
    return {"err_ood": err_ood, "pass": ok}


def run_rectangle(K: int, n_train: int, n_epochs: int,
                   batch: int, device: str) -> dict:
    print(f"\nTraining rectangle encoder  K={K}  n={n_train}  epochs={n_epochs}  device={device}")
    enc = EigenvalueEncoder(d_in=2, K=K)
    history = train(enc, rectangle_dataset(n_train, K),
                     n_epochs=n_epochs, batch=batch, device=device)
    print(f"  Final log-MSE: {history[-1]:.4f}")

    # In-family OOD: sizes outside [0.5, 1.5], aspect ratios spanning [0.5, 3.0].
    ood_pairs = [(a, a / rho)
                 for a in (1.8, 2.2)
                 for rho in (0.5, 1.0, 2.0, 3.0)]
    err_ood = eval_rectangle(enc, ood_pairs, K)

    print("  In-family OOD (a∈{1.8,2.2}, a/b∈[0.5,3.0]):")
    for k, e in enumerate(err_ood):
        print(f"    mode k={k+1:2d}   rel-err = {e:.4f}")
    ok = bool((err_ood < 0.05).all())
    print(f"  PoC criterion (< 5 %): {'PASS ✓' if ok else 'FAIL ✗'}")

    # Extrapolation map: scan well outside the training box (training:
    # a ∈ [0.5, 1.5], ρ = a/b ∈ [0.5, 3.0]).
    a_grid   = np.linspace(0.3, 2.5, 24)
    rho_grid = np.linspace(0.3, 4.0, 24)
    err_map  = rectangle_error_grid(enc, K, a_grid, rho_grid)
    return {"err_ood": err_ood, "pass": ok,
            "a_grid": a_grid, "rho_grid": rho_grid, "err_map": err_map}


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


def _heatmap_panel(ax, a_grid, rho_grid, err_map, title,
                    train_a=(0.5, 1.5), train_rho=(0.5, 3.0)):
    # log scale so the 5 % level is visible regardless of magnitude
    pcm = ax.pcolormesh(rho_grid, a_grid, np.clip(err_map, 1e-4, 1.0),
                        cmap="viridis", shading="auto",
                        norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=1.0))
    # 5 % contour overlay
    cs = ax.contour(rho_grid, a_grid, err_map, levels=[0.05],
                    colors="red", linewidths=1.5, linestyles="--")
    ax.clabel(cs, fmt="5 %%", inline=True, fontsize=9)
    # Training box overlay
    ax.add_patch(plt.Rectangle((train_rho[0], train_a[0]),
                                train_rho[1] - train_rho[0],
                                train_a[1] - train_a[0],
                                fill=False, edgecolor="white",
                                linewidth=1.5, linestyle="-"))
    ax.text(train_rho[0] + 0.05, train_a[1] - 0.08, "training box",
            color="white", fontsize=8, va="top")
    ax.set_xlabel(r"Aspect ratio $\rho = a/b$")
    ax.set_ylabel("Side length $a$")
    ax.set_title(title)
    plt.colorbar(pcm, ax=ax, label="mean rel. error (K modes)")


def make_plot(interval, rectangle, K, out_dir):
    panels = []
    if interval is not None:
        panels.append(("bar", interval["err_ood"],
                       "E3a: interval, in-family OOD"))
    if rectangle is not None:
        panels.append(("bar", rectangle["err_ood"],
                       "E3a: rectangle, in-family OOD"))
        panels.append(("heatmap", rectangle,
                       "E3a: rectangle, extrapolation map"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2),
                             squeeze=False)
    for ax, (kind, data, title) in zip(axes[0], panels):
        if kind == "bar":
            _bar_panel(ax, data, title)
        else:
            _heatmap_panel(ax, data["a_grid"], data["rho_grid"],
                            data["err_map"], title)

    fig.suptitle("Experiment E3a: LB eigenvalue prediction (Phase A, Pillar 2)",
                 fontsize=12)
    fig.tight_layout()
    out = Path(out_dir) / "e3a_eigenvalue_errors.pdf"
    fig.savefig(out)
    print(f"\nPlot: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _auto_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K",        type=int, default=10,
                    help="Number of eigenvalues the encoder predicts.")
    ap.add_argument("--n_train",  type=int, default=1000)
    ap.add_argument("--n_epochs", type=int, default=600)
    ap.add_argument("--batch",    type=int, default=128)
    ap.add_argument("--device",   default="auto",
                    help="'auto' (cuda if available), 'cuda', or 'cpu'.")
    ap.add_argument("--family",   choices=["interval", "rectangle", "both"],
                    default="both", help="Geometry family/families to train")
    ap.add_argument("--out_dir",  default="results_e3a")
    args = ap.parse_args()
    K = args.K
    device = _auto_device(args.device)
    Path(args.out_dir).mkdir(exist_ok=True)

    interval = rectangle = None
    if args.family in ("interval", "both"):
        interval = run_interval(K, args.n_train, args.n_epochs,
                                 args.batch, device)
    if args.family in ("rectangle", "both"):
        rectangle = run_rectangle(K, args.n_train, args.n_epochs,
                                   args.batch, device)

    make_plot(interval, rectangle, K, args.out_dir)

    print("\nInterpretation:")
    print("  In-family OOD: a PASS means the encoder learned the eigenvalue")
    print("  scaling law (1/L² for intervals, (m/a)²+(n/b)² for rectangles)")
    print("  rather than a per-shape lookup.")
    if rectangle is not None:
        emap = rectangle["err_map"]
        frac_under_5 = float((emap < 0.05).mean())
        print(f"  Rectangle extrapolation: {100*frac_under_5:.1f} % of the "
              f"({len(rectangle['a_grid'])}×{len(rectangle['rho_grid'])}) "
              f"(a, ρ) grid stays under the 5 % threshold,")
        print(f"  including regions well outside the training box "
              f"(a ∈ [0.5, 1.5], ρ ∈ [0.5, 3.0]).")


if __name__ == "__main__":
    main()
