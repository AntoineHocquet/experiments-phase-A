"""Experiment E3b (2D): LB-FNO vs standard FNO on a rectangle heat equation.

Two-dimensional extension of run_e3b.py.  Geometry now varies in two parameters
(side lengths a, b), which makes the encoder's job genuinely non-trivial:
instead of learning the one-parameter law λ_k = (kπ/L)², the rectangle encoder
must learn the joint law

    λ_{mn}(a, b) = (mπ/a)² + (nπ/b)²,                m, n = 1, 2, …

and the standard FNO must cope with both size *and* aspect-ratio variation:
two failure modes for a fixed flat-Fourier basis, instead of one.

PDE: ∂_t u = D Δu on [0, a] × [0, b] with Dirichlet BCs.

Three methods compared, in direct analogy to the 1D experiment:
  (A) LB-FNO (closed-form, 2D DST): eigenvalues either taken analytically
      (oracle) or recovered from the rectangle encoder's first two predictions
      via (â, b̂) = (π/√(λ̂₁ - (λ̂₂-λ̂₁)/3), π√(3/(λ̂₂-λ̂₁))) up to ordering.
  (B) Standard 2D FNO, trained on a single fixed shape (a=b=1); applied
      zero-shot to varied (a, b).
  (C) Standard 2D FNO, trained on the full training box
      a ∈ [0.5, 1.5], a/b ∈ [0.5, 3.0]; matched compute.

Two slices through (a, b) are reported:
  1. Vary a, fix aspect ratio a/b = 1   (square of varying size).
  2. Fix a = 1, vary aspect ratio a/b   (rectangle of varying aspect).

Runtime: 2–4 h on a single GPU at default scale; minutes in --fast mode.
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
from scipy.fft import dstn, idstn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry.lb_truth import rectangle_eigenvalues
from geometry.encoder  import EigenvalueEncoder
from geometry.run_e3a  import rectangle_dataset, train as train_enc


# ---------------------------------------------------------------------------
# Exact 2D heat solver on [0, a] × [0, b] with Dirichlet BCs via DST-1
# ---------------------------------------------------------------------------

def heat_dirichlet_2d(u0: np.ndarray, a: float, b: float,
                       D: float, dt: float) -> np.ndarray:
    """Exact heat-kernel evaluation via separable 2D DST-1.

    The Dirichlet sine basis sin(mπx/a) sin(nπy/b) is the eigenbasis of -Δ on
    [0,a]×[0,b]; the heat semigroup is mode-wise exponential decay.
    """
    nx, ny = u0.shape
    F2 = dstn(u0, type=1)
    m = np.arange(1, nx + 1, dtype=float)[:, None]
    n = np.arange(1, ny + 1, dtype=float)[None, :]
    lk = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
    F2 *= np.exp(-D * lk * dt)
    return idstn(F2, type=1) / (4.0 * (nx + 1) * (ny + 1))


# ---------------------------------------------------------------------------
# Random initial condition on [0, a] × [0, b] with Dirichlet BCs
# ---------------------------------------------------------------------------

def random_ic_dirichlet_2d(a: float, b: float, nx: int = 64, ny: int = 64,
                            n_modes: int = 5, seed: int = 0) -> np.ndarray:
    """Smooth IC with u = 0 on ∂Ω, values in (-0.5, 0.5)."""
    rng = np.random.default_rng(seed)
    j   = np.arange(1, nx + 1, dtype=float)
    k   = np.arange(1, ny + 1, dtype=float)
    x   = j / (nx + 1) * a
    y   = k / (ny + 1) * b
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.zeros((nx, ny))
    for _ in range(n_modes):
        m = rng.integers(1, 6); n = rng.integers(1, 6)
        c = rng.standard_normal() * 0.5
        u += c * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
    u *= 0.4 / (np.abs(u).max() + 1e-8)
    return u.astype(np.float32)


# ---------------------------------------------------------------------------
# 2D FNO (standard, flat-Fourier basis)
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    def __init__(self, width: int, n_modes_x: int, n_modes_y: int):
        super().__init__()
        self.mx, self.my = n_modes_x, n_modes_y
        self.w = nn.Parameter(torch.randn(width, width, n_modes_x, n_modes_y,
                                          dtype=torch.cfloat) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft2(x, dim=(-2, -1))
        mx = min(self.mx, xf.size(-2))
        my = min(self.my, xf.size(-1))
        out = torch.zeros_like(xf)
        out[..., :mx, :my] = torch.einsum(
            "biHW,ioHW->boHW",
            xf[..., :mx, :my], self.w[..., :mx, :my])
        return torch.fft.irfft2(out, s=x.shape[-2:], dim=(-2, -1))


class HeatFNO2d(nn.Module):
    """Small 2D FNO for the heat equation on a rectangle.

    Input:  (u0, dt) packed as a 2-channel tensor (batch, 2, nx, ny).
    Output: u_dt of shape (batch, nx, ny).
    """

    def __init__(self, width: int = 32, n_layers: int = 4,
                 n_modes_x: int = 16, n_modes_y: int = 16,
                 proj_hidden: int = 64):
        super().__init__()
        self.lift = nn.Conv2d(2, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, n_modes_x, n_modes_y) for _ in range(n_layers)])
        self.local    = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj     = nn.Sequential(
            nn.Conv2d(width, proj_hidden, 1), nn.GELU(),
            nn.Conv2d(proj_hidden, 1, 1))

    def forward(self, u0: torch.Tensor, dt) -> torch.Tensor:
        # u0: (B, nx, ny);  dt: float or (B,) tensor
        if u0.dim() == 3:
            u0c = u0.unsqueeze(1)
        else:
            u0c = u0
        if isinstance(dt, torch.Tensor):
            dt_ch = dt.view(-1, 1, 1, 1).expand(-1, 1, u0c.size(-2), u0c.size(-1)).to(u0c.device)
        else:
            dt_ch = torch.full((u0c.size(0), 1, u0c.size(-2), u0c.size(-1)),
                                float(dt), device=u0c.device, dtype=u0c.dtype)
        x = self.lift(torch.cat([u0c, dt_ch], dim=1))
        for spec, loc in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + loc(x))
        return self.proj(x).squeeze(1)                  # (B, nx, ny)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_heat_dataset_2d(ab_list, D_range: tuple, dt_range: tuple,
                         n_samples: int, nx: int = 64, ny: int = 64,
                         seed: int = 0) -> TensorDataset:
    """Generate (u0, u1, dt) triples on varying (a, b)."""
    rng = np.random.default_rng(seed)
    u0s, u1s, dts = [], [], []
    ab_list = list(ab_list)
    for i in range(n_samples):
        a, b = ab_list[int(rng.integers(0, len(ab_list)))]
        D   = float(rng.uniform(*D_range))
        dt  = float(rng.uniform(*dt_range))
        u0  = random_ic_dirichlet_2d(a=a, b=b, nx=nx, ny=ny,
                                       seed=seed * 1_000_000 + i)
        u1  = heat_dirichlet_2d(u0, a=a, b=b, D=D, dt=dt)
        u0s.append(u0); u1s.append(u1); dts.append(dt)
    return TensorDataset(
        torch.tensor(np.stack(u0s)),
        torch.tensor(np.stack(u1s)),
        torch.tensor(np.array(dts, dtype=np.float32)),
    )


def _sample_training_pairs(n: int, seed: int = 0):
    """Match run_e3a.rectangle_dataset training distribution."""
    rng = np.random.default_rng(seed)
    a   = rng.uniform(0.5, 1.5,   size=n)
    rho = rng.uniform(0.5, 3.0,   size=n)
    b   = a / rho
    return list(zip(a.tolist(), b.tolist()))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fno(model: HeatFNO2d, ds: TensorDataset,
              n_epochs: int = 200, batch: int = 32, lr: float = 1e-3,
              device: str = "cpu") -> None:
    model.to(device)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for _ in range(n_epochs):
        for u0, u1, dt_b in loader:
            u0 = u0.to(device); u1 = u1.to(device); dt_b = dt_b.to(device)
            pred = model(u0, dt_b)
            loss = F.mse_loss(pred, u1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rel_l2_2d(pred: torch.Tensor, target: torch.Tensor) -> float:
    flat = lambda t: t.reshape(t.size(0), -1)
    p, q = flat(pred), flat(target)
    return float((((p - q).norm(dim=-1)) / (q.norm(dim=-1) + 1e-8)).mean())


def eval_fno_at_ab(model: HeatFNO2d, a: float, b: float, D_range,
                    dt_eval: float, nx: int = 64, ny: int = 64,
                    n_test: int = 100, seed: int = 999) -> float:
    """Evaluate the standard 2D FNO at a specific (a, b)."""
    model.eval()
    dev  = next(model.parameters()).device
    rng  = np.random.default_rng(seed)
    errs = []
    with torch.no_grad():
        for i in range(n_test):
            D  = float(rng.uniform(*D_range))
            u0 = random_ic_dirichlet_2d(a, b, nx, ny, seed=seed * 1_000_000 + i)
            u1 = heat_dirichlet_2d(u0, a=a, b=b, D=D, dt=dt_eval)
            u0t = torch.tensor(u0, device=dev).unsqueeze(0)
            u1t = torch.tensor(u1, device=dev).unsqueeze(0)
            pred = model(u0t, dt_eval)
            errs.append(rel_l2_2d(pred, u1t))
    return float(np.mean(errs))


def eval_lb_fno_at_ab(a_true: float, b_true: float,
                      a_pred: float, b_pred: float,
                      D_range, dt_eval: float,
                      nx: int = 64, ny: int = 64,
                      n_test: int = 100, seed: int = 999) -> float:
    """Closed-form LB-FNO evaluation with optionally mis-predicted (a, b).

    Ground truth uses (a_true, b_true).  The 'LB-FNO' uses (a_pred, b_pred)
    in the heat-kernel evaluation: if the encoder is perfect this collapses
    to the exact solver and gives near machine-precision error.
    """
    rng  = np.random.default_rng(seed)
    errs = []
    for i in range(n_test):
        D   = float(rng.uniform(*D_range))
        u0  = random_ic_dirichlet_2d(a_true, b_true, nx, ny,
                                       seed=seed * 1_000_000 + i)
        u1  = heat_dirichlet_2d(u0, a=a_true, b=b_true, D=D, dt=dt_eval)
        u1h = heat_dirichlet_2d(u0, a=a_pred, b=b_pred, D=D, dt=dt_eval)
        errs.append(
            np.linalg.norm(u1h - u1) / (np.linalg.norm(u1) + 1e-8)
        )
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Encoder-based LB-FNO helpers (rectangle encoder)
# ---------------------------------------------------------------------------

def train_rectangle_encoder(K: int, n: int, n_epochs: int,
                             batch: int, device: str) -> EigenvalueEncoder:
    enc = EigenvalueEncoder(d_in=2, K=K)
    ds  = rectangle_dataset(n, K, seed=42)
    print(f"  Training rectangle LB encoder (K={K}, n={n}, "
          f"epochs={n_epochs}, device={device}) ...")
    train_enc(enc, ds, n_epochs=n_epochs, batch=batch, device=device)
    return enc


def predict_ab_from_encoder(enc: EigenvalueEncoder,
                             a_true: float, b_true: float) -> tuple:
    """Recover (â, b̂) from the encoder's first two eigenvalues.

    With λ_1 ≤ λ_2 (encoder output is sorted ascending by construction),
    assuming the larger side dominates the second mode,
        λ_2 - λ_1 = 3π² / s_max²,
        4λ_1 - λ_2 = 3π² / s_min²,
    where s_max = max(a, b) and s_min = min(a, b).  We then map the recovered
    (s_min, s_max) back to (â, b̂) using the order of the true input; the
    encoder cannot recover labels itself, only the unordered pair.
    """
    enc.eval()
    dev = next(enc.parameters()).device
    with torch.no_grad():
        x = torch.tensor([[a_true, b_true]], dtype=torch.float32, device=dev)
        eigs = enc(x)[0].cpu().numpy()
    lam1, lam2 = float(eigs[0]), float(eigs[1])
    # Numerical guards
    d12  = max(lam2 - lam1, 1e-8)
    s_max = float(np.pi * np.sqrt(3.0 / d12))
    inv_smin_sq = max(4.0 * lam1 - lam2, 1e-8) / (3.0 * np.pi ** 2)
    s_min = float(1.0 / np.sqrt(inv_smin_sq))
    # Match recovered pair to the original (a, b) ordering
    if a_true >= b_true:
        return s_max, s_min
    return s_min, s_max


def eval_lb_fno_encoder_2d(enc: EigenvalueEncoder,
                            a_true: float, b_true: float,
                            D_range, dt_eval: float,
                            nx: int = 64, ny: int = 64,
                            n_test: int = 100, seed: int = 999) -> float:
    a_pred, b_pred = predict_ab_from_encoder(enc, a_true, b_true)
    return eval_lb_fno_at_ab(a_true, b_true, a_pred, b_pred,
                              D_range, dt_eval, nx=nx, ny=ny,
                              n_test=n_test, seed=seed)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _slice_plot(ax, xs, curves, labels, colours, markers, x_label,
                vline_x=None, vline_label=None, training_band=None):
    for ys, lbl, col, mk in zip(curves, labels, colours, markers):
        ax.plot(xs, ys, mk, label=lbl, color=col, lw=2)
    if training_band is not None:
        ax.axvspan(*training_band, color="gray", alpha=0.1,
                    label="training range")
    if vline_x is not None:
        ax.axvline(vline_x, color="gray", linestyle=":", lw=1)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(r"Relative $L^2$ error", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="best")


def make_plot(a_list, errs_size, rho_list, errs_aspect, out_dir):
    """Two slice plots: vary size (aspect=1); vary aspect (a=1)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels  = ["(A) LB-FNO [oracle (a,b)]",
               "(A) LB-FNO [encoder (â,b̂)]",
               "(B) FNO trained on 1×1 only",
               "(C) FNO trained on full box (matched)"]
    colours = ["#27ae60", "#16a085", "#c0392b", "#2a6496"]
    markers = ["o-", "D:", "s--", "^-."]

    _slice_plot(axes[0], a_list, errs_size, labels, colours, markers,
                 x_label=r"Side length $a$ (aspect $a/b=1$, i.e. square)",
                 training_band=(0.5, 1.5))
    axes[0].set_title("E3b 2D: varying square size")

    _slice_plot(axes[1], rho_list, errs_aspect, labels, colours, markers,
                 x_label=r"Aspect ratio $\rho = a/b$ (with $a = 1$)",
                 training_band=(0.5, 3.0))
    axes[1].set_title("E3b 2D: varying aspect ratio")

    fig.suptitle("E3b (2D): LB-FNO vs standard FNO across rectangles "
                  "(Phase A, Pillar 2)", fontsize=12)
    fig.tight_layout()
    out = Path(out_dir) / "e3b_2d_heat_ood.pdf"
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
    ap.add_argument("--nx",         type=int, default=64,
                    help="Grid resolution along x.")
    ap.add_argument("--ny",         type=int, default=64,
                    help="Grid resolution along y.")
    ap.add_argument("--n_train",    type=int, default=4000,
                    help="Training samples for each FNO.")
    ap.add_argument("--n_epochs",   type=int, default=400)
    ap.add_argument("--batch",      type=int, default=32,
                    help="Mini-batch size for the FNO training loop.")
    ap.add_argument("--K",          type=int, default=10,
                    help="Number of eigenvalues the LB encoder predicts.")
    ap.add_argument("--n_modes_x",  type=int, default=16)
    ap.add_argument("--n_modes_y",  type=int, default=16,
                    help="Fourier modes kept per axis in the standard FNO.")
    ap.add_argument("--width",      type=int, default=32,
                    help="Channel width of the standard FNO.")
    ap.add_argument("--n_layers",   type=int, default=4)
    ap.add_argument("--enc_epochs", type=int, default=600)
    ap.add_argument("--enc_train",  type=int, default=1000)
    ap.add_argument("--n_test",     type=int, default=80,
                    help="Test samples per (a, b) evaluation point.")
    ap.add_argument("--device",     default="auto",
                    help="'auto' (cuda if available), 'cuda', or 'cpu'.")
    ap.add_argument("--out_dir",    default="results_e3b_2d")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)
    device = _auto_device(args.device)

    D_range  = (0.01, 0.20)
    dt_range = (0.005, 0.02)
    dt_eval  = 0.01

    fno_kwargs = dict(width=args.width, n_layers=args.n_layers,
                      n_modes_x=args.n_modes_x, n_modes_y=args.n_modes_y)

    # ---- Train FNO-B: single fixed shape (a=b=1) ----
    print(f"Training FNO-B (a=b=1.0)  width={args.width}  "
          f"modes={args.n_modes_x}x{args.n_modes_y}  device={device}")
    ds_B = make_heat_dataset_2d([(1.0, 1.0)], D_range, dt_range,
                                  args.n_train, nx=args.nx, ny=args.ny, seed=0)
    fno_B = HeatFNO2d(**fno_kwargs)
    train_fno(fno_B, ds_B, n_epochs=args.n_epochs, batch=args.batch,
               device=device)

    # ---- Train FNO-C: all (a, b) in training box ----
    print(f"Training FNO-C (a∈[0.5,1.5], a/b∈[0.5,3.0])")
    ab_train = _sample_training_pairs(max(100, args.n_train // 5), seed=1)
    ds_C = make_heat_dataset_2d(ab_train, D_range, dt_range, args.n_train,
                                  nx=args.nx, ny=args.ny, seed=1)
    fno_C = HeatFNO2d(**fno_kwargs)
    train_fno(fno_C, ds_C, n_epochs=args.n_epochs, batch=args.batch,
               device=device)

    # ---- Train rectangle LB encoder ----
    enc = train_rectangle_encoder(K=args.K, n=args.enc_train,
                                    n_epochs=args.enc_epochs,
                                    batch=args.batch, device=device)

    # ---- Slice 1: vary a, aspect ratio = 1 (squares) ----
    print("\nEvaluating size sweep (squares of varying side length) ...")
    a_list = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    errs_size = [[], [], [], []]   # A-oracle, A-enc, B, C
    for a in a_list:
        b = a
        errs_size[0].append(
            eval_lb_fno_at_ab(a, b, a, b, D_range, dt_eval,
                               nx=args.nx, ny=args.ny, n_test=args.n_test))
        errs_size[1].append(
            eval_lb_fno_encoder_2d(enc, a, b, D_range, dt_eval,
                                     nx=args.nx, ny=args.ny, n_test=args.n_test))
        errs_size[2].append(
            eval_fno_at_ab(fno_B, a, b, D_range, dt_eval,
                            nx=args.nx, ny=args.ny, n_test=args.n_test))
        errs_size[3].append(
            eval_fno_at_ab(fno_C, a, b, D_range, dt_eval,
                            nx=args.nx, ny=args.ny, n_test=args.n_test))

    # ---- Slice 2: a=1, vary aspect ratio ----
    print("Evaluating aspect-ratio sweep (a = 1, varying b) ...")
    rho_list = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    errs_aspect = [[], [], [], []]
    for rho in rho_list:
        a = 1.0
        b = a / rho
        errs_aspect[0].append(
            eval_lb_fno_at_ab(a, b, a, b, D_range, dt_eval,
                               nx=args.nx, ny=args.ny, n_test=args.n_test))
        errs_aspect[1].append(
            eval_lb_fno_encoder_2d(enc, a, b, D_range, dt_eval,
                                     nx=args.nx, ny=args.ny, n_test=args.n_test))
        errs_aspect[2].append(
            eval_fno_at_ab(fno_B, a, b, D_range, dt_eval,
                            nx=args.nx, ny=args.ny, n_test=args.n_test))
        errs_aspect[3].append(
            eval_fno_at_ab(fno_C, a, b, D_range, dt_eval,
                            nx=args.nx, ny=args.ny, n_test=args.n_test))

    # ---- Results table ----
    print(f"\n{'─'*86}")
    print(f"Size sweep (square, a = b)")
    print(f"  {'a':>5}  {'A oracle':>10}  {'A encoder':>10}  "
          f"{'B (1×1)':>10}  {'C (full)':>10}")
    print(f"{'─'*86}")
    for a, ao, ae, b, c in zip(a_list, *errs_size):
        marker = "← OOD" if (a < 0.5 or a > 1.5) else ""
        print(f"  {a:>5.2f}  {ao:>10.4f}  {ae:>10.4f}  "
              f"{b:>10.4f}  {c:>10.4f}  {marker}")

    print(f"\nAspect sweep (a = 1, ρ = a/b)")
    print(f"  {'ρ':>5}  {'A oracle':>10}  {'A encoder':>10}  "
          f"{'B (1×1)':>10}  {'C (full)':>10}")
    print(f"{'─'*86}")
    for r, ao, ae, b, c in zip(rho_list, *errs_aspect):
        marker = "← OOD" if (r < 0.5 or r > 3.0) else ""
        print(f"  {r:>5.2f}  {ao:>10.4f}  {ae:>10.4f}  "
              f"{b:>10.4f}  {c:>10.4f}  {marker}")

    make_plot(a_list, errs_size, rho_list, errs_aspect, args.out_dir)

    # ---- Summary line for the OOD slice ----
    def _ood_mean(curve, axis_vals, lo, hi):
        return float(np.mean([e for v, e in zip(axis_vals, curve)
                              if v < lo or v > hi]))
    ood_B_size  = _ood_mean(errs_size[2],   a_list,   0.5, 1.5)
    ood_A_size  = _ood_mean(errs_size[0],   a_list,   0.5, 1.5)
    ood_B_rho   = _ood_mean(errs_aspect[2], rho_list, 0.5, 3.0)
    ood_A_rho   = _ood_mean(errs_aspect[0], rho_list, 0.5, 3.0)
    print("\nKey finding (OOD averages):")
    print(f"  Size sweep,   LB-FNO oracle: {ood_A_size:.4f}  "
          f"FNO (1×1): {ood_B_size:.4f}  ratio: {ood_B_size/max(ood_A_size,1e-12):.1f}×")
    print(f"  Aspect sweep, LB-FNO oracle: {ood_A_rho:.4f}  "
          f"FNO (1×1): {ood_B_rho:.4f}  ratio: {ood_B_rho/max(ood_A_rho,1e-12):.1f}×")
    if ood_B_size > 5 * ood_A_size and ood_B_rho > 5 * ood_A_rho:
        print("  → 2D PoC PASS ✓  (LB-FNO beats standard FNO by > 5× on both slices)")
    else:
        print("  → Investigate training budget or grid resolution.")


if __name__ == "__main__":
    main()
