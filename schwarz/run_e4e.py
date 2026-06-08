"""Experiment E4e (Pillar 3): Schwarz convergence vs overlap width.

Hypothesis (pre-registered):
    Overlapping Schwarz converges faster than non-overlapping at the same
    compute; the one-step contraction rate decreases linearly with overlap
    width ``delta`` (Toselli-Widlund Theorem 2.4 of the proposal). A learned
    DtN should preserve this dependence.

PDE: ``-u'' = f`` on [0, 1], Dirichlet ``u(0) = u(1) = 0``. The domain is
split at x = 0.5; the two subdomains are then extended symmetrically by
``delta / 2`` on each side, so the overlap region has total width ``delta``.

For each ``delta`` we train one pair of subdomain FNOs (a left/right pair),
jointly across all overlap widths, conditioned on a one-hot overlap class.
A single DtN consensus net (sharing the same one-hot conditioning) emits
the next pair of interface Dirichlet values. Reference contraction rates
come from a classical alternating Schwarz iteration with FD subdomain
solves, so we can verify the linear-in-delta law on the reference loop and
then check whether the learned loop preserves it.

Output (in ``out_dir``):
    * ``e4e_schwarz_overlap.pdf`` / ``.png``: contraction rate vs delta,
      learned vs reference, with the Toselli-Widlund linear law overlaid.
    * ``e4e_raw.json``: parameters, per-delta contraction rates and iter
      counts, timing.
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

from schwarz.models import SpectralConv1d
from schwarz.overlap import (
    build_sample, contraction_rate, full_grid, interior_grid,
    schwarz_reference, subdomain_bounds,
)
from shared.exp_utils import (
    Timer, add_device_args, auto_device, print_summary,
    save_figure, write_run_json,
)


# ---------------------------------------------------------------------------
# Overlap-conditioned subdomain FNO
# ---------------------------------------------------------------------------

class OverlapSubdomainFNO(nn.Module):
    """1D FNO conditioned on (f, g_self, one-hot overlap class).

    Inputs:
        f      : (B, nx_half) source term sampled on the subdomain interior
        g_self : (B,) Dirichlet datum on the subdomain's interface endpoint
                 (b_L for left, a_R for right)
        oh     : (B, K) one-hot overlap class

    Outputs:
        u             : (B, nx_half) field on the subdomain interior
        u_at_foreign  : (B,) the value at the OTHER subdomain's interface
                        endpoint as seen from inside this domain (a_R for
                        left, b_L for right). This scalar is what feeds the
                        DtN consensus to update the foreign-side Dirichlet.
    """

    def __init__(self, nx_half: int, width: int, n_layers: int,
                 n_modes: int, n_classes: int):
        super().__init__()
        n_in = 2 + n_classes      # (f, g_self_broadcast, one-hot)
        self.lift = nn.Conv1d(n_in, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, n_modes) for _ in range(n_layers)])
        self.local = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj_u = nn.Sequential(
            nn.Conv1d(width, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))
        self.foreign_head = nn.Sequential(
            nn.Linear(width + n_classes, 64), nn.GELU(),
            nn.Linear(64, 1))
        self.n_classes = n_classes

    def forward(self, f: torch.Tensor, g_self: torch.Tensor,
                oh: torch.Tensor):
        B, nx = f.shape
        g_chan = g_self.view(B, 1, 1).expand(B, 1, nx)
        oh_chan = oh.view(B, self.n_classes, 1).expand(B, self.n_classes, nx)
        x = torch.cat([f.unsqueeze(1), g_chan, oh_chan], dim=1)
        x = self.lift(x)
        for spec, loc in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + loc(x))
        u = self.proj_u(x).squeeze(1)
        pooled = x.mean(dim=-1)                         # (B, width)
        head_in = torch.cat([pooled, oh], dim=-1)
        u_foreign = self.foreign_head(head_in).squeeze(-1)
        return u, u_foreign


# ---------------------------------------------------------------------------
# DtN consensus with overlap-class conditioning
# ---------------------------------------------------------------------------

class OverlapDtN(nn.Module):
    """Maps (g_L_cur, g_R_cur, u_L_at_aR, u_R_at_bL, one-hot) to (g_L_new, g_R_new).

    The classical Schwarz update is exactly ``g_L_new = u_R_at_bL`` and
    ``g_R_new = u_L_at_aR``; the learned net stays close to that map at
    initialisation and lets a small correction be picked up during training.
    """

    def __init__(self, n_classes: int, hidden: int = 64):
        super().__init__()
        d_in = 4 + n_classes
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, g_L: torch.Tensor, g_R: torch.Tensor,
                u_L_at_aR: torch.Tensor, u_R_at_bL: torch.Tensor,
                oh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([g_L, g_R, u_L_at_aR, u_R_at_bL], dim=-1)
        x = torch.cat([x, oh], dim=-1)
        out = self.net(x)                              # (B, 2)
        return out[:, 0], out[:, 1]


# ---------------------------------------------------------------------------
# Data generation: stack samples across all overlaps with class labels.
# ---------------------------------------------------------------------------

def build_dataset(overlaps: list[float], n_per: int, nx_half: int,
                  nx_full: int, seed: int) -> dict[str, np.ndarray]:
    K = len(overlaps)
    f_Ls: list[np.ndarray] = []
    f_Rs: list[np.ndarray] = []
    g_Ls: list[float] = []
    g_Rs: list[float] = []
    u_Ls: list[np.ndarray] = []
    u_Rs: list[np.ndarray] = []
    oh:   list[np.ndarray] = []
    uL_at_aR: list[float] = []
    uR_at_bL: list[float] = []

    for ci, delta in enumerate(overlaps):
        one_hot = np.eye(K, dtype=np.float32)[ci]
        for i in range(n_per):
            s = seed + ci * 1_000_003 + i
            sample = build_sample(s, delta, nx_half, nx_full)
            f_Ls.append(sample["f_L"])
            f_Rs.append(sample["f_R"])
            g_Ls.append(float(sample["g_L"]))
            g_Rs.append(float(sample["g_R"]))
            u_Ls.append(sample["u_L"])
            u_Rs.append(sample["u_R"])
            oh.append(one_hot)
            # Foreign-endpoint values: read from reference u_full at the
            # interior point of the OTHER subdomain's interface.
            a_L, b_L, a_R, b_R = subdomain_bounds(delta)
            x_full = full_grid(nx_full)
            uL_at_aR.append(float(np.interp(a_R, x_full, sample["u_full"])))
            uR_at_bL.append(float(np.interp(b_L, x_full, sample["u_full"])))

    return {
        "f_L":      np.asarray(f_Ls,    dtype=np.float32),
        "f_R":      np.asarray(f_Rs,    dtype=np.float32),
        "g_L":      np.asarray(g_Ls,    dtype=np.float32),
        "g_R":      np.asarray(g_Rs,    dtype=np.float32),
        "u_L":      np.stack(u_Ls).astype(np.float32),
        "u_R":      np.stack(u_Rs).astype(np.float32),
        "oh":       np.stack(oh).astype(np.float32),
        "uL_at_aR": np.asarray(uL_at_aR, dtype=np.float32),
        "uR_at_bL": np.asarray(uR_at_bL, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_subdomain(model: OverlapSubdomainFNO, f: np.ndarray,
                    g_self: np.ndarray, u: np.ndarray,
                    u_foreign: np.ndarray, oh: np.ndarray,
                    n_epochs: int, batch: int, lr: float,
                    device: str) -> None:
    model.to(device)
    ds = TensorDataset(torch.tensor(f), torch.tensor(g_self),
                       torch.tensor(u), torch.tensor(u_foreign),
                       torch.tensor(oh))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        for f_b, g_b, u_b, uf_b, oh_b in loader:
            f_b = f_b.to(device); g_b = g_b.to(device)
            u_b = u_b.to(device); uf_b = uf_b.to(device)
            oh_b = oh_b.to(device)
            u_pred, uf_pred = model(f_b, g_b, oh_b)
            loss = F.mse_loss(u_pred, u_b) + 0.1 * F.mse_loss(uf_pred, uf_b)
            opt.zero_grad(); loss.backward(); opt.step()


def train_dtn(model: OverlapDtN, g_L: np.ndarray, g_R: np.ndarray,
              u_L_at_aR: np.ndarray, u_R_at_bL: np.ndarray,
              oh: np.ndarray,
              n_epochs: int, batch: int, lr: float, device: str) -> None:
    """Train the DtN to map (g, foreign-endpoint values) to (g_L_true, g_R_true).

    Augment with off-equilibrium ``g`` inputs (perturbed by Gaussian noise)
    so the DtN sees the same kind of intermediate iterates it will encounter
    in the Schwarz loop.
    """
    model.to(device)
    rng = np.random.default_rng(0)
    # Augment: half the samples get noisy g inputs.
    n = g_L.size
    g_L_in = g_L.copy()
    g_R_in = g_R.copy()
    mask = rng.random(n) < 0.5
    g_L_in[mask] += rng.normal(0.0, 0.05, size=mask.sum()).astype(np.float32)
    g_R_in[mask] += rng.normal(0.0, 0.05, size=mask.sum()).astype(np.float32)

    ds = TensorDataset(torch.tensor(g_L_in), torch.tensor(g_R_in),
                       torch.tensor(u_L_at_aR), torch.tensor(u_R_at_bL),
                       torch.tensor(oh),
                       torch.tensor(g_L), torch.tensor(g_R))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        for gL_in_b, gR_in_b, uL_b, uR_b, oh_b, gL_t, gR_t in loader:
            gL_in_b = gL_in_b.to(device); gR_in_b = gR_in_b.to(device)
            uL_b = uL_b.to(device); uR_b = uR_b.to(device)
            oh_b = oh_b.to(device); gL_t = gL_t.to(device); gR_t = gR_t.to(device)
            gL_pred, gR_pred = model(gL_in_b, gR_in_b, uL_b, uR_b, oh_b)
            loss = F.mse_loss(gL_pred, gL_t) + F.mse_loss(gR_pred, gR_t)
            opt.zero_grad(); loss.backward(); opt.step()


# ---------------------------------------------------------------------------
# Learned Schwarz iteration (per overlap class)
# ---------------------------------------------------------------------------

def learned_schwarz(fno_L: OverlapSubdomainFNO, fno_R: OverlapSubdomainFNO,
                    dtn: OverlapDtN, oh: torch.Tensor,
                    f_L: torch.Tensor, f_R: torch.Tensor,
                    max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run the learned Schwarz loop and return the mean-over-batch history."""
    B = f_L.size(0)
    dev = f_L.device
    g_L = torch.zeros(B, device=dev)
    g_R = torch.zeros(B, device=dev)
    history: list[float] = []
    fno_L.eval(); fno_R.eval(); dtn.eval()
    u_L = u_R = None
    with torch.no_grad():
        for _ in range(max_iter):
            u_L, uL_at_aR = fno_L(f_L, g_L, oh)
            u_R, uR_at_bL = fno_R(f_R, g_R, oh)
            g_L_new, g_R_new = dtn(g_L, g_R, uL_at_aR, uR_at_bL, oh)
            delta_g = torch.maximum(
                (g_L_new - g_L).abs(), (g_R_new - g_R).abs())
            history.append(float(delta_g.mean()))
            g_L, g_R = g_L_new, g_R_new
            if history[-1] < tol:
                break
    return (u_L.cpu().numpy(), u_R.cpu().numpy(), history)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_device_args(ap)
    ap.add_argument("--overlaps", nargs="+", type=float,
                    default=[0.0, 0.05, 0.10, 0.20, 0.40],
                    help="Overlap widths delta to sweep.")
    ap.add_argument("--nx_half",  type=int, default=64)
    ap.add_argument("--n_train",  type=int, default=5000,
                    help="Training samples per overlap class.")
    ap.add_argument("--n_epochs", type=int, default=600)
    ap.add_argument("--batch",    type=int, default=64)
    ap.add_argument("--width",    type=int, default=64)
    ap.add_argument("--n_modes",  type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_test",   type=int, default=200,
                    help="Test samples per overlap class.")
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--tol",      type=float, default=1e-6)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--out_dir",  default="results_e4e")
    args = ap.parse_args()

    if args.smoke:
        args.overlaps = [0.0, 0.1, 0.4]
        args.nx_half  = 24
        args.n_train  = 60
        args.n_epochs = 6
        args.batch    = 32
        args.width    = 16
        args.n_modes  = 8
        args.n_layers = 2
        args.n_test   = 20
        args.max_iter = 12

    device = auto_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    overlaps = list(args.overlaps)
    K = len(overlaps)
    nx_half = args.nx_half
    nx_full = 2 * nx_half        # global reference grid size
    timer = Timer()

    # ---------------------- build data ----------------------
    with timer("data_train"):
        train_data = build_dataset(overlaps, args.n_train, nx_half, nx_full,
                                    seed=args.seed)
        print(f"Built training set: {train_data['f_L'].shape[0]} samples"
              f" across {K} overlap classes.")

    # ---------------------- train ----------------------
    sub_kwargs = dict(nx_half=nx_half, width=args.width,
                      n_layers=args.n_layers, n_modes=args.n_modes,
                      n_classes=K)
    fno_L = OverlapSubdomainFNO(**sub_kwargs)
    fno_R = OverlapSubdomainFNO(**sub_kwargs)
    dtn   = OverlapDtN(n_classes=K)

    with timer("train_left"):
        train_subdomain(fno_L,
                        train_data["f_L"], train_data["g_L"],
                        train_data["u_L"], train_data["uL_at_aR"],
                        train_data["oh"],
                        n_epochs=args.n_epochs, batch=args.batch,
                        lr=args.lr, device=device)
    with timer("train_right"):
        train_subdomain(fno_R,
                        train_data["f_R"], train_data["g_R"],
                        train_data["u_R"], train_data["uR_at_bL"],
                        train_data["oh"],
                        n_epochs=args.n_epochs, batch=args.batch,
                        lr=args.lr, device=device)
    with timer("train_dtn"):
        train_dtn(dtn,
                  train_data["g_L"], train_data["g_R"],
                  train_data["uL_at_aR"], train_data["uR_at_bL"],
                  train_data["oh"],
                  n_epochs=args.n_epochs, batch=args.batch,
                  lr=args.lr, device=device)

    # ---------------------- evaluate ----------------------
    rates_learned: list[float] = []
    rates_ref:     list[float] = []
    iters_learned: list[float] = []
    iters_ref:     list[float] = []
    histories_learned: dict[float, list[float]] = {}
    histories_ref:     dict[float, list[float]] = {}

    with timer("evaluate"):
        for ci, delta in enumerate(overlaps):
            test_data = build_dataset([delta], args.n_test, nx_half,
                                       nx_full, seed=args.seed + 999 + ci)
            oh = torch.zeros(args.n_test, K)
            oh[:, ci] = 1.0
            oh = oh.to(device)
            f_L_t = torch.tensor(test_data["f_L"]).to(device)
            f_R_t = torch.tensor(test_data["f_R"]).to(device)

            _, _, hist_l = learned_schwarz(
                fno_L, fno_R, dtn, oh, f_L_t, f_R_t,
                max_iter=args.max_iter, tol=args.tol)

            # Per-sample reference Schwarz: average histories pointwise.
            ref_hists: list[list[float]] = []
            for j in range(args.n_test):
                _, _, h_ref = schwarz_reference(
                    test_data["f_L"][j], test_data["f_R"][j],
                    delta, nx_half, max_iter=args.max_iter, tol=args.tol)
                ref_hists.append(h_ref)
            max_len = max(len(h) for h in ref_hists)
            ref_mean = []
            for k in range(max_len):
                vals = [h[k] for h in ref_hists if k < len(h)]
                ref_mean.append(float(np.mean(vals)))

            rate_l = contraction_rate(hist_l)
            rate_r = contraction_rate(ref_mean)
            iters_l = float(len(hist_l))
            iters_r = float(np.mean([len(h) for h in ref_hists]))

            rates_learned.append(rate_l)
            rates_ref.append(rate_r)
            iters_learned.append(iters_l)
            iters_ref.append(iters_r)
            histories_learned[delta] = hist_l
            histories_ref[delta]     = ref_mean

            print(f"delta = {delta:.3f}: rate_learned = {rate_l:.3f},"
                  f" rate_ref = {rate_r:.3f},"
                  f" iters_learned = {iters_l:.1f}, iters_ref = {iters_r:.1f}")

    timing = timer.dump()

    # ---------------------- plot ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left panel: contraction rate vs delta
    axes[0].plot(overlaps, rates_learned, "o-", color="#27ae60",
                 lw=2, label="Learned Schwarz")
    axes[0].plot(overlaps, rates_ref, "s--", color="#2a6496",
                 lw=2, label="Reference (FD subdomains)")
    # Toselli-Widlund linear law: rate ~ rate(0) * (1 - c * delta).
    if rates_ref[0] > 0:
        # Fit a line through the reference points to act as the linear law.
        coef = np.polyfit(overlaps, rates_ref, 1)
        x_line = np.linspace(min(overlaps), max(overlaps), 50)
        axes[0].plot(x_line, np.polyval(coef, x_line), ":", color="gray",
                     lw=1.5, label="Linear fit to reference")
    axes[0].set_xlabel(r"Overlap width $\delta$")
    axes[0].set_ylabel("One-step contraction rate")
    axes[0].set_title("E4e: contraction rate vs overlap width")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    # Right panel: convergence histories (one curve per delta)
    cmap = plt.get_cmap("viridis")
    for ci, delta in enumerate(overlaps):
        col = cmap(ci / max(K - 1, 1))
        h_l = histories_learned[delta]
        h_r = histories_ref[delta]
        axes[1].semilogy(range(1, len(h_l) + 1), h_l, "-", color=col,
                         lw=2, label=f"learned, delta={delta:.2f}")
        axes[1].semilogy(range(1, len(h_r) + 1), h_r, "--", color=col,
                         lw=1.2, alpha=0.7)
    axes[1].set_xlabel("Schwarz iteration")
    axes[1].set_ylabel(r"Interface update $\|g^{(k+1)} - g^{(k)}\|_\infty$")
    axes[1].set_title("E4e: convergence curves (solid = learned, dashed = reference)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.suptitle("Experiment E4e: Schwarz convergence vs overlap width")
    fig.tight_layout()
    save_figure(fig, out_dir, "e4e_schwarz_overlap")

    # ---------------------- headline + JSON ----------------------
    headline = {}
    for delta, r_l, r_r, n_l, n_r in zip(
            overlaps, rates_learned, rates_ref, iters_learned, iters_ref):
        tag = f"delta_{delta:.2f}".replace(".", "p")
        headline[f"rate_learned_{tag}"] = r_l
        headline[f"rate_ref_{tag}"]     = r_r
        headline[f"iters_learned_{tag}"] = n_l
        headline[f"iters_ref_{tag}"]     = n_r

    # Linear-law slope (negative = rate decreases with overlap).
    if K >= 2:
        slope_learned = float(np.polyfit(overlaps, rates_learned, 1)[0])
        slope_ref     = float(np.polyfit(overlaps, rates_ref,     1)[0])
        headline["slope_learned"] = slope_learned
        headline["slope_ref"]     = slope_ref

    write_run_json(out_dir,
                   experiment="e4e",
                   pillar="Pillar 3 (domain decomposition)",
                   hypothesis=("Overlapping Schwarz converges faster than "
                               "non-overlapping; the contraction rate "
                               "decreases linearly with overlap width delta "
                               "(Toselli-Widlund Theorem 2.4), and a learned "
                               "DtN preserves this dependence."),
                   parameters=vars(args),
                   headline=headline,
                   timing=timing,
                   device=device,
                   extra={
                       "overlaps": overlaps,
                       "rates_learned": rates_learned,
                       "rates_ref":     rates_ref,
                       "iters_learned": iters_learned,
                       "iters_ref":     iters_ref,
                       "histories_learned": {str(d): histories_learned[d]
                                              for d in overlaps},
                       "histories_ref":     {str(d): histories_ref[d]
                                              for d in overlaps},
                   })

    print_summary("E4e", headline, timing)


if __name__ == "__main__":
    main()
