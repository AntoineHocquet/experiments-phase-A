"""Shared Pillar 1 core: a parameter-conditioned 1D Fourier neural operator and
the brick / dataset / training / evaluation utilities used by the
operator-splitting experiments (E2b, E2d, E2e) and the Pillar 1 x Pillar 2
bridge (E5a).

Two linear elementary generators on [0, 1] with periodic boundary conditions:

    A (diffusion):  d_t u = D u_xx              (exact heat kernel)
    B (advection):  d_t u = -v(x) u_x           (linear transport)

with v(x) = v0 * (1 + s * p(x)) for a fixed smooth profile p(x); the shear
amplitude s controls the commutator [A, B] (at s = 0 the velocity is constant
and [A, B] = 0). Each expert is parameter-conditioned (as in the proposal's
S-MoE backbone): the diffusion brick sees (u, dt, D), the advection brick sees
(u, dt, v0), and the monolithic baseline sees (u, dt, D, v0). Without this
conditioning a brick cannot represent its operator family.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Make `shared` importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.simulators import (
    brick_diffusion, brick_advection, simulate_full, batch_random_ics,
)


# ---------------------------------------------------------------------------
# FNO architecture (parameter-conditioned)
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    def __init__(self, width: int, n_modes: int):
        super().__init__()
        self.n_modes = n_modes
        self.w = nn.Parameter(
            torch.randn(width, width, n_modes, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, width, nx)
        x_ft = torch.fft.rfft(x, dim=-1)
        m = min(self.n_modes, x_ft.size(-1))
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :m] = torch.einsum("biM,ioM->boM", x_ft[..., :m], self.w[..., :m])
        return torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)


class FNO1d(nn.Module):
    """1D FNO conditioned on a vector of scalars (dt and PDE parameters).

    Each conditioning scalar is appended as a constant channel, so the model can
    be evaluated at arbitrary step sizes (including Strang half-steps) and across
    the parameter family it was trained on.
    """

    def __init__(self, n_cond: int = 2, width: int = 64,
                 n_layers: int = 4, n_modes: int = 32):
        super().__init__()
        self.lift     = nn.Conv1d(1 + n_cond, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, n_modes) for _ in range(n_layers)])
        self.local    = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj     = nn.Sequential(
            nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, 1, 1))

    def forward(self, u: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # u: (B, nx);  cond: (B, n_cond)
        cond_ch = cond.view(cond.size(0), cond.size(1), 1).expand(-1, -1, u.size(-1))
        x = torch.cat([u.unsqueeze(1), cond_ch.to(u.dtype)], dim=1)   # (B, 1+n_cond, nx)
        x = self.lift(x)
        for spec, loc in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + loc(x))
        return self.proj(x).squeeze(1)                                # (B, nx)


def make_brick(model: nn.Module, param: torch.Tensor):
    """Wrap a parameter-conditioned FNO as a splitting brick callable (u, dt) -> u.

    ``param`` is the per-sample physical parameter (D or v0) for the current
    batch; ``dt`` is supplied by the splitting scheme (full or half step).
    """
    def brick(u: torch.Tensor, dt) -> torch.Tensor:
        dt_col = torch.full((u.size(0),), float(dt), device=u.device, dtype=u.dtype)
        cond = torch.stack([dt_col, param.to(u.device).to(u.dtype)], dim=-1)
        return model(u, cond)
    return brick


# ---------------------------------------------------------------------------
# Dataset generation (fully batched; the simulators broadcast over samples)
# ---------------------------------------------------------------------------

def _draw(rng, lo, hi, n):
    return torch.tensor(rng.uniform(lo, hi, n), dtype=torch.float32).unsqueeze(-1)


def generate_brick_dataset(brick: str, n_samples: int, nx: int,
                           shear: float, profile: torch.Tensor,
                           dt_range: tuple = (0.005, 0.02),
                           seed: int = 0) -> TensorDataset:
    """Generate (u0, u1, cond) triples for one brick; cond = [dt, param]."""
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n_samples, nx, seed=seed)
    dt = _draw(rng, *dt_range, n_samples)
    if brick == "diffusion":
        D = _draw(rng, 0.01, 0.20, n_samples)
        u1 = brick_diffusion(u0, D=D, dt=dt)
        cond = torch.cat([dt, D], dim=-1)
    elif brick == "advection":
        v0 = _draw(rng, 0.10, 2.00, n_samples)
        u1 = brick_advection(u0, v0=v0, dt=dt, shear=shear, profile=profile)
        cond = torch.cat([dt, v0], dim=-1)
    else:
        raise ValueError(brick)
    return TensorDataset(u0, u1, cond)


def generate_joint_dataset(n_samples: int, nx: int, shear: float,
                           profile: torch.Tensor,
                           dt_range: tuple = (0.005, 0.02),
                           seed: int = 99) -> TensorDataset:
    """Generate (u0, u1, cond) triples for the monolithic FNO; cond = [dt, D, v0]."""
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n_samples, nx, seed=seed)
    D  = _draw(rng, 0.01, 0.20, n_samples)
    v0 = _draw(rng, 0.10, 2.00, n_samples)
    dt = _draw(rng, *dt_range, n_samples)
    u1 = simulate_full(u0, D=D, v0=v0, dt=dt, shear=shear, profile=profile)
    return TensorDataset(u0, u1, torch.cat([dt, D, v0], dim=-1))


def generate_test_dataset(n_samples: int, nx: int, shear: float,
                          profile: torch.Tensor, eval_dt: float,
                          seed: int = 777) -> TensorDataset:
    """Test set at fixed dt; keeps (u0, u1, D, v0) so per-brick conds can be built."""
    rng = np.random.default_rng(seed)
    u0 = batch_random_ics(n_samples, nx, seed=seed)
    D  = _draw(rng, 0.01, 0.20, n_samples)
    v0 = _draw(rng, 0.10, 2.00, n_samples)
    u1 = simulate_full(u0, D=D, v0=v0, dt=eval_dt, shear=shear, profile=profile)
    return TensorDataset(u0, u1, D.squeeze(-1), v0.squeeze(-1))


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train(model: nn.Module, dataset: TensorDataset,
          n_epochs: int = 200, batch_size: int = 64,
          lr: float = 1e-3, device: str = "cpu") -> list:
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_epochs):
        running = 0.0
        for u0, u1, cond in loader:
            u0, u1, cond = u0.to(device), u1.to(device), cond.to(device)
            loss = F.mse_loss(model(u0, cond), u1)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        losses.append(running / len(loader))
    return losses


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (
        (pred - target).norm(dim=-1) / (target.norm(dim=-1) + 1e-8)
    ).mean().item()


def evaluate_joint(model: nn.Module, test_ds: TensorDataset,
                   eval_dt: float, device: str) -> float:
    model.eval()
    loader = DataLoader(test_ds, batch_size=128)
    errors = []
    with torch.no_grad():
        for u0, u1, D, v0 in loader:
            u0, u1, D, v0 = u0.to(device), u1.to(device), D.to(device), v0.to(device)
            cond = torch.stack([torch.full_like(D, eval_dt), D, v0], dim=-1)
            errors.append(rel_l2(model(u0, cond), u1))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Data-efficiency gain
# ---------------------------------------------------------------------------

def data_efficiency_gain(budgets, joint_errs, composed_err, n_composed):
    """Genuine sample-efficiency gain.

    Returns (qualifier, gain) where gain = (joint samples needed to match the
    composed model's error) / (n_composed samples the composed model used).

    qualifier in {"=", ">=", "<"}: "=" if the crossing falls inside the budget
    grid (interpolated in log-log); ">=" if the joint never matches the composed
    accuracy within the grid (composition strictly more data-efficient; a lower
    bound); "<" if the joint already matches at the smallest budget tried.
    """
    b = np.asarray(budgets, dtype=float)
    e = np.minimum.accumulate(np.asarray(joint_errs, dtype=float))  # best error with <= b samples
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
