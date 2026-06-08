"""Reaction brick and full semilinear-PDE reference for experiment E2e.

Two ingredients are provided here so the E2e script can stay focused on the
training and evaluation logic:

  1. ``brick_reaction(u, gamma, dt)``  closed-form solution of the pointwise
     cubic ODE  ``du/dt = -gamma u (1 - u^2)``.
  2. ``simulate_semilinear(u, D, gamma, dt, n_substeps)``  high-accuracy
     reference solution of the full semilinear PDE
     ``du/dt = D u_xx - gamma u (1 - u^2)``  using a Strang-substepped
     pseudo-spectral integrator (exact diffusion in Fourier, exact reaction
     pointwise).

Both functions accept ``(B, nx)`` tensors with scalar or per-sample
``(B, 1)`` parameters, matching the conventions of ``shared.simulators``.
"""

from __future__ import annotations

import math

import torch

from shared.simulators import brick_diffusion


# ---------------------------------------------------------------------------
# Reaction brick: closed-form pointwise ODE
# ---------------------------------------------------------------------------

def brick_reaction(u: torch.Tensor, gamma, dt) -> torch.Tensor:
    """Exact solution of  ``du/dt = -gamma u (1 - u^2)``  per grid point.

    Setting  ``w = 1/u^2``  turns the cubic ODE into the linear ODE
    ``dw/dt = 2 gamma (w - 1)``  with solution
    ``w(t) = 1 + (1/u0^2 - 1) exp(2 gamma t)``,
    so ``u(t)^2 = 1 / (1 + (1/u0^2 - 1) exp(2 gamma t))``,
    with the sign of ``u`` preserved from the initial datum.

    Parameters ``gamma`` and ``dt`` may be Python floats or broadcastable
    ``(B, 1)`` tensors so the brick can be evaluated in fully batched fashion.
    """
    eps = 1e-12
    u2 = u * u + eps
    factor = torch.exp(2.0 * gamma * dt)
    new_inv = 1.0 + (1.0 / u2 - 1.0) * factor
    new_u2 = 1.0 / (new_inv + eps)
    return torch.sign(u) * torch.sqrt(torch.clamp(new_u2, min=0.0))


# ---------------------------------------------------------------------------
# Full semilinear PDE reference (Strang-substepped)
# ---------------------------------------------------------------------------

def simulate_semilinear(u: torch.Tensor, D, gamma, dt,
                        n_substeps: int = 32) -> torch.Tensor:
    """Reference solution of  ``du/dt = D u_xx - gamma u (1 - u^2)``.

    Each sub-step applies one Strang step of  ``[diffusion, reaction]``.
    Both bricks are exact in their own right (heat kernel and pointwise cubic
    closed form), so the only error is the per-substep splitting truncation
    ``O((dt/n_substeps)^3)``; ``n_substeps = 32`` makes that error negligible
    compared with the FNO approximation error at the test budgets we use.
    """
    h = dt / n_substeps
    for _ in range(n_substeps):
        u = brick_diffusion(u, D=D, dt=h / 2.0)
        u = brick_reaction(u, gamma=gamma, dt=h)
        u = brick_diffusion(u, D=D, dt=h / 2.0)
    return u
