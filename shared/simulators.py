"""Linear PDE simulators for the operator-splitting experiment (E2, Pillar 1).

Two elementary **linear** generators on the periodic domain [0, 1]:
  A (diffusion):  ∂_t u = D ∂_xx u                     [exact: heat kernel]
  B (advection):  ∂_t u = −v(x) ∂_x u                  [linear transport]

The velocity field is
    v(x) = v0 · (1 + s · p(x)),
where p(x) is a fixed zero-mean, unit-amplitude smooth profile and s ≥ 0 is the
*shear amplitude* that controls the commutator norm.  Following the
advection–diffusion commutator of Appendix B of the proposal,

    [A, B] = D (∇·v)(∇·) + lower order   (in 1D: [A, B] = −2D v'(x) ∂_xx − D v''(x) ∂_x),

the commutator vanishes for constant velocity (s = 0 ⇒ Lie–Trotter exact) and
grows with the velocity gradient, i.e. with s.  This replaces the earlier
nonlinear Fisher–KPP reaction: keeping both generators linear means the
Baker–Campbell–Hausdorff theory of Pillar 1 applies directly, and the experiment
does not anticipate the harder nonlinear semigroup theory (for which no clean
Strang/BCH commutator expansion exists).

Full PDE (ground truth for the joint baseline):
  ∂_t u = D ∂_xx u − v(x) ∂_x u                        [linear]

Numerics.  Diffusion is evaluated exactly via the heat kernel.  For constant
velocity (s = 0) both the advection brick and the joint problem are
constant-coefficient and are evaluated *exactly* in Fourier space.  For s > 0 the
variable-coefficient advection brick and the joint PDE are integrated with a
dealiased pseudo-spectral RK4 scheme, sub-stepped finely enough that the
ground-truth error is far below the FNO approximation error.

All functions accept torch.Tensor of shape (..., nx); scalar PDE parameters may
be passed as floats or as (..., 1) tensors for fully batched data generation.
"""

import math
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rfft_wavenumbers(nx: int, device) -> torch.Tensor:
    """Real-FFT wavenumbers k for a domain of length 1."""
    return torch.fft.rfftfreq(nx, d=1.0 / nx).to(device)


def _rk4(rhs, u: torch.Tensor, dt) -> torch.Tensor:
    """One classical RK4 step. dt may be a scalar or a (..., 1) tensor."""
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _evolve_pseudospectral(u: torch.Tensor, D, v_field: torch.Tensor,
                           dt, n_substeps: int) -> torch.Tensor:
    """Integrate ∂_t u = D ∂_xx u − v(x) ∂_x u with dealiased pseudo-spectral RK4.

    Spatial derivatives are spectral; products are de-aliased with the 2/3 rule.
    Used only for the variable-velocity (s > 0) case; D = 0 gives pure advection.
    """
    nx = u.shape[-1]
    k = _rfft_wavenumbers(nx, u.device)
    twopik = 2.0 * math.pi * k
    mask = (k <= (nx // 3)).to(u.dtype)          # 2/3 dealiasing mask on rfft modes
    h = dt / n_substeps

    def rhs(w):
        wf = torch.fft.rfft(w, dim=-1)
        ux = torch.fft.irfft(1j * twopik * wf * mask, n=nx, dim=-1)
        uxx = torch.fft.irfft(-(twopik ** 2) * wf * mask, n=nx, dim=-1)
        out = D * uxx - v_field * ux
        return torch.fft.irfft(torch.fft.rfft(out, dim=-1) * mask, n=nx, dim=-1)

    for _ in range(n_substeps):
        u = _rk4(rhs, u, h)
    return u


def velocity_field(v0, shear: float, profile: torch.Tensor) -> torch.Tensor:
    """v(x) = v0 · (1 + shear · p(x)).  v0 may be scalar or (..., 1)."""
    return v0 * (1.0 + shear * profile)


def shear_profile(nx: int = 128, n_modes: int = 3, seed: int = 0,
                  device: str = "cpu") -> torch.Tensor:
    """Fixed zero-mean, unit-max-abs smooth velocity profile p(x) on [0, 1).

    A seeded sum of the first ``n_modes`` Fourier harmonics; deterministic for a
    given (nx, n_modes, seed).  Used to give the velocity field a non-trivial
    spatial gradient so that the diffusion/advection commutator is non-zero.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    p = np.zeros(nx)
    for kf in range(1, n_modes + 1):
        a, b = rng.standard_normal(2)
        p += a * np.sin(2.0 * np.pi * kf * x) + b * np.cos(2.0 * np.pi * kf * x)
    p -= p.mean()
    p /= (np.abs(p).max() + 1e-12)
    return torch.tensor(p, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Brick A: pure diffusion (exact)
# ---------------------------------------------------------------------------

def brick_diffusion(u: torch.Tensor, D, dt) -> torch.Tensor:
    """Exact solution of ∂_t u = D ∂_xx u for time dt.

    u(t + dt) = IFFT( exp(−D (2πk)² dt) · FFT(u) ).
    D and dt may be scalars or (..., 1) tensors (batched data generation).
    """
    nx = u.shape[-1]
    k = _rfft_wavenumbers(nx, u.device)
    decay = torch.exp(-((2.0 * math.pi * k) ** 2) * D * dt)
    return torch.fft.irfft(decay * torch.fft.rfft(u, dim=-1), n=nx, dim=-1)


# ---------------------------------------------------------------------------
# Brick B: linear advection (constant or variable velocity)
# ---------------------------------------------------------------------------

def brick_advection(u: torch.Tensor, v0, dt, shear: float = 0.0,
                    profile: torch.Tensor = None,
                    n_substeps: int = 64) -> torch.Tensor:
    """Solution of ∂_t u = −v(x) ∂_x u for time dt, with v(x)=v0·(1+shear·p(x)).

    For shear = 0 (constant velocity) this is the exact spectral shift
    u(t+dt)(x) = u(x − v0 dt), so [A, B] = 0 and Lie–Trotter is exact.
    For shear > 0 the variable-coefficient transport is integrated with a
    dealiased pseudo-spectral RK4 scheme.
    """
    nx = u.shape[-1]
    k = _rfft_wavenumbers(nx, u.device)
    if shear == 0.0 or profile is None:
        phase = torch.exp(-1j * 2.0 * math.pi * k * v0 * dt)
        return torch.fft.irfft(phase * torch.fft.rfft(u, dim=-1), n=nx, dim=-1)
    return _evolve_pseudospectral(u, 0.0, velocity_field(v0, shear, profile),
                                  dt, n_substeps)


# ---------------------------------------------------------------------------
# Brick C: cubic reaction (pointwise, exact)
# ---------------------------------------------------------------------------

def brick_reaction(u: torch.Tensor, gamma, dt) -> torch.Tensor:
    """Exact pointwise solution of the cubic reaction ODE.

    The reaction is ∂_t u = -gamma · u · (1 - u^2), the Allen-Cahn reaction
    term (separable from the spatial operators because it is purely local).
    Writing g = gamma, the ODE du/dt = g u (u^2 - 1) is separable; integrating
    via partial fractions yields the closed form

        u(t + dt)^2 = u(t)^2 / (u(t)^2 + (1 - u(t)^2) · exp(2 g dt))

    with the sign of u preserved (the ODE has u = 0 as a stable fixed point for
    g > 0 and |u| < 1). gamma and dt may be scalars or broadcastable tensors.
    """
    u2 = u * u
    exp2gdt = torch.exp(2.0 * gamma * dt)
    denom = u2 + (1.0 - u2) * exp2gdt
    new_u2 = u2 / (denom + 1e-20)
    return torch.sign(u) * torch.sqrt(torch.clamp(new_u2, min=0.0))


# ---------------------------------------------------------------------------
# Full PDE (ground truth for joint baseline)
# ---------------------------------------------------------------------------

def simulate_full(u: torch.Tensor, D, v0, dt, shear: float = 0.0,
                  profile: torch.Tensor = None,
                  n_substeps: int = 160) -> torch.Tensor:
    """Advance ∂_t u = D ∂_xx u − v(x) ∂_x u by time dt.

    For shear = 0 the operator is constant-coefficient and is solved exactly in
    Fourier space; for shear > 0 a dealiased pseudo-spectral RK4 scheme is used
    (more sub-steps than the advection brick, to absorb the diffusion stiffness).
    """
    nx = u.shape[-1]
    k = _rfft_wavenumbers(nx, u.device)
    twopik = 2.0 * math.pi * k
    if shear == 0.0 or profile is None:
        symbol = torch.exp((-(twopik ** 2) * D - 1j * twopik * v0) * dt)
        return torch.fft.irfft(symbol * torch.fft.rfft(u, dim=-1), n=nx, dim=-1)
    return _evolve_pseudospectral(u, D, velocity_field(v0, shear, profile),
                                  dt, n_substeps)


def simulate_full_reaction(u: torch.Tensor, D, v0, gamma, dt,
                           shear: float = 0.0,
                           profile: torch.Tensor = None,
                           n_substeps: int = 160) -> torch.Tensor:
    """Advance Allen-Cahn-with-drift: ∂_t u = D ∂_xx u - v(x) ∂_x u - gamma u(1 - u^2).

    Integrated with a dealiased pseudo-spectral RK4 scheme for the
    advection-diffusion-reaction system. gamma may be a scalar or a (..., 1)
    tensor; the reaction term is local so it broadcasts trivially.
    """
    nx = u.shape[-1]
    k = _rfft_wavenumbers(nx, u.device)
    twopik = 2.0 * math.pi * k
    mask = (k <= (nx // 3)).to(u.dtype)
    if shear == 0.0 or profile is None:
        v_field = v0 * torch.ones_like(u)
    else:
        v_field = velocity_field(v0, shear, profile)
    h = dt / n_substeps

    def rhs(w):
        wf = torch.fft.rfft(w, dim=-1)
        ux = torch.fft.irfft(1j * twopik * wf * mask, n=nx, dim=-1)
        uxx = torch.fft.irfft(-(twopik ** 2) * wf * mask, n=nx, dim=-1)
        out = D * uxx - v_field * ux - gamma * w * (1.0 - w * w)
        return torch.fft.irfft(torch.fft.rfft(out, dim=-1) * mask, n=nx, dim=-1)

    for _ in range(n_substeps):
        u = _rk4(rhs, u, h)
    return u


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def random_ic(nx: int = 128, n_modes: int = 5,
              seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """Smooth, zero-mean random IC on [0, 1) (band-limited to a few modes)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    u = np.zeros(nx)
    for _ in range(n_modes):
        freq = rng.integers(1, 6)
        a, b = rng.standard_normal(2) * 0.5
        u += a * np.sin(2.0 * np.pi * freq * x) + b * np.cos(2.0 * np.pi * freq * x)
    u = 0.4 * u / (np.abs(u).max() + 1e-8)
    return torch.tensor(u, dtype=torch.float32, device=device)


def batch_random_ics(n: int, nx: int = 128,
                     seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """Return (n, nx) tensor of independent random ICs."""
    return torch.stack([
        random_ic(nx, seed=seed * 10_000 + i, device=device)
        for i in range(n)
    ])
