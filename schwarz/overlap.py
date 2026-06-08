"""Overlap-aware 1D Poisson Schwarz helpers (used by run_e4e.py).

Generalises ``schwarz/poisson.py`` to two-subdomain decompositions with an
arbitrary overlap width ``delta``.

Geometry (interface still centred at x = 0.5):

    left  subdomain  Omega_L = [0,            0.5 + delta/2]
    right subdomain  Omega_R = [0.5 - delta/2, 1.0]
    overlap region   Omega_o = [0.5 - delta/2, 0.5 + delta/2]   (width delta)

For ``delta = 0`` this reduces to the non-overlapping split used in E4.

Each half problem is solved by second-order finite differences on
``nx_half`` interior points; the grid spacing therefore depends on the
half-width ``H_L = 0.5 + delta/2`` (left) and ``H_R = 0.5 + delta/2`` (right).

The classical Schwarz iteration is:

    u_L^{k+1} solves -u'' = f on Omega_L,  u(0)=0,           u(beta_L)=g_L^k
    u_R^{k+1} solves -u'' = f on Omega_R,  u(alpha_R)=g_R^k, u(1)=0
    g_L^{k+1} <- u_R^{k+1}(beta_L)
    g_R^{k+1} <- u_L^{k+1}(alpha_R)

The reference assembled solution on the full domain partitions Omega_o by
weighted averaging (linear partition of unity).
"""

from __future__ import annotations

import numpy as np

from schwarz.poisson import _tridiag_solve


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def subdomain_bounds(delta: float) -> tuple[float, float, float, float]:
    """Return (a_L, b_L, a_R, b_R) for the two overlap-extended subdomains."""
    a_L = 0.0
    b_L = 0.5 + 0.5 * delta
    a_R = 0.5 - 0.5 * delta
    b_R = 1.0
    return a_L, b_L, a_R, b_R


def interior_grid(a: float, b: float, nx_half: int) -> tuple[np.ndarray, float]:
    """Interior grid on (a, b) with ``nx_half`` points and Dirichlet ends."""
    h = (b - a) / (nx_half + 1)
    x = np.linspace(a + h, b - h, nx_half)
    return x, float(h)


# ---------------------------------------------------------------------------
# Half-domain Poisson solver with arbitrary endpoints
# ---------------------------------------------------------------------------

def poisson_half_overlap(f_vals: np.ndarray, g_left: float, g_right: float,
                         a: float, b: float,
                         nx_half: int) -> tuple[np.ndarray, float, float]:
    """Solve ``-u'' = f`` on ``[a, b]`` with Dirichlet ``u(a)=g_left``,
    ``u(b)=g_right`` on ``nx_half`` interior points.

    Returns ``(u_interior, u_left_interior, u_right_interior)`` where the
    last two scalars are the values right next to the endpoints; they are
    used by the Schwarz iteration to read off the new interface data.
    """
    h = (b - a) / (nx_half + 1)
    rhs = f_vals.copy().astype(float)
    rhs[0]  += g_left  / h**2
    rhs[-1] += g_right / h**2
    u = _tridiag_solve(nx_half, rhs, h)
    return u, float(u[0]), float(u[-1])


# ---------------------------------------------------------------------------
# Sampling f on a subdomain from a globally-defined random RHS
# ---------------------------------------------------------------------------

def sample_f_on_grid(seed: int, n_modes: int, x: np.ndarray) -> np.ndarray:
    """Evaluate a smooth sine-series RHS on an arbitrary grid ``x``.

    The user-supplied grid lets the left and right subdomains share a single
    right-hand side ``f``.
    """
    rng = np.random.default_rng(seed)
    f = np.zeros_like(x, dtype=float)
    for _ in range(n_modes):
        k = rng.integers(1, 8)
        amp = rng.standard_normal()
        f += amp * np.sin(k * np.pi * x)
    return f.astype(np.float32)


# ---------------------------------------------------------------------------
# Reference full-domain Poisson solution (same as poisson.py but exposed)
# ---------------------------------------------------------------------------

def full_solution(nx_full: int, f_full: np.ndarray) -> np.ndarray:
    h = 1.0 / (nx_full + 1)
    return _tridiag_solve(nx_full, f_full.astype(float), h)


def full_grid(nx_full: int) -> np.ndarray:
    h = 1.0 / (nx_full + 1)
    return np.linspace(h, 1.0 - h, nx_full)


# ---------------------------------------------------------------------------
# Pairing the same f across all overlap widths
# ---------------------------------------------------------------------------

def build_sample(seed: int, delta: float, nx_half: int, nx_full: int,
                 n_modes: int = 6) -> dict:
    """Build one (delta-specific) sample: f sampled coherently on the global
    fine grid and re-evaluated on each subdomain interior grid.

    Returns a dict with the per-sample tensors needed both for training and
    for evaluation. The reference solution is computed on the global grid;
    g_L_true and g_R_true are read from this reference at the two interface
    locations ``b_L`` and ``a_R``.
    """
    a_L, b_L, a_R, b_R = subdomain_bounds(delta)

    # Global reference (always nx_full points on [0, 1]).
    x_full = full_grid(nx_full)
    f_full = sample_f_on_grid(seed, n_modes, x_full)
    u_full = full_solution(nx_full, f_full)

    # Subdomain grids and RHS values.
    x_L, _ = interior_grid(a_L, b_L, nx_half)
    x_R, _ = interior_grid(a_R, b_R, nx_half)
    f_L = sample_f_on_grid(seed, n_modes, x_L)
    f_R = sample_f_on_grid(seed, n_modes, x_R)

    # True interface Dirichlet data, by linear interpolation from the global
    # reference grid.
    g_L_true = float(np.interp(b_L, x_full, u_full))
    g_R_true = float(np.interp(a_R, x_full, u_full))

    # Per-subdomain reference solutions at the true g_L, g_R values: these
    # are the training targets.
    u_L_ref, _, _ = poisson_half_overlap(f_L, 0.0,      g_L_true, a_L, b_L, nx_half)
    u_R_ref, _, _ = poisson_half_overlap(f_R, g_R_true, 0.0,      a_R, b_R, nx_half)

    return {
        "f_L":      f_L,
        "f_R":      f_R,
        "g_L":      np.float32(g_L_true),
        "g_R":      np.float32(g_R_true),
        "u_L":      u_L_ref.astype(np.float32),
        "u_R":      u_R_ref.astype(np.float32),
        "f_full":   f_full.astype(np.float32),
        "u_full":   u_full.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Reference Schwarz iteration (FD subdomain solver, classical g <- u update)
# ---------------------------------------------------------------------------

def schwarz_reference(f_L: np.ndarray, f_R: np.ndarray,
                      delta: float, nx_half: int,
                      max_iter: int = 60, tol: float = 1e-10
                      ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Classical alternating Schwarz with FD subdomain solves.

    Returns ``(u_L, u_R, history)`` where ``history`` is the per-iteration
    interface update magnitude ``|(g_L_new, g_R_new) - (g_L, g_R)|`` (max).
    """
    a_L, b_L, a_R, b_R = subdomain_bounds(delta)
    x_L, _ = interior_grid(a_L, b_L, nx_half)
    x_R, _ = interior_grid(a_R, b_R, nx_half)
    g_L = 0.0
    g_R = 0.0
    history: list[float] = []
    u_L = u_R = np.zeros(nx_half)
    for _ in range(max_iter):
        u_L, _, u_L_at_bL = poisson_half_overlap(f_L, 0.0, g_L, a_L, b_L, nx_half)
        u_R, u_R_at_aR, _ = poisson_half_overlap(f_R, g_R, 0.0, a_R, b_R, nx_half)
        # New Dirichlet data is the other subdomain's value at the foreign endpoint.
        g_L_new = float(np.interp(b_L, x_R, u_R))
        g_R_new = float(np.interp(a_R, x_L, u_L))
        delta_g = max(abs(g_L_new - g_L), abs(g_R_new - g_R))
        history.append(delta_g)
        g_L, g_R = g_L_new, g_R_new
        if delta_g < tol:
            break
    return u_L, u_R, history


# ---------------------------------------------------------------------------
# Contraction rate from a convergence history
# ---------------------------------------------------------------------------

def contraction_rate(history: list[float]) -> float:
    """Estimate one-step contraction rate as the geometric mean of
    successive ratios ``history[k+1] / history[k]`` over the iterations
    where both endpoints are positive.
    """
    h = np.asarray(history, dtype=float)
    if h.size < 2:
        return float("nan")
    ratios = []
    for k in range(h.size - 1):
        if h[k] > 1e-15 and h[k + 1] > 1e-15:
            ratios.append(h[k + 1] / h[k])
    if not ratios:
        return float("nan")
    return float(np.exp(np.mean(np.log(ratios))))


__all__ = [
    "subdomain_bounds", "interior_grid",
    "poisson_half_overlap", "sample_f_on_grid",
    "full_solution", "full_grid",
    "build_sample", "schwarz_reference", "contraction_rate",
]
