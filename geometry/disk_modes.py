"""Bessel-mode utilities for the disk Dirichlet heat equation.

Closed-form Dirichlet eigenpairs on the disk of radius R, in polar coords:

  phi_{m, n}(r, theta) = J_m(z_{m, n} r / R) * (cos(m theta) or sin(m theta))
  lambda_{m, n}         = (z_{m, n} / R)^2

with z_{m, n} the n-th positive zero of the Bessel function J_m. The functions
below build a discrete Bessel expansion / inversion pair on a uniform Cartesian
grid covering the bounding box [-R, R]^2. Outside the disk the field is set to
zero (Dirichlet condition on the disk boundary).

The transform is not orthonormal in the discrete inner product since we sample
on a Cartesian grid that does not respect the polar measure r dr dtheta. We
therefore implement the forward step as a least-squares projection (lstsq) onto
the truncated Bessel basis, and the backward step as a plain linear combination
of basis functions. For a smooth Dirichlet IC the truncation error of the basis
dominates, which is exactly the regime the LB-FNO is meant to exploit.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jn, jn_zeros


# ---------------------------------------------------------------------------
# Bessel-zero cache (jn_zeros is non-trivial; we cache per (m, n) pair)
# ---------------------------------------------------------------------------

_ZERO_CACHE: dict[int, np.ndarray] = {}


def bessel_zeros(m: int, n_zeros: int) -> np.ndarray:
    """First ``n_zeros`` positive zeros of J_m, cached across calls."""
    cached = _ZERO_CACHE.get(m)
    if cached is not None and len(cached) >= n_zeros:
        return cached[:n_zeros]
    zeros = jn_zeros(m, n_zeros).astype(float)
    _ZERO_CACHE[m] = zeros
    return zeros


# ---------------------------------------------------------------------------
# Mode enumeration: pick the K smallest eigenvalues for a disk of radius R
# ---------------------------------------------------------------------------

def disk_mode_list(R: float, K: int,
                   max_order: int | None = None,
                   n_zeros: int | None = None
                   ) -> list[tuple[int, int, str, float, float]]:
    """Return the K smallest disk eigenmodes as a list of tuples.

    Each entry is ``(m, n, parity, z_mn, lambda_mn)`` with parity in
    {"c", "s"} indicating cosine or sine angular dependence. For m = 0 only
    the cosine entry is present (the sine mode is identically zero).
    """
    if max_order is None:
        max_order = max(int(np.ceil(np.sqrt(K))) + 2, 4)
    if n_zeros is None:
        n_zeros = max(K, 8)
    modes: list[tuple[int, int, str, float, float]] = []
    for m in range(0, max_order + 1):
        zeros = bessel_zeros(m, n_zeros)
        for n_idx, z in enumerate(zeros, start=1):
            lam = (z / R) ** 2
            modes.append((m, n_idx, "c", float(z), float(lam)))
            if m > 0:
                modes.append((m, n_idx, "s", float(z), float(lam)))
    modes.sort(key=lambda t: t[4])
    return modes[:K]


# ---------------------------------------------------------------------------
# Disk mask and grid coordinates for the bounding box [-R, R]^2
# ---------------------------------------------------------------------------

def disk_grid(R: float, nx: int):
    """Return Cartesian grid (X, Y), polar (rgrid, theta), and disk mask.

    The grid covers [-R, R] x [-R, R] with ``nx`` points per axis (so nx*nx
    total). The mask is True strictly inside the open disk r < R; points on
    the boundary are excluded so Dirichlet BC is enforced by zeroing.
    """
    coords = np.linspace(-R, R, nx, dtype=np.float64)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    rgrid = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(Y, X)
    mask = rgrid < R
    return X, Y, rgrid, theta, mask


def basis_matrix(R: float, nx: int, modes) -> tuple[np.ndarray, np.ndarray]:
    """Build the (n_inside, K) basis matrix of Bessel-mode values on the grid.

    Returns ``(B, mask_flat)`` where ``B`` is the matrix whose columns are the
    K basis functions evaluated at the interior grid points, and ``mask_flat``
    is the boolean array selecting those points out of the full ``nx * nx``
    grid (row-major flat indexing).
    """
    _, _, rgrid, theta, mask = disk_grid(R, nx)
    mask_flat = mask.reshape(-1)
    r_in = rgrid[mask].astype(np.float64)
    th_in = theta[mask].astype(np.float64)
    K = len(modes)
    B = np.zeros((r_in.size, K), dtype=np.float64)
    for k, (m, _n, parity, z, _lam) in enumerate(modes):
        radial = jn(m, z * r_in / R)
        if m == 0:
            angular = np.ones_like(th_in)
        elif parity == "c":
            angular = np.cos(m * th_in)
        else:
            angular = np.sin(m * th_in)
        B[:, k] = radial * angular
    return B, mask_flat


# ---------------------------------------------------------------------------
# Random initial condition on the disk: smooth Dirichlet sample
# ---------------------------------------------------------------------------

def random_ic_disk(R: float, nx: int, n_modes: int = 5,
                   seed: int = 0,
                   k_max: int = 4) -> np.ndarray:
    """Smooth Dirichlet IC on the disk of radius R sampled on the bounding box.

    The IC is built directly from a few low Bessel modes so the Dirichlet BC
    holds exactly on the disk boundary. Values are rescaled to the range
    (-0.4, 0.4). Outside the disk the field is zero.
    """
    rng = np.random.default_rng(seed)
    _, _, rgrid, theta, mask = disk_grid(R, nx)
    u = np.zeros_like(rgrid)
    # Draw n_modes random (m, n, parity) triples from the lowest k_max^2 modes.
    candidates = disk_mode_list(R, K=k_max * k_max)
    if not candidates:
        return u.astype(np.float32)
    idx = rng.integers(0, len(candidates), size=n_modes)
    for j in idx:
        m, _n_idx, parity, z, _lam = candidates[int(j)]
        c = float(rng.standard_normal()) * 0.5
        radial = jn(m, z * rgrid / R)
        if m == 0:
            angular = np.ones_like(theta)
        elif parity == "c":
            angular = np.cos(m * theta)
        else:
            angular = np.sin(m * theta)
        u += c * radial * angular
    u[~mask] = 0.0
    peak = float(np.abs(u).max())
    if peak > 1e-8:
        u *= 0.4 / peak
    return u.astype(np.float32)


# ---------------------------------------------------------------------------
# Closed-form heat semigroup on the disk via the truncated Bessel basis
# ---------------------------------------------------------------------------

def heat_dirichlet_disk(u0: np.ndarray, R: float, D: float, dt: float,
                        K: int = 24, modes=None,
                        basis_cache: dict | None = None) -> np.ndarray:
    """Closed-form heat evolution on a disk of radius R via Bessel expansion.

    Steps:
      1. Project u0 onto the truncated Bessel basis (least squares).
      2. Decay each coefficient by exp(-D lambda_k dt).
      3. Sum the decayed basis to recover u(dt).

    Setting ``modes`` to ``None`` triggers a fresh ``disk_mode_list(R, K)``
    call. The optional ``basis_cache`` dict (keyed by (R, nx, K)) saves a
    pinv computation, useful when reusing the same R across many evaluations.
    """
    nx = u0.shape[0]
    assert u0.shape == (nx, nx), "u0 must be square (nx, nx)"
    if modes is None:
        modes = disk_mode_list(R, K)
    key = (float(R), int(nx), int(K))
    if basis_cache is not None and key in basis_cache:
        B, mask_flat, pinv = basis_cache[key]
    else:
        B, mask_flat = basis_matrix(R, nx, modes)
        pinv = np.linalg.pinv(B)
        if basis_cache is not None:
            basis_cache[key] = (B, mask_flat, pinv)
    u_in = u0.reshape(-1)[mask_flat]
    coeffs = pinv @ u_in
    lam = np.array([lm for (_m, _n, _p, _z, lm) in modes], dtype=np.float64)
    coeffs = coeffs * np.exp(-D * lam * dt)
    rebuilt = B @ coeffs
    out = np.zeros(nx * nx, dtype=np.float64)
    out[mask_flat] = rebuilt
    return out.reshape(nx, nx).astype(np.float32)
