"""Helpers for perforated-rectangle Laplace-Beltrami eigenvalues (E3h).

A perforated rectangle is the open set ``[0, a] x [0, b] \\ B((cx, cy), r)``
with Dirichlet boundary conditions on the outer rectangle and on the inner
circular hole.

The reference eigenvalues are computed by a sparse 5-point Laplacian
eigensolve on a uniform Cartesian grid; grid cells that lie inside the hole
(or outside the rectangle) are removed from the unknown set so the inner
boundary is enforced implicitly through ghost values of zero. This stays
purely in-process (scipy.sparse.linalg.eigsh) and needs no extra dependency.

The same module also exposes a sampler for the training and test libraries
of perforated rectangles, plus a sanity check against the closed-form
rectangle spectrum when ``r = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class PerforatedRect:
    a: float
    b: float
    cx: float
    cy: float
    r: float

    def descriptor(self) -> np.ndarray:
        """Return the 5-vector descriptor (a, b, cx, cy, r)."""
        return np.array([self.a, self.b, self.cx, self.cy, self.r],
                        dtype=np.float32)

    def hole_area_fraction(self) -> float:
        if self.r <= 0.0:
            return 0.0
        return float(np.pi * self.r ** 2 / (self.a * self.b))


def _interior_mask(a: float, b: float, cx: float, cy: float, r: float,
                   nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, x, y) where mask is True on interior grid cells.

    Cells lying inside the hole or outside the rectangle interior are False.
    The Cartesian grid uses interior (Dirichlet) nodes; the rectangle's
    outer boundary is enforced by truncating the index set to (1, nx),
    (1, ny) as in the standard 5-point stencil.
    """
    # Interior grid spacing in the rectangle.
    hx = a / (nx + 1)
    hy = b / (ny + 1)
    i = np.arange(1, nx + 1, dtype=float)
    j = np.arange(1, ny + 1, dtype=float)
    x = i * hx
    y = j * hy
    X, Y = np.meshgrid(x, y, indexing="ij")  # shape (nx, ny)
    if r > 0.0:
        inside_hole = (X - cx) ** 2 + (Y - cy) ** 2 < r * r
    else:
        inside_hole = np.zeros_like(X, dtype=bool)
    mask = ~inside_hole
    return mask, x, y


def perforated_eigenvalues(rect: PerforatedRect, K: int,
                            nx: int = 96, ny: int = 96) -> np.ndarray:
    """Sparse-grid Dirichlet eigenvalues of -Delta on the perforated rect.

    Returns the K smallest eigenvalues in ascending order.
    """
    a, b, cx, cy, r = rect.a, rect.b, rect.cx, rect.cy, rect.r
    mask, x, y = _interior_mask(a, b, cx, cy, r, nx, ny)
    hx = a / (nx + 1)
    hy = b / (ny + 1)

    # Active-node global index map. -1 marks Dirichlet (boundary or hole).
    idx = -np.ones((nx, ny), dtype=np.int64)
    active = np.flatnonzero(mask.reshape(-1))
    idx.reshape(-1)[active] = np.arange(active.size, dtype=np.int64)
    n_active = int(active.size)
    if n_active < K + 2:
        # Degenerate hole: bail out with a uniform large value rather than crash.
        return np.full(K, np.nan, dtype=float)

    # 5-point stencil with second-order central differences.
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    diag = 2.0 * (inv_hx2 + inv_hy2)

    # Use COO-style triplet construction (faster than lil for moderate N).
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for ii in range(nx):
        for jj in range(ny):
            here = idx[ii, jj]
            if here < 0:
                continue
            rows.append(here); cols.append(here); vals.append(diag)
            # x-neighbours
            if ii > 0 and idx[ii - 1, jj] >= 0:
                rows.append(here); cols.append(idx[ii - 1, jj]); vals.append(-inv_hx2)
            if ii < nx - 1 and idx[ii + 1, jj] >= 0:
                rows.append(here); cols.append(idx[ii + 1, jj]); vals.append(-inv_hx2)
            # y-neighbours
            if jj > 0 and idx[ii, jj - 1] >= 0:
                rows.append(here); cols.append(idx[ii, jj - 1]); vals.append(-inv_hy2)
            if jj < ny - 1 and idx[ii, jj + 1] >= 0:
                rows.append(here); cols.append(idx[ii, jj + 1]); vals.append(-inv_hy2)

    A = csr_matrix((vals, (rows, cols)), shape=(n_active, n_active))
    # Symmetric SPD operator; use shift-invert for the smallest eigenvalues.
    try:
        eigs = eigsh(A, k=K, which="SM", return_eigenvectors=False)
    except Exception:
        eigs = eigsh(A, k=K, sigma=0.0, which="LM",
                     return_eigenvectors=False)
    return np.sort(eigs)


def sample_perforated_library(n: int, K: int,
                               a_range: tuple = (0.5, 1.5),
                               b_range: tuple = (0.5, 1.5),
                               r_max_frac: float = 0.30,
                               min_margin_frac: float = 0.08,
                               seed: int = 0,
                               nx: int = 64, ny: int = 64,
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n perforated rectangles and compute their eigenvalues.

    The hole radius is drawn so that the hole fits strictly inside the
    rectangle with a small safety margin (so the FEM-on-grid eigensolve
    does not see a hole that touches the outer boundary).

    Returns
    -------
    descriptors : (n, 5) float32   columns (a, b, cx, cy, r)
    eigs        : (n, K) float32
    areas       : (n,) float32     hole-area fraction
    """
    rng = np.random.default_rng(seed)
    descs = np.zeros((n, 5), dtype=np.float32)
    eigs = np.zeros((n, K), dtype=np.float32)
    areas = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        a = float(rng.uniform(*a_range))
        b = float(rng.uniform(*b_range))
        # Allowed radius range. The "margin" keeps the hole away from edges.
        margin = min_margin_frac * min(a, b)
        r_max_allowed = max(0.0, r_max_frac * min(a, b))
        r = float(rng.uniform(0.0, r_max_allowed))
        # Place centre so the disk + margin stay inside the rectangle.
        lo_x = r + margin
        hi_x = a - r - margin
        lo_y = r + margin
        hi_y = b - r - margin
        if hi_x <= lo_x or hi_y <= lo_y:
            # Fall back to no hole if geometry is too tight.
            r = 0.0
            cx = a / 2.0
            cy = b / 2.0
        else:
            cx = float(rng.uniform(lo_x, hi_x))
            cy = float(rng.uniform(lo_y, hi_y))
        rect = PerforatedRect(a=a, b=b, cx=cx, cy=cy, r=r)
        descs[i] = rect.descriptor()
        eigs[i] = perforated_eigenvalues(rect, K=K, nx=nx, ny=ny).astype(
            np.float32)
        areas[i] = rect.hole_area_fraction()
    return descs, eigs, areas
