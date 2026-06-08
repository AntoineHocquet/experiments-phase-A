"""FEM-on-grid Dirichlet eigensolver for right triangles.

We parametrise the right triangle with legs ``(L, H)`` where ``L`` is the base
length and ``H = L * tan(alpha)`` is the height; ``alpha`` is the apex angle
opposite the base. The right angle sits at the origin, the base lies along the
x-axis from ``(0, 0)`` to ``(L, 0)`` and the apex sits at ``(0, H)``. The
hypotenuse is the line ``x / L + y / H = 1``.

Reference eigenvalues are computed by a five-point Laplacian on a regular grid
covering the bounding rectangle ``[0, L] x [0, H]``, with the rectangular
nodes that fall outside the triangle masked out (Dirichlet BC are enforced by
not assembling those rows / columns). This is the same masked-grid trick used
as a FEM stand-in for E3d: it gives the
correct ordering and scaling of the first few eigenvalues at modest grid
resolution, without pulling in new dependencies.

The eigensolve is done with ``scipy.sparse.linalg.eigsh`` (smallest magnitude
mode) on the assembled sparse matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


def triangle_mask(nx: int, ny: int) -> np.ndarray:
    """Return a boolean mask of shape (nx, ny) for the interior of the right
    triangle with legs along the axes; True means "inside (strictly)".

    The grid covers the bounding rectangle with ``nx`` interior columns and
    ``ny`` interior rows. A grid point with normalised coordinates
    ``(u, v) = ((i + 1) / (nx + 1), (j + 1) / (ny + 1))`` is inside iff
    ``u + v < 1``.
    """
    i = np.arange(1, nx + 1, dtype=float) / (nx + 1)   # shape (nx,)
    j = np.arange(1, ny + 1, dtype=float) / (ny + 1)   # shape (ny,)
    U, V = np.meshgrid(i, j, indexing="ij")
    return (U + V) < 1.0


def triangle_eigenvalues(L: float, H: float, K: int,
                          nx: int = 48, ny: int = 48) -> np.ndarray:
    """Dirichlet eigenvalues of -Delta on the right triangle with legs
    ``L`` (along x) and ``H`` (along y), returned sorted ascending, first K.

    Uses a masked five-point Laplacian on the bounding rectangle. The grid
    spacing is ``hx = L / (nx + 1)`` and ``hy = H / (ny + 1)``; only interior
    nodes (i, j) with ``(i + 1) / (nx + 1) + (j + 1) / (ny + 1) < 1`` are kept
    as unknowns. The bordering nodes outside the triangle act as homogeneous
    Dirichlet ghosts (their contribution is just absent from the stencil).
    """
    if H <= 0 or L <= 0:
        raise ValueError("Triangle legs must be positive.")

    hx = L / (nx + 1)
    hy = H / (ny + 1)

    mask = triangle_mask(nx, ny)               # (nx, ny) bool
    idx = -np.ones((nx, ny), dtype=int)
    flat = np.flatnonzero(mask.ravel())
    idx_flat = idx.ravel()
    idx_flat[flat] = np.arange(flat.size)
    idx = idx_flat.reshape(nx, ny)
    n_dof = int(flat.size)

    if n_dof < K + 2:
        raise ValueError(
            f"Grid too coarse: only {n_dof} interior nodes for K={K} modes.")

    A = lil_matrix((n_dof, n_dof))
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    diag = 2.0 * (inv_hx2 + inv_hy2)

    for i in range(nx):
        for j in range(ny):
            p = idx[i, j]
            if p < 0:
                continue
            A[p, p] = diag
            for di, dj, w in ((-1, 0, inv_hx2), (1, 0, inv_hx2),
                              (0, -1, inv_hy2), (0,  1, inv_hy2)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    q = idx[ii, jj]
                    if q >= 0:
                        A[p, q] = -w
                # else: outside triangle or outside box; ghost Dirichlet, drop.

    A = A.tocsr()
    # Shift-invert near sigma=0 finds the smallest eigenvalues robustly.
    vals = eigsh(A, k=K, sigma=0.0, which="LM", return_eigenvectors=False)
    return np.sort(vals).astype(float)
