"""FEM-on-grid Dirichlet eigensolver for L-shaped domains.

An L-shape is parametrised by four positive numbers ``(a, b, cx, cy)`` with
``0 < cx < a`` and ``0 < cy < b``. The domain is the Boolean difference

    L(a, b, cx, cy) = [0, a] x [0, b]  \\  [a - cx, a] x [b - cy, b]

i.e. the top-right rectangle of size ``cx x cy`` is removed. The reentrant
corner sits at ``(a - cx, b - cy)`` and produces an ``r^{2/3}`` singularity in
the lowest eigenmodes (interior angle ``3 pi / 2``).

Reference eigenvalues are computed by a five-point Laplacian on a regular grid
that covers the bounding rectangle ``[0, a] x [0, b]``, with interior nodes
inside the removed corner masked out (Dirichlet BC on the L's outer boundary
and on the reentrant notch are enforced by simply dropping those rows /
columns from the assembly). This mirrors the masked-grid stand-in for FEM that
``geometry/triangle_modes.py`` uses for triangles.

The eigensolve uses ``scipy.sparse.linalg.eigsh`` with a shift-invert near
``sigma = 0`` to find the smallest eigenvalues robustly.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


def lshape_mask(nx: int, ny: int, a: float, b: float,
                cx: float, cy: float) -> np.ndarray:
    """Boolean mask of shape ``(nx, ny)`` for the L-shape interior.

    The grid covers the bounding rectangle ``[0, a] x [0, b]`` with ``nx``
    interior columns and ``ny`` interior rows. A node at physical position
    ``(x_i, y_j) = ((i + 1) hx, (j + 1) hy)`` is inside the L-shape iff it is
    NOT inside the removed top-right cut ``[a - cx, a] x [b - cy, b]``.
    """
    hx = a / (nx + 1)
    hy = b / (ny + 1)
    i = np.arange(1, nx + 1, dtype=float) * hx
    j = np.arange(1, ny + 1, dtype=float) * hy
    X, Y = np.meshgrid(i, j, indexing="ij")
    inside_cut = (X >= a - cx) & (Y >= b - cy)
    return ~inside_cut


def lshape_eigenpairs(a: float, b: float, cx: float, cy: float, K: int,
                       nx: int = 64, ny: int = 64,
                       return_vectors: bool = False):
    """Dirichlet eigenvalues (and optionally eigenvectors) of -Delta on the
    L-shape ``L(a, b, cx, cy)``, sorted ascending, first ``K`` values.

    Parameters
    ----------
    a, b      : bounding-rectangle side lengths (positive).
    cx, cy    : sizes of the removed top-right corner (``0 < cx < a``,
                ``0 < cy < b``).
    K         : number of eigenmodes to return.
    nx, ny    : interior grid resolutions of the bounding rectangle.
    return_vectors : if True, also return eigenvector array shaped
                ``(K, nx, ny)`` with zeros on masked-out nodes (the cut).

    Returns
    -------
    eigs : (K,) ndarray, ascending Dirichlet eigenvalues.
    phis : (K, nx, ny) ndarray (only if ``return_vectors`` is True). Each mode
        is normalised to unit l2 over the interior grid (consistent units).
    """
    if a <= 0 or b <= 0 or cx <= 0 or cy <= 0:
        raise ValueError("All L-shape dimensions must be positive.")
    if cx >= a or cy >= b:
        raise ValueError("Cut must be strictly inside the bounding rectangle.")

    hx = a / (nx + 1)
    hy = b / (ny + 1)

    mask = lshape_mask(nx, ny, a, b, cx, cy)        # (nx, ny) bool
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
                # Else: outside box or inside the cut; ghost Dirichlet, drop.

    A = A.tocsr()
    if return_vectors:
        vals, vecs = eigsh(A, k=K, sigma=0.0, which="LM")
        order = np.argsort(vals)
        vals = vals[order].astype(float)
        vecs = vecs[:, order]
        phis = np.zeros((K, nx, ny), dtype=float)
        for kk in range(K):
            full = np.zeros(nx * ny, dtype=float)
            full[flat] = vecs[:, kk]
            phis[kk] = full.reshape(nx, ny)
            # Normalise to unit l2 over the active DOFs.
            nrm = np.linalg.norm(phis[kk])
            if nrm > 0:
                phis[kk] /= nrm
        return vals, phis

    vals = eigsh(A, k=K, sigma=0.0, which="LM", return_eigenvectors=False)
    return np.sort(vals).astype(float)


def lshape_eigenvalues(a: float, b: float, cx: float, cy: float, K: int,
                        nx: int = 64, ny: int = 64) -> np.ndarray:
    """Convenience wrapper that returns eigenvalues only."""
    return lshape_eigenpairs(a, b, cx, cy, K, nx=nx, ny=ny,
                              return_vectors=False)


def bounding_rectangle_eigenfunction(a: float, b: float, nx: int, ny: int,
                                       mask: np.ndarray | None = None,
                                       m: int = 1, n: int = 1) -> np.ndarray:
    """Restriction of the rectangle Dirichlet eigenfunction
    ``phi_{mn}(x, y) = sin(m pi x / a) sin(n pi y / b)`` to a grid that may be
    masked (for visualising the encoder's "rectangle-only" prediction next to
    the true L-shape eigenmode).

    The result has shape ``(nx, ny)``; if ``mask`` is given, masked-out
    entries are zeroed and the array is l2-normalised over the kept entries.
    """
    hx = a / (nx + 1)
    hy = b / (ny + 1)
    xs = np.arange(1, nx + 1, dtype=float) * hx
    ys = np.arange(1, ny + 1, dtype=float) * hy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    phi = np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
    if mask is not None:
        phi = phi * mask
    nrm = np.linalg.norm(phi)
    if nrm > 0:
        phi = phi / nrm
    return phi
