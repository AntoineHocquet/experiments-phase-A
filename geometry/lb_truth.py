"""Analytical Laplace-Beltrami eigenvalues for simple reference geometries.

These closed-form expressions serve as free, exact ground truth for training
and evaluating the geometry encoder. No FEM solver is needed for any of the
geometries below.

Geometries and their Dirichlet eigenvalues (ascending order, first K values):
  Interval [0, L]:         lambda_k  = (k pi / L)^2                k = 1, 2, ...
  Rectangle [0,a]x[0,b]:   lambda_mn = (m pi / a)^2 + (n pi / b)^2  m, n = 1, 2, ...
  Box [0,a]x[0,b]x[0,c]:   lambda    = pi^2 (n^2/a^2 + m^2/b^2 + l^2/c^2)
"""

import numpy as np


def interval_eigenvalues(L: float, K: int) -> np.ndarray:
    """Dirichlet eigenvalues on [0, L]: lambda_k = (k pi / L)^2 for k = 1, ..., K."""
    k = np.arange(1, K + 1, dtype=float)
    return (k * np.pi / L) ** 2


def rectangle_eigenvalues(a: float, b: float, K: int) -> np.ndarray:
    """Dirichlet eigenvalues on [0,a] × [0,b] sorted ascending, first K."""
    max_mn = K + 4           # over-generate to be safe
    eigs = [
        (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
        for m in range(1, max_mn + 1)
        for n in range(1, max_mn + 1)
    ]
    return np.sort(eigs)[:K].astype(float)


def box_eigenvalues(a: float, b: float, c: float, K: int) -> np.ndarray:
    """Dirichlet eigenvalues on the 3D box [0,a] x [0,b] x [0,c], sorted ascending.

    Closed form: lambda_{nml} = pi^2 (n^2/a^2 + m^2/b^2 + l^2/c^2), with
    n, m, l = 1, 2, ...  Returns the first K values in ascending order.
    """
    max_nml = K + 4           # over-generate to be safe
    pi2 = np.pi ** 2
    eigs = [
        pi2 * ((n / a) ** 2 + (m / b) ** 2 + (l / c) ** 2)
        for n in range(1, max_nml + 1)
        for m in range(1, max_nml + 1)
        for l in range(1, max_nml + 1)
    ]
    return np.sort(eigs)[:K].astype(float)
