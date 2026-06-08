"""1D Poisson tridiagonal solve (Pillar 3).

Provides the second-order finite-difference solve of -u'' = f on [0, 1] with
homogeneous Dirichlet boundary conditions. It supplies the exact reference used
by the overlapping Schwarz experiment (E4e, `overlap.py`).
"""

import numpy as np
from scipy.linalg import solve_banded


def _tridiag_solve(n: int, rhs: np.ndarray, h: float) -> np.ndarray:
    """Solve -u'' = rhs on n interior points with spacing h, u(0)=u((n+1)h)=0.

    System: (2u_i - u_{i-1} - u_{i+1}) / h^2 = rhs_i.
    Uses banded LAPACK for O(n) time.
    """
    diag  =  2.0 / h**2 * np.ones(n)
    off   = -1.0 / h**2 * np.ones(n - 1)
    ab    = np.zeros((3, n))
    ab[0, 1:] = off      # superdiagonal
    ab[1, :]  = diag     # main diagonal
    ab[2, :-1] = off     # subdiagonal
    return solve_banded((1, 1), ab, rhs)
