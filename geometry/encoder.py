"""Geometry encoder: maps a compact geometric descriptor to LB eigenvalues.

Architecture: a depth-3 MLP with GELU activations, **operating in log–log space**.
  Caller-visible API:
    Input  : geometry descriptor of dimension d_in (positive entries: an
             interval length L, a rectangle's (a, b), or a disk radius R).
    Output : K predicted LB eigenvalues, strictly positive.
  Internal:
    Inputs are log-transformed before the MLP; the MLP outputs log-eigenvalues
    which are exponentiated on the way out.  Because the analytic scaling laws
    (e.g. λ_k = (kπ/L)² ⇒ log λ_k = 2 log(kπ) − 2 log L) are *affine* in
    log-space, the MLP only has to fit an affine map per mode, something it
    extrapolates well beyond the training range.  In raw (L, λ) coordinates a
    vanilla MLP cannot extrapolate the 1/L² power law and the in-family OOD
    test at L > 2 fails badly even at near-zero training loss.

The training loss in run_e3a.py is ``F.mse_loss(torch.log(pred), torch.log(y))``,
which with this encoder reduces to MSE between the MLP output and ``log y``;
pure log-space regression, no Jacobian games.
"""

import torch
import torch.nn as nn


class EigenvalueEncoder(nn.Module):
    """MLP: geometry descriptor → K predicted LB eigenvalues.

    Internally maps ``log(descriptor) → log(eigenvalues)``; exponentiated on the
    way out so the caller still sees positive λ values and the API is unchanged.

    Parameters
    ----------
    d_in   : dimension of the geometry descriptor (positive entries)
    K      : number of eigenvalues to predict
    hidden : width of each hidden layer
    depth  : number of hidden layers (≥ 1)
    """

    def __init__(self, d_in: int = 1, K: int = 10,
                 hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, K))
        self.net = nn.Sequential(*layers)
        self.K   = K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (…, d_in) positive descriptor → (…, K) positive eigenvalues.

        The clamp_min guards against numerical zeros / negatives in unusual
        inputs (e.g. an aggressive optimisation step on a learned descriptor);
        in all default callers x is strictly positive.
        """
        return torch.exp(self.net(torch.log(x.clamp_min(1e-8))))
