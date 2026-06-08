"""Spectral convolution layer for the Schwarz subdomain operators (Pillar 3).

Used by the overlapping-Schwarz experiment E4e (`run_e4e.py`), whose subdomain
Fourier operators and learned interface map are built on this block.
"""

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, n_modes: int):
        super().__init__()
        self.n_modes = n_modes
        self.w = nn.Parameter(
            torch.randn(width, width, n_modes, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf  = torch.fft.rfft(x, dim=-1)
        m   = min(self.n_modes, xf.size(-1))
        out = torch.zeros_like(xf)
        out[..., :m] = torch.einsum("biM,ioM->boM", xf[..., :m], self.w[..., :m])
        return torch.fft.irfft(out, n=x.size(-1), dim=-1)
