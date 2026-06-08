"""Hardcoded operator-splitting composition (Strang).

Strang splitting composes two elementary "bricks", each a callable
``f(u, dt) -> u`` (torch.Tensor), to second order:

    F_A(dt/2) o F_B(dt) o F_A(dt/2),   error O(dt^3) per step.

The composition is algebraically determined; nothing is learned or optimised.

Reference: Strang, G. (1968). SIAM J. Numer. Anal. 5(3):506-517.
"""

from typing import Callable
import torch


Brick = Callable[[torch.Tensor, float], torch.Tensor]


def strang(brick_a: Brick, brick_b: Brick,
           u: torch.Tensor, dt: float) -> torch.Tensor:
    """F_A(dt/2) o F_B(dt) o F_A(dt/2), second-order Strang splitting.

    Error per step: O(dt^3). Requires each brick to accept an arbitrary dt
    (not just a fixed training dt), so that the half-steps are in-distribution.
    """
    u = brick_a(u, dt / 2.0)
    u = brick_b(u, dt)
    u = brick_a(u, dt / 2.0)
    return u
