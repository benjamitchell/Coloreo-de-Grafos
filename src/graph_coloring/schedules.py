"""Cooling schedules for simulated annealing.

A schedule is a callable ``T(i)`` returning the temperature at iteration
``i`` (``0 <= i < n_iter``). All schedules here are strictly positive.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

Schedule = Callable[[int], float]


@dataclass(frozen=True)
class Constant:
    """Fixed temperature: plain Metropolis sampling at ``T``."""

    T: float

    def __call__(self, i: int) -> float:
        return self.T


@dataclass(frozen=True)
class Geometric:
    """Exponential cooling ``T(i) = T0 * alpha ** (i // step)``.

    The most common choice in practice. ``alpha`` close to 1 cools slowly.
    ``T_min`` keeps the temperature away from zero.
    """

    T0: float = 2.0
    alpha: float = 0.999
    step: int = 1
    T_min: float = 1e-3

    def __call__(self, i: int) -> float:
        return max(self.T0 * self.alpha ** (i // self.step), self.T_min)


@dataclass(frozen=True)
class Linear:
    """Linear cooling from ``T0`` down to ``T_min`` over ``n_iter`` steps."""

    T0: float = 2.0
    n_iter: int = 100_000
    T_min: float = 1e-3

    def __call__(self, i: int) -> float:
        frac = min(i / self.n_iter, 1.0)
        return max(self.T0 * (1.0 - frac), self.T_min)


@dataclass(frozen=True)
class Logarithmic:
    """Logarithmic cooling ``T(i) = c / log(i + 2)``.

    For ``c`` larger than the depth of the deepest local minimum this
    schedule converges in probability to a global minimum (Hajek, 1988),
    but it is far too slow to be practical. Included for comparison.
    """

    c: float = 2.0

    def __call__(self, i: int) -> float:
        return self.c / math.log(i + 2)


def geometric_for(n_iter: int, T0: float = 0.6, T_end: float = 0.1) -> Geometric:
    """Geometric schedule going from ``T0`` to ``T_end`` in ``n_iter`` steps."""
    alpha = (T_end / T0) ** (1.0 / max(n_iter - 1, 1))
    return Geometric(T0=T0, alpha=alpha, T_min=T_end)
