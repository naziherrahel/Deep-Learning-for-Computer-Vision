"""Numerical-gradient utilities supplied to students."""

from __future__ import annotations

import random
from typing import Callable, List

import torch


def relative_error(a: torch.Tensor | float, b: torch.Tensor | float) -> float:
    """Return a scale-independent relative error for two scalar values."""
    a_value = float(a)
    b_value = float(b)
    return abs(a_value - b_value) / (abs(a_value) + abs(b_value) + 1e-12)


def grad_check_sparse(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    analytic_grad: torch.Tensor,
    num_checks: int = 10,
    h: float = 1e-6,
    seed: int = 0,
) -> List[float]:
    """Compare sampled analytic-gradient entries with finite differences."""
    if x.shape != analytic_grad.shape:
        raise ValueError("x and analytic_grad must have the same shape")

    rng = random.Random(seed)
    errors: List[float] = []
    with torch.no_grad():
        for _ in range(num_checks):
            index = tuple(rng.randrange(size) for size in x.shape)
            old_value = x[index].item()

            x[index] = old_value + h
            loss_plus = f(x).item()
            x[index] = old_value - h
            loss_minus = f(x).item()
            x[index] = old_value

            numerical = (loss_plus - loss_minus) / (2.0 * h)
            analytic = analytic_grad[index].item()
            error = relative_error(numerical, analytic)
            errors.append(error)
            print(
                f"index={index} numerical={numerical:+.7e} "
                f"analytic={analytic:+.7e} relative_error={error:.3e}"
            )
    return errors
