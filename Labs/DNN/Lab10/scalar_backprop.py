"""Scalar computational graph and manual backpropagation for Lab 10.

The graph is

    s0 = w0 * x0
    s1 = w1 * x1
    s2 = s0 + s1
    s3 = s2 + w2
    L = sigmoid(s3)

Each value is a scalar PyTorch tensor. Implement the forward pass and the
manual backward pass without calling ``Tensor.backward`` or ``torch.autograd``
inside the TODO blocks.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch


ScalarMap = Mapping[str, torch.Tensor]
Cache = Dict[str, torch.Tensor]
GradientMap = Dict[str, torch.Tensor]

_INPUT_NAMES = ("w0", "x0", "w1", "x1", "w2")


def make_fixture(dtype: torch.dtype = torch.float64) -> Dict[str, torch.Tensor]:
    """Return fixed scalar inputs for the notebook checks."""
    return {
        "w0": torch.tensor(2.0, dtype=dtype),
        "x0": torch.tensor(-1.0, dtype=dtype),
        "w1": torch.tensor(-3.0, dtype=dtype),
        "x1": torch.tensor(-2.0, dtype=dtype),
        "w2": torch.tensor(-3.0, dtype=dtype),
    }


def _validate_inputs(values: ScalarMap) -> None:
    """Check that the graph receives the five required scalar tensors."""
    missing = set(_INPUT_NAMES).difference(values)
    extra = set(values).difference(_INPUT_NAMES)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected: {', '.join(sorted(extra))}")
        raise KeyError("Invalid graph inputs (" + "; ".join(details) + ")")
    for name in _INPUT_NAMES:
        if values[name].ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor")


def sigmoid(value: torch.Tensor) -> torch.Tensor:
    """Return the scalar sigmoid value."""
    output = None

    ############################################################################
    # TODO: Implement sigmoid(value) = 1 / (1 + exp(-value)).                 #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                             END OF YOUR CODE                            #
    ############################################################################

    return output


def scalar_forward(values: ScalarMap) -> Tuple[torch.Tensor, Cache]:
    """Compute the graph values and return the loss and a cache for backprop."""
    _validate_inputs(values)
    w0, x0 = values["w0"], values["x0"]
    w1, x1 = values["w1"], values["x1"]
    w2 = values["w2"]

    s0 = s1 = s2 = s3 = loss = None

    ############################################################################
    # TODO: Compute the forward pass in this order:                            #
    #                                                                          #
    #   s0 = w0 * x0                                                          #
    #   s1 = w1 * x1                                                          #
    #   s2 = s0 + s1                                                          #
    #   s3 = s2 + w2                                                          #
    #   loss = sigmoid(s3)                                                    #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                             END OF YOUR CODE                            #
    ############################################################################

    cache = {
        "w0": w0,
        "x0": x0,
        "w1": w1,
        "x1": x1,
        "w2": w2,
        "s0": s0,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "loss": loss,
    }
    return loss, cache


def scalar_backward(cache: Cache) -> GradientMap:
    """Compute manual gradients of the scalar loss with respect to all inputs."""
    required = {"w0", "x0", "w1", "x1", "w2", "s0", "s1", "s2", "s3", "loss"}
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"Cache is missing: {', '.join(sorted(missing))}")

    grads: GradientMap = {}

    ############################################################################
    # TODO: Backpropagate from loss to w0, x0, w1, x1, and w2.                #
    #                                                                          #
    # Start with grad_loss = 1. For sigmoid, use:                              #
    #                                                                          #
    #   grad_s3 = grad_loss * loss * (1 - loss)                               #
    #                                                                          #
    # Addition copies its upstream gradient to each input. For multiplication, #
    # multiply the upstream gradient by the other input. Store the final      #
    # gradients in grads using the input names shown above.                    #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                             END OF YOUR CODE                            #
    ############################################################################

    return grads


def numerical_gradient(values: ScalarMap, name: str, h: float = 1e-6) -> torch.Tensor:
    """Approximate one input derivative with a centered finite difference."""
    _validate_inputs(values)
    if name not in _INPUT_NAMES:
        raise KeyError(f"Unknown scalar input: {name}")
    if h <= 0:
        raise ValueError("h must be positive")

    plus = {key: value.detach().clone() for key, value in values.items()}
    minus = {key: value.detach().clone() for key, value in values.items()}
    plus[name] = plus[name] + h
    minus[name] = minus[name] - h
    plus_loss, _ = scalar_forward(plus)
    minus_loss, _ = scalar_forward(minus)
    return (plus_loss - minus_loss) / (2.0 * h)


def relative_error(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return a scale-aware difference between two scalar values."""
    numerator = (first - second).abs().item()
    denominator = max(1e-12, first.abs().item() + second.abs().item())
    return numerator / denominator


def gradient_report(values: ScalarMap, h: float = 1e-6) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Compare manual gradients with finite differences for every graph input."""
    _, cache = scalar_forward(values)
    analytic = scalar_backward(cache)
    report = {}
    for name in _INPUT_NAMES:
        numerical = numerical_gradient(values, name, h=h)
        report[name] = (numerical, analytic[name], relative_error(numerical, analytic[name]))
    return report
