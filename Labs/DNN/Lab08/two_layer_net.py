"""Two-layer neural-network forward pass for Lab 08.

The network computes

    X -> fully connected layer -> ReLU -> fully connected layer -> scores.

This lab intentionally stops at the forward pass.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


Parameters = Dict[str, torch.Tensor]


def relu(values: torch.Tensor) -> torch.Tensor:
    """Apply the ReLU activation elementwise.

    The output has the same shape, dtype, and device as ``values``. Do not use
    ``torch.nn`` modules in this lab.
    """
    activated = None

    ############################################################################
    # TODO: Apply ReLU to values. Negative entries become zero and positive    #
    # entries are unchanged.                                                   #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                             END OF YOUR CODE                            #
    ############################################################################

    return activated


def two_layer_forward(
    params: Parameters,
    X: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scores and hidden activations for a two-layer neural network.

    ``params`` contains:

    - ``W1`` with shape ``(D, H)`` and ``b1`` with shape ``(H,)``;
    - ``W2`` with shape ``(H, C)`` and ``b2`` with shape ``(C,)``.

    ``X`` has shape ``(N, D)``. Return class scores with shape ``(N, C)`` and
    post-ReLU hidden activations with shape ``(N, H)``.
    """
    required = {"W1", "b1", "W2", "b2"}
    missing = required.difference(params)
    if missing:
        raise KeyError(f"Missing parameter(s): {', '.join(sorted(missing))}")
    if X.ndim != 2:
        raise ValueError("X must have shape (N, D)")

    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    if W1.ndim != 2 or W2.ndim != 2:
        raise ValueError("W1 and W2 must be matrices")
    if X.shape[1] != W1.shape[0] or W1.shape[1] != W2.shape[0]:
        raise ValueError("Incompatible X, W1, and W2 shapes")
    if b1.shape != (W1.shape[1],) or b2.shape != (W2.shape[1],):
        raise ValueError("Bias shapes must match their layer outputs")

    hidden = None
    scores = None

    ############################################################################
    # TODO: Compute the network forward pass:                                 #
    #                                                                          #
    #   hidden_linear = X @ W1 + b1                                           #
    #   hidden = relu(hidden_linear)                                          #
    #   scores = hidden @ W2 + b2                                             #
    #                                                                          #
    # ``@`` and ``torch.matmul`` are both valid matrix-multiplication forms   #
    # for these two-dimensional tensors.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                             END OF YOUR CODE                            #
    ############################################################################

    return scores, hidden


def predict_two_layer(params: Parameters, X: torch.Tensor) -> torch.Tensor:
    """Return the index of the largest class score for each input."""
    scores, _ = two_layer_forward(params, X)
    predicted_classes = None

    ############################################################################
    # TODO: Select the highest-scoring class for every row of scores.          #
    # Return a tensor of shape (N,) containing class indices.                  #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                             END OF YOUR CODE                            #
    ############################################################################

    return predicted_classes
