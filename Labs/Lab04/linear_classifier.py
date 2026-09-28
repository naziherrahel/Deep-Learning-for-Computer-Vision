"""Student starter code for Lab 04: linear scores and multiclass SVM loss."""

from __future__ import annotations

from typing import Tuple

import torch


def linear_scores(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return class scores for a batch of examples.

    Args:
        X: Tensor of shape ``(N, D)``.
        W: Tensor of shape ``(D, C)``.
        b: Optional bias tensor of shape ``(C,)``.

    Returns:
        Tensor of shape ``(N, C)``.
    """
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must both be rank-2 tensors")
    if X.shape[1] != W.shape[0]:
        raise ValueError("X.shape[1] must equal W.shape[0]")
    if b is not None and b.shape != (W.shape[1],):
        raise ValueError("b must have shape (number_of_classes,)")

    scores = X @ W
    if b is not None:
        scores = scores + b
    return scores


def svm_loss_naive(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.0,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute multiclass SVM loss and gradient using explicit loops.

    Lab 04 uses the unregularized objective

    ``mean(data_loss)``.

    The ``reg`` argument remains in the interface so this function can be
    extended in Lab 05. Keep it at ``0.0`` in this lab.

    Args:
        W: Weights of shape ``(D, C)``.
        X: Minibatch of shape ``(N, D)``.
        y: Integer labels of shape ``(N,)``.
        reg: Reserved for Lab 05; must be ``0.0`` in Lab 04.
        delta: SVM margin, normally ``1.0``.

    Returns:
        A scalar loss and a gradient tensor with the same shape as ``W``.
    """
    if X.ndim != 2 or W.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected X=(N,D), W=(D,C), and y=(N,)")
    if X.shape[0] != y.shape[0] or X.shape[1] != W.shape[0]:
        raise ValueError("Input shapes are incompatible")
    if reg != 0.0:
        raise ValueError("Lab 04 uses reg=0.0; L2 regularization begins in Lab 05")

    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = W.new_tensor(0.0)
    dW = torch.zeros_like(W)

    for i in range(num_train):
        scores = linear_scores(X[i : i + 1], W).squeeze(0)
        correct_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_score + delta
            if margin > 0:
                # TODO 1: accumulate this positive margin into ``loss``.
                # TODO 2: add this example's contribution to ``dW``.
                pass

    # TODO 3: average the data loss and data gradient over the minibatch.
    raise NotImplementedError("Complete svm_loss_naive for Lab 04")
