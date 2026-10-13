"""Linear-classifier functions for Lab 06: gradient descent and SGD."""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch


LossFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    Tuple[torch.Tensor, torch.Tensor],
]


def svm_loss_vectorized(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the vectorized multiclass SVM loss and analytic gradient."""
    if X.ndim != 2 or W.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected X=(N,D), W=(D,C), and y=(N,)")
    if X.shape[0] != y.shape[0] or X.shape[1] != W.shape[0]:
        raise ValueError("Input shapes are incompatible")

    num_train = X.shape[0]
    scores = X @ W
    row_indices = torch.arange(num_train, device=y.device)
    correct_scores = scores[row_indices, y].unsqueeze(1)
    margins = torch.clamp(scores - correct_scores + 1.0, min=0.0)
    margins[row_indices, y] = 0.0

    loss = margins.sum() / num_train + reg * torch.sum(W * W)

    coefficients = (margins > 0).to(W.dtype)
    coefficients[row_indices, y] = -coefficients.sum(dim=1)
    dW = X.T @ coefficients / num_train + 2 * reg * W
    return loss, dW


def sample_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` training examples with replacement."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected X=(N,D) and y=(N,)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of examples")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    X_batch = None
    y_batch = None

    indices = torch.randint(
        low=0,
        high=X.shape[0],
        size=(batch_size,),
        generator=generator,
        device=X.device,
    )

    X_batch = X[indices]
    y_batch = y[indices]

    return X_batch, y_batch

def train_linear_classifier(
    loss_func: LossFunction,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-1,
    reg: float = 1e-3,
    num_iters: int = 400,
    batch_size: int = 64,
    W: torch.Tensor | None = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, List[float]]:
    """Train a linear classifier with minibatch stochastic gradient descent."""
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected X=(N,D) and y=(N,)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of examples")
    if learning_rate <= 0 or num_iters <= 0 or batch_size <= 0:
        raise ValueError("learning_rate, num_iters, and batch_size must be positive")

    generator = torch.Generator(device=X.device).manual_seed(seed)
    if W is None:
        num_classes = int(y.max().item()) + 1
        W = 1e-3 * torch.randn(
            X.shape[1],
            num_classes,
            dtype=X.dtype,
            device=X.device,
            generator=generator,
        )

    loss_history: List[float] = []
    for _ in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, batch_size, generator)
        loss, gradient = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(float(loss))

        #########################################################################
        # TODO:                                                                 #
        # Apply one SGD update to W using the learning rate and gradient.        #
        #########################################################################
        # Replace "pass" statement with your code
        W -= learning_rate * gradient
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Return the highest-scoring class for each example."""
    return (X @ W).argmax(dim=1)


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Return classification accuracy in the range [0, 1]."""
    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match")
    return float((y_pred == y_true).to(torch.float64).mean())
