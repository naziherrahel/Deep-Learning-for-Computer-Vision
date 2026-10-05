"""Student starter code for Lab 05: vectorized SVM training."""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch


LossFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    Tuple[torch.Tensor, torch.Tensor],
]


def linear_scores(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return class scores for a batch of examples."""
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must both be rank-2 tensors")
    if X.shape[1] != W.shape[0]:
        raise ValueError("X.shape[1] must equal W.shape[0]")
    if b is not None and b.shape != (W.shape[1],):
        raise ValueError("b must have shape (number_of_classes,)")

    # For 2-D PyTorch tensors, ``X @ W`` is equivalent to
    # ``torch.matmul(X, W)``. Both perform matrix multiplication.
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
    """Compute the regularized multiclass SVM loss with explicit loops.

    This completed reference is supplied so Lab 05 can focus on vectorization.
    It uses ``mean(data_loss) + reg * sum(W * W)``; the course convention does
    not include a factor of one half in the regularization loss.
    """
    if X.ndim != 2 or W.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected X=(N,D), W=(D,C), and y=(N,)")
    if X.shape[0] != y.shape[0] or X.shape[1] != W.shape[0]:
        raise ValueError("Input shapes are incompatible")

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
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss = loss / num_train + reg * torch.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    return loss, dW


def svm_loss_vectorized(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute multiclass SVM loss and gradient without Python loops.

    Use the objective ``mean(data_loss) + reg * sum(W * W)``.
    Do not call ``svm_loss_naive`` from this function.
    You may use either ``A @ B`` or ``torch.matmul(A, B)`` for matrix products.
    """
    loss = W.new_tensor(0.0)
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse intermediate values that you used to compute the loss.           #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def sample_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a minibatch with replacement from ``X`` and ``y``."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of examples")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = torch.randint(
        X.shape[0],
        (batch_size,),
        device=X.device,
        generator=generator,
    )
    return X[indices], y[indices]


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
    if W is None:
        num_classes = int(y.max().item()) + 1
        generator = torch.Generator(device=X.device).manual_seed(seed)
        W = 1e-3 * torch.randn(
            X.shape[1],
            num_classes,
            dtype=X.dtype,
            device=X.device,
            generator=generator,
        )
    else:
        generator = torch.Generator(device=X.device).manual_seed(seed)

    loss_history: List[float] = []
    for _ in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, batch_size, generator)
        loss, gradient = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(float(loss))

        W -= learning_rate * gradient

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Return the highest-scoring class for each example."""
    return linear_scores(X, W).argmax(dim=1)


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Return classification accuracy in the range [0, 1]."""
    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match")
    return float((y_pred == y_true).to(torch.float64).mean())
