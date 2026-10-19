"""Linear-classifier functions for Lab 07: optimization algorithms."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import torch


LossFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    Tuple[torch.Tensor, torch.Tensor],
]
OptimizerConfig = Dict[str, Any]
UpdateRule = Callable[
    [torch.Tensor, torch.Tensor, OptimizerConfig],
    Tuple[torch.Tensor, OptimizerConfig],
]


def svm_loss_vectorized(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the vectorized multiclass SVM loss and analytic gradient."""
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


def sgd_update(
    W: torch.Tensor,
    gradient: torch.Tensor,
    config: OptimizerConfig,
) -> Tuple[torch.Tensor, OptimizerConfig]:
    """Apply one stochastic-gradient-descent update."""
    learning_rate = float(config.get("learning_rate", 1e-2))
    return W - learning_rate * gradient, config


def sgd_momentum_update(
    W: torch.Tensor,
    gradient: torch.Tensor,
    config: OptimizerConfig,
) -> Tuple[torch.Tensor, OptimizerConfig]:
    """Apply one SGD-with-momentum update."""
    learning_rate = float(config.get("learning_rate", 1e-2))
    momentum = float(config.get("momentum", 0.9))
    velocity = config.get("velocity", torch.zeros_like(W))
    next_W = W.clone()

    #############################################################################
    # TODO:                                                                     #
    # Update velocity using                                                     #
    #                                                                           #
    #   velocity = momentum * velocity - learning_rate * gradient               #
    #                                                                           #
    # then use that velocity to update next_W.                                  #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    config["velocity"] = velocity
    return next_W, config


def nesterov_update(
    W: torch.Tensor,
    gradient: torch.Tensor,
    config: OptimizerConfig,
) -> Tuple[torch.Tensor, OptimizerConfig]:
    """Apply Nesterov momentum using the rearranged update from the lecture."""
    learning_rate = float(config.get("learning_rate", 1e-2))
    momentum = float(config.get("momentum", 0.9))
    velocity = config.get("velocity", torch.zeros_like(W))
    previous_velocity = velocity
    velocity = momentum * velocity - learning_rate * gradient
    next_W = W - momentum * previous_velocity + (1.0 + momentum) * velocity
    config["velocity"] = velocity
    return next_W, config


def adagrad_update(
    W: torch.Tensor,
    gradient: torch.Tensor,
    config: OptimizerConfig,
) -> Tuple[torch.Tensor, OptimizerConfig]:
    """Apply one AdaGrad update."""
    learning_rate = float(config.get("learning_rate", 1e-2))
    epsilon = float(config.get("epsilon", 1e-8))
    squared_gradient_sum = config.get(
        "squared_gradient_sum", torch.zeros_like(W)
    )
    squared_gradient_sum = squared_gradient_sum + gradient * gradient
    next_W = W - learning_rate * gradient / (
        torch.sqrt(squared_gradient_sum) + epsilon
    )
    config["squared_gradient_sum"] = squared_gradient_sum
    return next_W, config


def rmsprop_update(
    W: torch.Tensor,
    gradient: torch.Tensor,
    config: OptimizerConfig,
) -> Tuple[torch.Tensor, OptimizerConfig]:
    """Apply one RMSProp update."""
    learning_rate = float(config.get("learning_rate", 1e-2))
    decay_rate = float(config.get("decay_rate", 0.99))
    epsilon = float(config.get("epsilon", 1e-8))
    squared_gradient_average = config.get(
        "squared_gradient_average", torch.zeros_like(W)
    )
    squared_gradient_average = (
        decay_rate * squared_gradient_average
        + (1.0 - decay_rate) * gradient * gradient
    )
    next_W = W - learning_rate * gradient / (
        torch.sqrt(squared_gradient_average) + epsilon
    )
    config["squared_gradient_average"] = squared_gradient_average
    return next_W, config


def adam_update(
    W: torch.Tensor,
    gradient: torch.Tensor,
    config: OptimizerConfig,
) -> Tuple[torch.Tensor, OptimizerConfig]:
    """Apply one Adam update with bias correction."""
    learning_rate = float(config.get("learning_rate", 1e-3))
    beta1 = float(config.get("beta1", 0.9))
    beta2 = float(config.get("beta2", 0.999))
    epsilon = float(config.get("epsilon", 1e-8))
    first_moment = config.get("first_moment", torch.zeros_like(W))
    second_moment = config.get("second_moment", torch.zeros_like(W))
    timestep = int(config.get("timestep", 0)) + 1
    next_W = W.clone()

    #############################################################################
    # TODO:                                                                     #
    # 1. Update the first and second moment estimates.                          #
    # 2. Correct both estimates for their initialization bias.                  #
    # 3. Update next_W using the corrected estimates and epsilon.               #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    config["first_moment"] = first_moment
    config["second_moment"] = second_moment
    config["timestep"] = timestep
    return next_W, config


def sample_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a minibatch with replacement."""
    indices = torch.randint(
        X.shape[0],
        (batch_size,),
        device=X.device,
        generator=generator,
    )
    return X[indices], y[indices]


def train_linear_classifier(
    loss_func: LossFunction,
    update_rule: UpdateRule,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer_config: OptimizerConfig | None = None,
    reg: float = 1e-3,
    num_iters: int = 300,
    batch_size: int = 64,
    W: torch.Tensor | None = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, List[float]]:
    """Train a linear classifier with the selected update rule."""
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

    config = dict(optimizer_config or {})
    loss_history: List[float] = []
    for _ in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, batch_size, generator)
        loss, gradient = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(float(loss))
        W, config = update_rule(W, gradient, config)

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Return the highest-scoring class for each example."""
    return (X @ W).argmax(dim=1)


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Return classification accuracy in the range [0, 1]."""
    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match")
    return float((y_pred == y_true).to(torch.float64).mean())
