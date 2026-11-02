"""Small PyTorch models for Lab 09.

The lab uses a two-dimensional XOR-style dataset so that model capacity and
the effect of ReLU can be seen directly. PyTorch computes gradients for the
training step; manual backpropagation is deliberately outside this lab.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def make_xor_dataset(
    points_per_corner: int = 80,
    noise: float = 0.25,
    seed: int = 7,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a shuffled two-class dataset with an XOR-style arrangement."""
    if points_per_corner < 1:
        raise ValueError("points_per_corner must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]],
        dtype=dtype,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    X = centers.repeat_interleave(points_per_corner, dim=0)
    X = X + noise * torch.randn(X.shape, generator=generator, dtype=dtype)
    y = labels.repeat_interleave(points_per_corner)
    permutation = torch.randperm(X.shape[0], generator=generator)
    return X[permutation], y[permutation]


def make_two_layer_model(hidden_width: int) -> nn.Sequential:
    """Construct a two-layer classifier for two input features and two classes."""
    if hidden_width < 1:
        raise ValueError("hidden_width must be at least 1")

    model = None

    ############################################################################
    # TODO: Build and return a two-layer classifier with this exact sequence: #
    #                                                                        #
    #   nn.Linear(2, hidden_width) -> nn.ReLU() -> nn.Linear(hidden_width, 2) #
    #                                                                        #
    # Use nn.Sequential so that calling model(X) returns class scores.        #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                             END OF YOUR CODE                            #
    ############################################################################

    return model


def train_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one PyTorch training step and return the scalar cross-entropy loss."""
    loss = None

    ############################################################################
    # TODO: Implement one training step in the following order:              #
    #                                                                        #
    #   1. scores = model(X)                                                 #
    #   2. loss = F.cross_entropy(scores, y)                                 #
    #   3. optimizer.zero_grad()                                             #
    #   4. loss.backward()                                                   #
    #   5. optimizer.step()                                                  #
    #                                                                        #
    # Return loss.item() as a Python float. Do not derive gradients manually. #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                             END OF YOUR CODE                            #
    ############################################################################

    return loss.item()


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Return classification accuracy for a batch."""
    predictions = model(X).argmax(dim=1)
    return (predictions == y).float().mean().item()


def train_for_steps(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    steps: int,
) -> list[float]:
    """Train on one fixed batch for a requested number of gradient steps."""
    if steps < 1:
        raise ValueError("steps must be positive")
    return [train_step(model, X, y, optimizer) for _ in range(steps)]


@torch.no_grad()
def parameter_l2_norm(parameters: Iterable[torch.Tensor]) -> float:
    """Return the combined L2 norm of a model's trainable parameters."""
    squared_norm = sum(parameter.square().sum() for parameter in parameters)
    return torch.sqrt(squared_norm).item()


@torch.no_grad()
def prediction_grid(
    model: nn.Module,
    limit: float = 2.25,
    steps: int = 181,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a grid and its predicted labels for plotting a decision region."""
    if limit <= 0 or steps < 2:
        raise ValueError("limit must be positive and steps must be at least 2")
    parameter = next(model.parameters())
    axis = torch.linspace(limit * -1, limit, steps, device=parameter.device)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    predictions = model(points).argmax(dim=1).reshape(grid_x.shape)
    return grid_x.cpu(), grid_y.cpu(), predictions.cpu()
