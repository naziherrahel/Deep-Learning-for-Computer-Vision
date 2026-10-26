"""Deterministic, download-free fixtures for Lab 08."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


TensorDict = Dict[str, torch.Tensor]


def make_forward_fixture(
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor | TensorDict]:
    """Return a hand-checkable two-layer-network forward-pass fixture."""
    X = torch.tensor(
        [[1.0, 2.0, -1.0], [-2.0, 1.0, 3.0]],
        dtype=dtype,
    )
    params: TensorDict = {
        "W1": torch.tensor(
            [[1.0, -1.0, 0.0, 2.0], [0.0, 1.0, 1.0, -1.0], [1.0, 0.0, -1.0, 1.0]],
            dtype=dtype,
        ),
        "b1": torch.tensor([0.5, -0.5, 0.0, 1.0], dtype=dtype),
        "W2": torch.tensor(
            [[1.0, -1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
            dtype=dtype,
        ),
        "b2": torch.tensor([0.2, -0.1, 0.3], dtype=dtype),
    }
    return {
        "X": X,
        "params": params,
        "expected_hidden": torch.tensor(
            [[0.5, 0.5, 3.0, 0.0], [1.5, 2.5, 0.0, 0.0]],
            dtype=dtype,
        ),
        "expected_scores": torch.tensor(
            [[3.7, -0.1, -2.2], [1.7, 0.9, 2.8]],
            dtype=dtype,
        ),
        "expected_predictions": torch.tensor([0, 2], dtype=torch.int64),
    }


def make_linear_equivalence_fixture(
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return matrices for two linear layers that can be collapsed into one."""
    X = torch.tensor(
        [[-1.0, 0.5], [0.0, 2.0], [1.5, -0.5], [2.0, 1.0]],
        dtype=dtype,
    )
    W1 = torch.tensor(
        [[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]],
        dtype=dtype,
    )
    W2 = torch.tensor(
        [[1.0, -1.0], [0.0, 2.0], [2.0, 0.5]],
        dtype=dtype,
    )
    return X, W1, W2


def make_diamond_network(
    dtype: torch.dtype = torch.float64,
) -> TensorDict:
    """Return a fixed ReLU network whose first class is a diamond-shaped area."""
    return {
        "W1": torch.tensor(
            [[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]],
            dtype=dtype,
        ),
        "b1": torch.zeros(4, dtype=dtype),
        "W2": torch.tensor(
            [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
            dtype=dtype,
        ),
        "b2": torch.tensor([1.0, 0.0], dtype=dtype),
    }


def make_input_grid(
    limit: float = 2.0,
    steps: int = 181,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a regular 2-D grid as points and two plotting coordinate arrays."""
    axis = torch.linspace(-limit, limit, steps, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    return points, grid_x, grid_y
