"""Deterministic, download-free datasets for linear-classifier labs."""

from __future__ import annotations

from typing import Dict

import torch


def make_toy_classification(
    train_per_class: int = 120,
    val_per_class: int = 40,
    noise: float = 0.65,
    bias_trick: bool = True,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Create a three-class 2D dataset suitable for quick CPU experiments."""
    if train_per_class <= 0 or val_per_class <= 0:
        raise ValueError("Samples per class must be positive")
    if noise <= 0:
        raise ValueError("noise must be positive")

    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [[-2.2, -1.2], [2.2, -1.0], [0.0, 2.3]], dtype=dtype
    )

    def build_split(samples_per_class: int):
        features = []
        labels = []
        for class_index, center in enumerate(centers):
            points = center + noise * torch.randn(
                samples_per_class, 2, generator=generator, dtype=dtype
            )
            features.append(points)
            labels.append(
                torch.full((samples_per_class,), class_index, dtype=torch.int64)
            )
        X = torch.cat(features, dim=0)
        y = torch.cat(labels, dim=0)
        permutation = torch.randperm(X.shape[0], generator=generator)
        X = X[permutation]
        y = y[permutation]
        if bias_trick:
            X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=dtype)], dim=1)
        return X, y

    X_train, y_train = build_split(train_per_class)
    X_val, y_val = build_split(val_per_class)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
