"""Small teaching utilities for the 2026 computer-vision labs."""

from .data import make_toy_classification
from .grad import grad_check_sparse, relative_error

__all__ = ["grad_check_sparse", "make_toy_classification", "relative_error"]
