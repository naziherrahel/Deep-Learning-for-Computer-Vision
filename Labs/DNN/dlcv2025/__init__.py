# dlcv2025/__init__.py

# Import utils first (safe)
from . import utils

# Re-export convenience functions
reset_seed = utils.reset_seed
tensor_to_image = utils.tensor_to_image
visualize_dataset = utils.visualize_dataset

# Optional: lazy-load other submodules to avoid circular imports
import importlib

def __getattr__(name):
    """Dynamically import submodules when accessed."""
    if name in {"data", "grad", "submit"}:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
