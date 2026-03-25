"""Modular wrappers for autoregression experiments.

This package organizes reusable components:
- data/    : DataLoaders (default, by_gene, contrastive)
- models/  : Model architectures (mlp, twohead, residual, etc.)
- losses/  : Loss functions (mse, pairwise, huber, combined)
- trainers/: Training loops (default, curriculum, multi_task)
"""

from . import data
from . import models
from . import losses
from . import trainers

__version__ = "0.1.0"
__all__ = ["data", "models", "losses", "trainers"]
