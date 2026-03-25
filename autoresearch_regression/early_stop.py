"""Early stopping on validation RMSE (lower is better).

Kept in a small module so experiment scripts can import reliably even when
``prepare.py`` is edited for data-loading only.
"""

from __future__ import annotations

import os


class EarlyStopper:
    """Stop when val_rmse does not improve by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = max(0, patience)
        self.min_delta = min_delta
        self.best = float("inf")
        self._bad_epochs = 0

    def step(self, val_rmse: float) -> bool:
        """Return True if training should stop."""
        if self.patience <= 0:
            return False
        if val_rmse < self.best - self.min_delta:
            self.best = float(val_rmse)
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
        return self._bad_epochs >= self.patience


def early_stopper_from_env() -> EarlyStopper:
    """Build from ``EARLY_STOP_PATIENCE`` (0 = disabled) and ``EARLY_STOP_MIN_DELTA``."""
    p = int(os.environ.get("EARLY_STOP_PATIENCE", "3"))
    md = float(os.environ.get("EARLY_STOP_MIN_DELTA", "1e-4"))
    return EarlyStopper(p, md)
