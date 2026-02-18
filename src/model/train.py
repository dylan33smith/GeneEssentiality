"""Training loop and utilities for essentiality classification."""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_class_weights(labels: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data.

    Formula: weight[c] = n_total / (n_classes * count[c])
    Use with CrossEntropyLoss to upweight rare classes (always_essential, conditional).

    Args:
        labels: 1D tensor of class indices (0, 1, 2). Should be from training set only.
        n_classes: Number of classes (default 3).

    Returns:
        Tensor of shape (n_classes,) with weight per class.
    """
    # TODO: use torch.bincount(labels, minlength=n_classes) to get counts
    # TODO: avoid division by zero (classes with 0 samples)
    # TODO: compute weight[c] = total / (n_classes * count[c])
    pass


def make_loss_fn(labels: torch.Tensor, n_classes: int = 3, device: torch.device | None = None) -> nn.CrossEntropyLoss:
    """Create CrossEntropyLoss with class weights from training labels.

    Args:
        labels: Training set labels (1D tensor of 0, 1, 2).
        n_classes: Number of classes.
        device: Device to put weights on (e.g. same as model).

    Returns:
        Configured CrossEntropyLoss.
    """
    # TODO: call compute_class_weights(labels, n_classes)
    # TODO: move weights to device if provided
    # TODO: return nn.CrossEntropyLoss(weight=weights)
    pass
