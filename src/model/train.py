"""Training loop and utilities for essentiality classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

CLASS_NAMES_3 = ["always_essential", "conditional", "non_essential"]
CLASS_NAMES_2 = ["essential", "non_essential"]


def compute_class_weights(labels: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data.
    Inverse frequency assigns a weight that is inversely proportional to how often that class appears in the dataset
    - The goal is to make the signal from the rare class as strong as the signal from the common classes

    Formula: weight[c] = n_total / (n_classes * count[c])
    Use with CrossEntropyLoss to upweight rare classes (always_essential, conditional).

    Args:
        labels: 1D tensor of class indices (0, 1, 2). Should be from training set only.
        n_classes: Number of classes (default 3).

    Returns:
        Tensor of shape (n_classes,) with weight per class.
    """
    counts = torch.bincount(labels, minlength=n_classes)
    total_counts = counts.sum()
    weights = total_counts / (n_classes * (counts + 1))
    return weights


def make_loss_fn(labels: torch.Tensor, n_classes: int = 3, device: torch.device | None = None) -> nn.CrossEntropyLoss:
    """Create CrossEntropyLoss with class weights from training labels.

    Args:
        labels: Training set labels (1D tensor of 0, 1, 2).
        n_classes: Number of classes.
        device: Device to put weights on (e.g. same as model).

    Returns:
        Configured CrossEntropyLoss.
    """
    weights = compute_class_weights(labels, n_classes)
    if device is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss over the epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one evaluation epoch (no gradients). Returns mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    n_classes: int | None = None,
    class_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Collect all logits and labels from a dataloader (for metrics).

If n_classes is provided, also computes metrics (accuracy, F1, per-class AUROC,
        per-class AUPRC) and returns them as a third element.

    Returns:
        If n_classes is None: (logits, labels). logits (N, n_classes), labels (N,).
        If n_classes is set: (logits, labels, metrics) with metrics including
        auroc_<class> and auprc_<class> for each class.
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            logits = model(embeddings)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if n_classes is not None:
        metrics = compute_metrics(
            logits, labels, n_classes=n_classes, class_names=class_names
        )
        return logits, labels, metrics
    return logits, labels


def default_class_names(n_classes: int) -> list[str]:
    """Return default class name list for 2- or 3-class settings."""
    if n_classes == 2:
        return list(CLASS_NAMES_2)
    return list(CLASS_NAMES_3)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int = 3,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, weighted F1, per-class AUROC, per-class AUPRC.

    Args:
        logits: (N, n_classes) raw model output.
        labels: (N,) integer class indices.
        n_classes: Number of classes.
        class_names: Human-readable names for each class index.
            Defaults to 3-class or 2-class essentiality names.

    Returns:
        Dict with accuracy, weighted_f1, auroc_<class>, auprc_<class>, etc.
    """
    if class_names is None:
        class_names = default_class_names(n_classes)

    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    y_true = labels.numpy()

    metrics: dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, preds))
    metrics["weighted_f1"] = float(
        f1_score(y_true, preds, average="weighted", zero_division=0)
    )

    for c in range(n_classes):
        name = class_names[c]
        y_binary = (y_true == c).astype(int)
        if y_binary.sum() == 0:
            metrics[f"auroc_{name}"] = None
            metrics[f"auprc_{name}"] = None
            continue
        try:
            metrics[f"auroc_{name}"] = float(
                roc_auc_score(y_binary, probs[:, c])
            )
        except ValueError:
            metrics[f"auroc_{name}"] = None
        try:
            metrics[f"auprc_{name}"] = float(
                average_precision_score(y_binary, probs[:, c])
            )
        except ValueError:
            metrics[f"auprc_{name}"] = None

    return metrics
