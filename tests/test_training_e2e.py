"""End-to-end model training smoke tests with synthetic data."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.network import EssentialityMLP
from src.model.train import (
    collect_predictions,
    compute_metrics,
    eval_epoch,
    make_loss_fn,
    train_epoch,
)

DIM = 16
N_SAMPLES = 120
N_CLASSES = 3


def _make_synthetic_loader(n: int = N_SAMPLES, dim: int = DIM, batch_size: int = 32) -> DataLoader:
    x = torch.randn(n, dim)
    y = torch.tensor([i % N_CLASSES for i in range(n)])
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def test_training_loop_runs():
    model = EssentialityMLP(input_dim=DIM, hidden_dims=[16, 8])
    device = torch.device("cpu")
    loader = _make_synthetic_loader()
    labels = torch.tensor([i % N_CLASSES for i in range(N_SAMPLES)])
    loss_fn = make_loss_fn(labels, n_classes=N_CLASSES, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loss = train_epoch(model, loader, loss_fn, optimizer, device)
    val_loss = eval_epoch(model, loader, loss_fn, device)

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)


def test_training_loss_finite():
    model = EssentialityMLP(input_dim=DIM, hidden_dims=[16, 8])
    device = torch.device("cpu")
    loader = _make_synthetic_loader()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(5):
        loss = train_epoch(model, loader, loss_fn, optimizer, device)
        assert math.isfinite(loss), f"Loss is not finite: {loss}"


def test_model_saves_and_loads(tmp_path: Path):
    model = EssentialityMLP(input_dim=DIM, hidden_dims=[16, 8])
    device = torch.device("cpu")
    x = torch.randn(30, DIM)
    y = torch.tensor([i % N_CLASSES for i in range(30)])
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)

    checkpoint = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint)

    logits_before, _ = collect_predictions(model, loader, device)

    model2 = EssentialityMLP(input_dim=DIM, hidden_dims=[16, 8])
    model2.load_state_dict(torch.load(checkpoint, map_location=device))
    logits_after, _ = collect_predictions(model2, loader, device)

    assert torch.allclose(logits_before, logits_after, atol=1e-6)


def test_compute_metrics_on_predictions():
    model = EssentialityMLP(input_dim=DIM, hidden_dims=[16, 8])
    device = torch.device("cpu")
    loader = _make_synthetic_loader()

    logits, labels = collect_predictions(model, loader, device)
    m = compute_metrics(logits, labels, n_classes=N_CLASSES)

    assert 0.0 <= m["accuracy"] <= 1.0
    assert "weighted_f1" in m
    assert "auprc_always_essential" in m


def test_early_stopping_triggers():
    """Simulate early stopping: use a frozen model (no optimizer step) so val loss never improves."""
    model = EssentialityMLP(input_dim=DIM, hidden_dims=[16, 8])
    device = torch.device("cpu")
    loader = _make_synthetic_loader()
    loss_fn = nn.CrossEntropyLoss()

    patience = 3
    best_loss = eval_epoch(model, loader, loss_fn, device)
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(20):
        val_loss = eval_epoch(model, loader, loss_fn, device)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            stopped_early = True
            break

    assert stopped_early, "Expected early stopping to trigger when model is not training"
