"""Tests for src/model/train.py — training utilities and metrics."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.network import EssentialityMLP
from src.model.train import (
    CLASS_NAMES_3,
    collect_predictions,
    compute_class_weights,
    compute_metrics,
    eval_epoch,
    make_loss_fn,
    train_epoch,
)

DIM = 16
N_CLASSES = 3


def _make_tiny_loader(n: int = 60, dim: int = DIM) -> DataLoader:
    x = torch.randn(n, dim)
    y = torch.tensor([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)))
    return DataLoader(TensorDataset(x, y), batch_size=16)


class TestComputeClassWeights:
    def test_balanced(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        w = compute_class_weights(labels, n_classes=3)
        assert w.shape == (3,)
        assert torch.allclose(w[0], w[1], atol=1e-5)
        assert torch.allclose(w[0], w[2], atol=1e-5)

    def test_imbalanced(self):
        labels = torch.tensor([0] + [1] * 10 + [2] * 100)
        w = compute_class_weights(labels, n_classes=3)
        assert w[0] > w[1] > w[2]

    def test_shape(self):
        labels = torch.tensor([0, 1, 2, 0])
        w = compute_class_weights(labels, n_classes=3)
        assert w.shape == (3,)


class TestMakeLossFn:
    def test_returns_crossentropy(self):
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loss_fn = make_loss_fn(labels, n_classes=3)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.weight is not None
        assert loss_fn.weight.shape == (3,)

    def test_device(self):
        labels = torch.tensor([0, 1, 2])
        loss_fn = make_loss_fn(labels, n_classes=3, device=torch.device("cpu"))
        assert loss_fn.weight.device == torch.device("cpu")


class TestTrainEvalEpoch:
    def test_train_epoch_loss_decreases(self):
        model = EssentialityMLP(input_dim=DIM, hidden_dims=[8])
        device = torch.device("cpu")
        loader = _make_tiny_loader()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        loss1 = train_epoch(model, loader, loss_fn, optimizer, device)
        loss2 = train_epoch(model, loader, loss_fn, optimizer, device)
        assert loss2 < loss1

    def test_eval_epoch_no_grad(self):
        model = EssentialityMLP(input_dim=DIM, hidden_dims=[8])
        device = torch.device("cpu")
        loader = _make_tiny_loader()
        loss_fn = nn.CrossEntropyLoss()

        _ = eval_epoch(model, loader, loss_fn, device)
        for param in model.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0


class TestCollectPredictions:
    def test_shapes(self):
        model = EssentialityMLP(input_dim=DIM, hidden_dims=[8])
        device = torch.device("cpu")
        n = 30
        loader = _make_tiny_loader(n=n)

        logits, labels = collect_predictions(model, loader, device)
        assert logits.shape == (n, N_CLASSES)
        assert labels.shape == (n,)


class TestComputeMetrics:
    def test_keys(self):
        logits = torch.randn(30, 3)
        labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
        m = compute_metrics(logits, labels, n_classes=3)
        assert "accuracy" in m
        assert "weighted_f1" in m
        for name in CLASS_NAMES_3:
            assert f"auroc_{name}" in m
            assert f"auprc_{name}" in m

    def test_perfect_predictions(self):
        logits = torch.tensor([
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0],
            [-10.0, -10.0, 10.0],
        ])
        labels = torch.tensor([0, 1, 2])
        m = compute_metrics(logits, labels, n_classes=3)
        assert m["accuracy"] == 1.0

    def test_handles_missing_class(self):
        logits = torch.randn(20, 3)
        labels = torch.tensor([1] * 10 + [2] * 10)
        m = compute_metrics(logits, labels, n_classes=3)
        assert m["auroc_always_essential"] is None
        assert m["auprc_always_essential"] is None

    def test_binary_classes(self):
        logits = torch.randn(40, 2)
        labels = torch.tensor([0] * 20 + [1] * 20)
        m = compute_metrics(logits, labels, n_classes=2)
        assert "accuracy" in m
        assert "auroc_essential" in m
        assert "auprc_essential" in m
        assert "auroc_non_essential" in m

    def test_custom_class_names(self):
        logits = torch.randn(20, 2)
        labels = torch.tensor([0] * 10 + [1] * 10)
        m = compute_metrics(logits, labels, n_classes=2, class_names=["pos", "neg"])
        assert "auroc_pos" in m
        assert "auprc_neg" in m
