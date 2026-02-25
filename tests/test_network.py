"""Tests for src/model/network.py — EssentialityMLP architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.network import EssentialityMLP


def test_mlp_output_shape():
    model = EssentialityMLP(input_dim=1152)
    x = torch.randn(4, 1152)
    out = model(x)
    assert out.shape == (4, 3)


def test_mlp_custom_hidden_dims():
    model = EssentialityMLP(input_dim=32, hidden_dims=[64, 32])
    assert model.hidden_dims == [64, 32]
    x = torch.randn(2, 32)
    out = model(x)
    assert out.shape == (2, 3)


def test_mlp_default_hidden_dims():
    model = EssentialityMLP(input_dim=100)
    assert model.hidden_dims == [512, 256]


def test_mlp_dropout_applied():
    model = EssentialityMLP(input_dim=16, hidden_dims=[32], dropout=0.5)
    x = torch.randn(8, 16)

    model.train()
    outputs_train = [model(x) for _ in range(5)]
    differs = any(not torch.equal(outputs_train[0], o) for o in outputs_train[1:])
    assert differs, "Train mode should produce stochastic outputs with dropout"

    model.eval()
    out1 = model(x)
    out2 = model(x)
    assert torch.equal(out1, out2), "Eval mode should be deterministic"


def test_mlp_3d_input_flattened():
    model = EssentialityMLP(input_dim=16, hidden_dims=[8])
    x = torch.randn(3, 1, 16)
    out = model(x)
    assert out.shape == (3, 3)


def test_mlp_custom_n_classes():
    model = EssentialityMLP(input_dim=16, hidden_dims=[8], n_classes=2)
    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 2)


def test_mlp_gradient_flow():
    model = EssentialityMLP(input_dim=16, hidden_dims=[8])
    x = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 1])
    loss_fn = nn.CrossEntropyLoss()

    out = model(x)
    loss = loss_fn(out, labels)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
