"""MLP for essentiality classification from ProteomeLM embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class EssentialityMLP(nn.Module):
    """MLP: embedding -> 3-class logits (always_essential, conditional, non_essential)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        # set default hidden_dims if None (e.g. [512, 256])
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256]
        layers: list[nn.Module] = []

        curr_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(self.hidden_dims[-1], n_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch_size, n_classes)."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)
