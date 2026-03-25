"""Baseline MLP model for gene essentiality prediction."""

import torch
import torch.nn as nn


class GeneEssentialityMLP(nn.Module):
    def __init__(self, input_dim, hidden1=2048, hidden2=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
        
    def forward(self, x):
        return self.net(x).squeeze()

__all__ = ["GeneEssentialityMLP"]
