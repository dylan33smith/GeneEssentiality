"""Experiment: Residual/Offset model.

fitness(g, c) = gene_mean(g) + offset(g, c)

Separates "how fit is this gene" from "which conditions does it prefer".
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autoresearch_regression.prepare import build_loaders_and_val_frame, evaluate

logger = logging.getLogger(__name__)


class ResidualGeneModel(nn.Module):
    """Model with separate gene_mean and offset heads."""
    
    def __init__(self, gene_embed_dim=1152, n_conditions=225, hidden1=2048, hidden2=512, dropout=0.3):
        super().__init__()
        
        # Split input
        self.gene_mean_net = nn.Sequential(
            nn.Linear(gene_embed_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
        
        self.offset_net = nn.Sequential(
            nn.Linear(gene_embed_dim + n_conditions, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
        
    def forward(self, x):
        gene_embed = x[:, :1152]  # Separate gene embedding
        cond_onehot = x[:, 1152:]  # And condition
        
        gene_mean = self.gene_mean_net(gene_embed)
        offset = self.offset_net(x)
        
        return (gene_mean + offset).squeeze()
    
    def forward_for_eval(self, x):
        """Return just the combined output."""
        return self.forward(x)


class ResidualWrapper(nn.Module):
    """Wrapper for evaluation."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model.forward_for_eval(x)
    
    def eval(self):
        return self


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "15"))
    lr = float(os.environ.get("LR", "5e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    
    logger.info("Config: epochs=%d lr=%.4f", epochs, lr)
    
    train_loader, val_loader, test_loader, val_df = build_loaders_and_val_frame()
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))
    
    model = ResidualGeneModel().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    
    best_rmse = float("inf")
    best_spearman = float("-inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).squeeze()
            
            optimizer.zero_grad()
            out = model(x).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        train_mse = total_loss / max(n_batches, 1)
        
        # Validation
        wrapper = ResidualWrapper(model)
        metrics = evaluate(wrapper, val_loader, device, val_df)
        val_rmse = metrics["val_rmse"]
        val_spearman = metrics["mean_within_gene_spearman"]
        
        logger.info("epoch %d/%d train_mse=%.6f val_rmse=%.6f val_spearman=%.6f",
                   epoch + 1, epochs, train_mse, val_rmse, val_spearman)
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
        if val_spearman > best_spearman:
            best_spearman = val_spearman

    wrapper = ResidualWrapper(model)
    metrics = evaluate(wrapper, val_loader, device, val_df)
    print(f"val_rmse:          {metrics['val_rmse']:.6f}")
    print(f"mean_within_gene_spearman: {metrics['mean_within_gene_spearman']:.6f}")
    print(f"n_genes_used_for_spearman: {metrics['n_genes_used_for_spearman']}")
    print(f"n_genes_single_row: {metrics['n_genes_single_row']}")
    print(f"n_genes_nan_rho: {metrics['n_genes_nan_rho']}")
    print(f"best_rmse:         {best_rmse:.6f}")
    print(f"best_spearman:     {best_spearman:.6f}")


if __name__ == "__main__":
    train()
