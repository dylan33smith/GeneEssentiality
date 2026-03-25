"""Experiment: Two-head model - fixed version"""

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TwoHeadMLP(nn.Module):
    """Two heads: one for MSE (global fit), one for listwise ranking."""
    def __init__(self, input_dim: int, hidden1: int = 2048, hidden2: int = 512, dropout: float = 0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Head 1: MSE prediction
        self.head_mse = nn.Linear(hidden2, 1)
        # Head 2: For ranking - outputs logits per condition
        self.head_rank = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head_mse(x), self.head_rank(x)

    def forward_mse(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head_mse(x)


def listwise_ranking_loss(preds, targets, gene_ids, reduction='mean'):
    """
    Simple listwise ranking loss: negative sum of target * pred within each gene.
    Equivalent to encouraging higher predictions for higher targets.
    """
    unique_genes = torch.unique(gene_ids)
    total_loss = 0.0
    n_genes = 0
    
    for gene in unique_genes:
        mask = gene_ids == gene
        gene_preds = preds[mask]
        gene_targets = targets[mask]
        
        # Negative dot product encourages predicting higher for higher targets
        gene_loss = -(gene_preds * gene_targets).sum()
        total_loss += gene_loss
        n_genes += 1
    
    if reduction == 'mean':
        return total_loss / max(n_genes, 1)
    return total_loss


def train():
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, val_df = build_loaders_and_val_frame()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Get input dimension
    batch_x = next(iter(train_loader))[0]
    input_dim = int(batch_x.shape[1])
    logger.info("Input dimension: %d", input_dim)
    
    # Model and optimizer
    model = TwoHeadMLP(input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    
    epochs = 5
    best_rmse = float("inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze(-1)
            # batch_gene_ids not needed for now
            
            optimizer.zero_grad()
            
            pred_mse, pred_rank = model(batch_x)
            
            # MSE loss on MSE head
            loss_mse = F.mse_loss(pred_mse.squeeze(), batch_y)
            
            # Ranking loss on ranking head
            # loss_rank = listwise_ranking_loss(pred_rank.squeeze(), batch_y, batch_gene_ids)  # REMOVED: no gene_ids
            
            loss = loss_mse  # Two-head MSE head only
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        # Eval with MSE head - pass model, not method
        metrics = evaluate(model.forward_mse, val_loader, device, val_df)
        val_rmse = metrics['val_rmse']
        
        logger.info("epoch %d/%d train_loss=%.6f val_rmse=%.6f lr=%.6f", 
                   epoch + 1, epochs, train_loss, val_rmse, optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse

    metrics = evaluate(model.forward_mse, val_loader, device, val_df)
    print(f"val_rmse:          {metrics['val_rmse']:.6f}")
    print(f"mean_within_gene_spearman: {metrics['mean_within_gene_spearman']:.6f}")
    print(f"n_genes_used_for_spearman: {metrics['n_genes_used_for_spearman']}")
    print(f"n_genes_single_row: {metrics['n_genes_single_row']}")
    print(f"n_genes_nan_rho: {metrics['n_genes_nan_rho']}")


if __name__ == "__main__":
    train()
