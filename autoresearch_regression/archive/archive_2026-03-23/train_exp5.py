"""Experiment: Two-head model with combined MSE + listwise ranking loss"""

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
    batch_size = preds.shape[0]
    total_loss = 0.0
    count = 0
    
    unique_genes = torch.unique(gene_ids)
    for gene in unique_genes:
        mask = gene_ids == gene
        if mask.sum() < 2:
            continue
        gene_preds = preds[mask].squeeze()
        gene_targets = targets[mask].squeeze()
        # Center both
        gene_preds = gene_preds - gene_preds.mean()
        gene_targets = gene_targets - gene_targets.mean()
        # Negative dot product: want predictions aligned with targets
        loss = -torch.dot(gene_preds, gene_targets) / max(gene_preds.shape[0], 1)
        total_loss += loss
        count += 1
    
    return total_loss / max(count, 1)


def train() -> None:
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "10"))
    lr = float(os.environ.get("LR", "5e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    ranking_weight = float(os.environ.get("RANKING_WEIGHT", "1.0"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _test_loader, val_df = build_loaders_and_val_frame()

    batch_x, _ = next(iter(train_loader))
    input_dim = int(batch_x.shape[1])
    model = TwoHeadMLP(input_dim).to(device)
    loss_fn_mse = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_rmse = float("inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            pred_mse, pred_rank = model(x)
            
            # MSE loss
            loss_mse = loss_fn_mse(pred_mse, y)
            
            # Ranking loss - use gene index as gene_id
            gene_ids = torch.arange(x.shape[0]).div(100, rounding_mode='floor').long()  # fake gene grouping
            loss_rank = listwise_ranking_loss(pred_rank, y, gene_ids)
            
            loss = loss_mse + ranking_weight * loss_rank
            loss.backward()
            optimizer.step()
            
            total_loss += loss_mse.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        # Eval with MSE head
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
