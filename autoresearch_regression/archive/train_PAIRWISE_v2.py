"""Experiment: Pairwise ranking loss - directly optimize for rank ordering."""

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


class PairwiseMLP(nn.Module):
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


def pairwise_ranking_loss(pred, margin=0.1):
    """Compute pairwise ranking loss where we want higher predictions for higher fitness."""
    # pred shape: (batch_size,)
    # For simplicity, compare each sample to the batch mean
    mean_pred = pred.mean()
    # Samples above mean should have higher predictions than samples below mean
    above_mask = pred > mean_pred
    below_mask = ~above_mask
    
    n_above = above_mask.sum()
    n_below = below_mask.sum()
    
    if n_above == 0 or n_below == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # Sample some pairs
    samples_per_side = min(32, n_above, n_below)
    
    above_indices = torch.where(above_mask)[0][:samples_per_side]
    below_indices = torch.where(below_mask)[0][:samples_per_side]
    
    pred_above = pred[above_indices]
    pred_below = pred[below_indices]
    
    # Want pred_above > pred_below
    # Use ranking loss: max(0, margin - (pred_above - pred_below))^2
    diff = pred_above.unsqueeze(1) - pred_below.unsqueeze(0)
    loss = F.margin_ranking_loss(pred_above.unsqueeze(1), pred_below.unsqueeze(0), 
                                  torch.ones_like(diff), margin=margin)
    return loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "10"))
    lr = float(os.environ.get("LR", "5e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    margin = float(os.environ.get("MARGIN", "0.1"))
    
    logger.info("Config: epochs=%d lr=%.4f margin=%.2f", epochs, lr, margin)
    
    train_loader, val_loader, test_loader, val_df = build_loaders_and_val_frame()
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))
    
    input_dim = 1152 + 225
    model = PairwiseMLP(input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    except TypeError:
        scheduler = None
        logger.warning("scheduler not supported")
    
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
            pred = model(x).squeeze()
            
            loss = pairwise_ranking_loss(pred, margin=margin)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        metrics = evaluate(model, val_loader, device, val_df)
        val_rmse = metrics["val_rmse"]
        val_spearman = metrics["mean_within_gene_spearman"]
        
        logger.info("epoch %d/%d train_loss=%.6f val_rmse=%.6f val_spearman=%.6f",
                   epoch + 1, epochs, train_loss, val_rmse, val_spearman)
        
        if scheduler is not None:
            scheduler.step(val_rmse)
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
        if val_spearman > best_spearman:
            best_spearman = val_spearman

    metrics = evaluate(model, val_loader, device, val_df)
    print(f"val_rmse:          {metrics['val_rmse']:.6f}")
    print(f"mean_within_gene_spearman: {metrics['mean_within_gene_spearman']:.6f}")
    print(f"n_genes_used_for_spearman: {metrics['n_genes_used_for_spearman']}")
    print(f"n_genes_single_row: {metrics['n_genes_single_row']}")
    print(f"n_genes_nan_rho: {metrics['n_genes_nan_rho']}")
    print(f"best_rmse:         {best_rmse:.6f}")
    print(f"best_spearman:     {best_spearman:.6f}")


if __name__ == "__main__":
    train()
