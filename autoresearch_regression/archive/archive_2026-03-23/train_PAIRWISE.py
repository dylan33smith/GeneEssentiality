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


def pairwise_ranking_loss(pred_A, pred_B, margin=0.1):
    """Hinge loss: want pred_A > pred_B."""
    diff = pred_A - pred_B
    return F.hinge_embedding_loss(
        diff.unsqueeze(-1), 
        torch.ones_like(diff), 
        margin=margin
    )


def mine_pairs(y, max_pairs_per_sample=5):
    """Mine pairs (idx_A, idx_B) where y[A] > y[B]."""
    y_flat = y.view(-1)
    n = y_flat.size(0)
    
    # For each sample, find samples with higher fitness
    pos_mask = y_flat.unsqueeze(1) < y_flat.unsqueeze(0)
    
    row_grid = torch.arange(n, device=y.device).unsqueeze(1).expand(n, n)
    col_grid = torch.arange(n, device=y.device).unsqueeze(0).expand(n, n)
    
    # Get all valid pairs first (before subsampling)
    all_A = row_grid[pos_mask].view(-1)
    all_B = col_grid[pos_mask].view(-1)
    
    # Group by row and sample
    idx_A_list, idx_B_list = [], []
    row_counts = pos_mask.sum(dim=1)
    cumsum = torch.cat([torch.tensor([0], device=y.device), row_counts.cumsum(0)])
    
    for i in range(n):
        start, end = cumsum[i].item(), cumsum[i+1].item()
        if end > start:
            row_samples = all_A[start:end]
            col_samples = all_B[start:end]
            if len(row_samples) > max_pairs_per_sample:
                idx = torch.randperm(len(row_samples), device=y.device)[:max_pairs_per_sample]
                row_samples = row_samples[idx]
                col_samples = col_samples[idx]
            idx_A_list.append(row_samples)
            idx_B_list.append(col_samples)
    
    if len(idx_A_list) > 0:
        idx_A = torch.cat(idx_A_list)
        idx_B = torch.cat(idx_B_list)
    else:
        idx_A = torch.tensor([], device=y.device, dtype=torch.long)
        idx_B = torch.tensor([], device=y.device, dtype=torch.long)
    
    return idx_A, idx_B


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
    
    input_dim = 1682
    model = PairwiseMLP(input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1) if hasattr(ReduceLROnPlateau, '__init__') else None
    
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
            
            # Mine pairs
            idx_A, idx_B = mine_pairs(y, max_pairs_per_sample=3)
            
            if len(idx_A) > 0:
                loss = pairwise_ranking_loss(pred[idx_A], pred[idx_B], margin=margin)
            else:
                loss = torch.tensor(0.0, device=device)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        # Use MSE-based evaluation
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
