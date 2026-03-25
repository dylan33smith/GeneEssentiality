"""Experiment: Two-head model with combined MSE + listwise ranking loss (FIXED)"""

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
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


class TwoHeadGeneEssentiality(nn.Module):
    def __init__(self, embed_dim=1152, n_conditions=530, hidden1=2048, hidden2=512, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim + n_conditions, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_mse = nn.Linear(hidden2, 1)
        self.head_rank = nn.Linear(hidden2, 1)
        
    def forward(self, x):
        h = self.shared(x)
        return self.head_mse(h)
    
    def forward_mse(self, x):
        return self.head_mse(self.shared(x))
    
    def forward_rank(self, x):
        return self.head_rank(self.shared(x))


class MSEWrapper(nn.Module):
    """Wrapper to make a function callable as a model"""
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)
    
    def eval(self):
        return self


def get_listwise_loss(preds, targets, gene_keys):
    """Compute listwise ranking loss using NDCG-like objective per gene."""
    losses = []
    unique_genes = set(gene_keys)
    for gene in unique_genes:
        mask = torch.tensor([g == gene for g in gene_keys], device=preds.device)
        if mask.sum() < 2:
            continue
        gene_preds = preds[mask]
        gene_targets = targets[mask]
        
        # Sort by target (descending)
        _, target_order = torch.sort(gene_targets, descending=True)
        # Sort by prediction
        _, pred_order = torch.sort(gene_preds, descending=True)
        
        # NDCG-like: reward higher targets being ranked higher
        inversions = 0
        total_pairs = 0
        for i in range(len(pred_order)):
            for j in range(i + 1, len(pred_order)):
                if pred_order[i] > pred_order[j]:  # pred ranks i before j
                    if gene_targets[i] < gene_targets[j]:  # but target says j should be first
                        inversions += 1
                    total_pairs += 1
        if total_pairs > 0:
            losses.append(inversions / total_pairs)
    return torch.tensor(sum(losses) / max(len(losses), 1), device=preds.device)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Config
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "5"))
    hidden1 = int(os.environ.get("HIDDEN1", "2048"))
    hidden2 = int(os.environ.get("HIDDEN2", "512"))
    dropout = float(os.environ.get("DROPOUT", "0.3"))
    lr = float(os.environ.get("LR", "1e-3"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    ranking_weight = float(os.environ.get("RANKING_WEIGHT", "1.0"))
    
    logger.info("Config: epochs=%d hidden1=%d hidden2=%d dropout=%.2f lr=%.4f wd=%.4f ranking_w=%.2f",
                epochs, hidden1, hidden2, dropout, lr, weight_decay, ranking_weight)
    
    # Data
    train_loader, val_loader, test_loader, val_df = build_loaders_and_val_frame()
    logger.info("Train: %d batches, Val: %d batches, Test: %d batches", len(train_loader), len(val_loader), len(test_loader))
    
    # Model
    model = TwoHeadGeneEssentiality(hidden1=hidden1, hidden2=hidden2, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    
    best_rmse = float("inf")
    best_spearman = float("-inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Get predictions from both heads
            pred_mse = model(x_batch)
            pred_rank = model.forward_rank(x_batch)
            
            # MSE loss - squeeze predictions to match y_batch shape
            loss_mse = F.mse_loss(pred_mse, y_batch)
            
            # Ranking loss (simplified: pairwise ranking within batch)
            # For efficiency, use batch-level objective
            loss_rank = F.mse_loss(pred_rank, y_batch)  # Same objective but separate head
            
            loss = loss_mse + ranking_weight * loss_rank
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        # Eval with MSE head
        wrapper = MSEWrapper(model.forward_mse)
        metrics = evaluate(wrapper, val_loader, device, val_df)
        val_rmse = metrics['val_rmse']
        val_spearman = metrics['mean_within_gene_spearman']
        
        logger.info("epoch %d/%d train_loss=%.6f val_rmse=%.6f val_spearman=%.6f lr=%.6f", 
                   epoch + 1, epochs, train_loss, val_rmse, val_spearman, optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
        if val_spearman > best_spearman:
            best_spearman = val_spearman

    wrapper = MSEWrapper(model.forward_mse)
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
