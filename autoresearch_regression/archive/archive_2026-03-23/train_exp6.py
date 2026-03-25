"""Experiment: Log-transformed target fitness"""

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autoresearch_regression.prepare import build_loaders_and_val_frame, evaluate
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _default_mlp(input_dim: int) -> nn.Module:
    hidden1 = int(os.environ.get("HIDDEN1", "2048"))
    hidden2 = int(os.environ.get("HIDDEN2", "512"))
    dropout = float(os.environ.get("DROPOUT", "0.3"))
    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden2, 1),
    )


def train() -> None:
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "15"))
    lr = float(os.environ.get("LR", "5e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _test_loader, val_df = build_loaders_and_val_frame()

    batch_x, _ = next(iter(train_loader))
    input_dim = int(batch_x.shape[1])
    model = _default_mlp(input_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_rmse = float("inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            # Log-transform the target (add small epsilon for safety)
            y_log = torch.log(y + 1e-6)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y_log)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        # For evaluation, we need to compare exp(pred) with y
        metrics = evaluate_log_target(model, val_loader, device, val_df)
        val_rmse = metrics['val_rmse']
        
        logger.info("epoch %d/%d train_mse=%.6f val_rmse=%.6f lr=%.6f", 
                   epoch + 1, epochs, train_loss, val_rmse, optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse

    metrics = evaluate_log_target(model, val_loader, device, val_df)
    print(f"val_rmse:          {metrics['val_rmse']:.6f}")
    print(f"mean_within_gene_spearman: {metrics['mean_within_gene_spearman']:.6f}")
    print(f"n_genes_used_for_spearman: {metrics['n_genes_used_for_spearman']}")
    print(f"n_genes_single_row: {metrics['n_genes_single_row']}")
    print(f"n_genes_nan_rho: {metrics['n_genes_nan_rho']}")


def evaluate_log_target(model_fn, val_loader, device, val_df):
    """Evaluate model trained on log targets, but report metrics on original scale."""
    model_fn.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model_fn(x)
            # Exponentiate predictions back to original scale
            pred_exp = torch.exp(pred)
            all_preds.append(pred_exp.cpu())
            all_targets.append(y)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Compute RMSE on original scale
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))
    
    # Spearman is rank-based, so log/exp doesn't matter
    # But compute on original scale for consistency
    val_df['pred'] = all_preds.numpy()
    val_df['target'] = all_targets.numpy()
    
    # Compute per-gene Spearman
    gene_spearmans = []
    n_nan = 0
    n_single = 0
    for gene_id, group in val_df.groupby('gene_key'):
        if len(group) < 2:
            n_single += 1
            continue
        pred = group['pred'].values
        target = group['target'].values
        try:
            rho, _ = scipy.stats.spearmanr(pred, target)
            if not np.isnan(rho):
                gene_spearmans.append(rho)
            else:
                n_nan += 1
        except:
            n_nan += 1
    
    import numpy as np
    import scipy.stats
    
    return {
        'val_rmse': rmse.item(),
        'mean_within_gene_spearman': float(np.mean(gene_spearmans)) if gene_spearmans else 0.0,
        'n_genes_used_for_spearman': len(gene_spearmans),
        'n_genes_single_row': n_single,
        'n_genes_nan_rho': n_nan,
    }


if __name__ == "__main__":
    train()
