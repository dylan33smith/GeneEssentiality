"""Experiment: LayerNorm + GELU activations"""

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class NormedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 2048, hidden2: int = 512, dropout: float = 0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)


def train() -> None:
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "10"))
    lr = float(os.environ.get("LR", "2e-4"))  # Lower LR for LayerNorm models
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _test_loader, val_df = build_loaders_and_val_frame()

    batch_x, _ = next(iter(train_loader))
    input_dim = int(batch_x.shape[1])
    model = NormedMLP(input_dim).to(device)
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
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        
        metrics = evaluate(model, val_loader, device, val_df)
        val_rmse = metrics['val_rmse']
        
        logger.info("epoch %d/%d train_mse=%.6f val_rmse=%.6f lr=%.6f", 
                   epoch + 1, epochs, train_loss, val_rmse, optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse

    metrics = evaluate(model, val_loader, device, val_df)
    print(f"val_rmse:          {metrics['val_rmse']:.6f}")
    print(f"mean_within_gene_spearman: {metrics['mean_within_gene_spearman']:.6f}")
    print(f"n_genes_used_for_spearman: {metrics['n_genes_used_for_spearman']}")
    print(f"n_genes_single_row: {metrics['n_genes_single_row']}")
    print(f"n_genes_nan_rho: {metrics['n_genes_nan_rho']}")


if __name__ == "__main__":
    train()
