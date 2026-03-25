"""Experiment: Deeper network (4 hidden layers) for more representational capacity"""

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


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple = (2048, 1024, 512, 256), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def train() -> None:
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "40"))
    lr = float(os.environ.get("LR", "5e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "5e-4"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _test_loader, val_df = build_loaders_and_val_frame()

    batch_x, _ = next(iter(train_loader))
    input_dim = int(batch_x.shape[1])
    model = DeepMLP(input_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    logger.info(f"Model architecture: DeepMLP with hidden dims (2048, 1024, 512, 256)")
    
    best_rmse = float("inf")
    best_spearman = float("-inf")
    
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
        val_spearman = metrics['mean_within_gene_spearman']
        
        logger.info("epoch %d/%d train_mse=%.6f val_rmse=%.6f val_spearman=%.6f lr=%.6f", 
                   epoch + 1, epochs, train_loss, val_rmse, val_spearman, optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
        if val_spearman > best_spearman:
            best_spearman = val_spearman

    logger.info("best_rmse:         %.6f", best_rmse)
    logger.info("best_spearman:     %.6f", best_spearman)


if __name__ == "__main__":
    train()
