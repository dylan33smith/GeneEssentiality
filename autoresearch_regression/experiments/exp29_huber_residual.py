"""exp29 — Residual model + Huber loss (HUBER_DELTA env-tunable).

Tests Huber on the residual backbone; run with delta=0.1/0.5/1.0 via env
to sweep the robustness threshold.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autoresearch_regression.early_stop import early_stopper_from_env
from autoresearch_regression.prepare import (
    GENE_EMBED_DIM,
    EpochLog,
    build_loaders_and_val_frame,
    evaluate,
    print_metrics,
)
from autoresearch_regression.experiments._shared import ResidualModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train() -> dict[str, Any]:
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "8"))
    lr = float(os.environ.get("LR", "5e-4"))
    wd = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    delta = float(os.environ.get("HUBER_DELTA", "0.5"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _test_loader, val_df = build_loaders_and_val_frame()

    input_dim = int(next(iter(train_loader))[0].shape[1])
    condition_dim = input_dim - GENE_EMBED_DIM
    logger.info("input_dim=%d  gene_embed_dim=%d  condition_dim=%d  huber_delta=%.3f",
                input_dim, GENE_EMBED_DIM, condition_dim, delta)
    model = ResidualModel(GENE_EMBED_DIM, condition_dim).to(device)
    loss_fn = nn.HuberLoss(delta=delta)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stop = early_stopper_from_env()

    elog = EpochLog("exp29_huber_residual")
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1
        train_loss = total_loss / max(n, 1)
        metrics = evaluate(model, val_loader, device, val_df)
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info("epoch %d/%d train_huber=%.6f val_rmse=%.6f val_spearman=%.6f lr=%.6f",
                     epoch + 1, epochs, train_loss,
                     metrics["val_rmse"], metrics["mean_within_gene_spearman"], cur_lr)
        elog.record(epoch + 1, train_loss=train_loss, val_rmse=metrics["val_rmse"],
                    val_spearman=metrics["mean_within_gene_spearman"], lr=cur_lr)
        scheduler.step(metrics["val_rmse"])
        if early_stop.step(metrics["val_rmse"]):
            logger.info("early stop at epoch %d/%d", epoch + 1, epochs)
            break

    metrics = evaluate(model, val_loader, device, val_df)
    print_metrics(metrics)
    elog.save(final_metrics=metrics)
    return metrics


if __name__ == "__main__":
    train()
