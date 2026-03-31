"""exp25 — Residual model + listwise ranking loss + gene-packed batching.

Combines the residual decomposition (gene_mean + condition offset) with
listwise KL ranking pressure via dense within-gene batches.
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
    build_loaders_gene_packed,
    evaluate,
    print_metrics,
)
from autoresearch_regression.experiments._shared import ResidualModel, listwise_ranking_loss

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train() -> dict[str, Any]:
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "8"))
    lr = float(os.environ.get("LR", "5e-4"))
    wd = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    rank_weight = float(os.environ.get("RANK_WEIGHT", "0.15"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _test_loader, val_df = build_loaders_gene_packed()

    sample = next(iter(train_loader))
    input_dim = int(sample[0].shape[1])
    condition_dim = input_dim - GENE_EMBED_DIM
    logger.info("input_dim=%d  gene_embed_dim=%d  condition_dim=%d",
                input_dim, GENE_EMBED_DIM, condition_dim)
    model = ResidualModel(GENE_EMBED_DIM, condition_dim).to(device)
    mse_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stop = early_stopper_from_env()

    elog = EpochLog("exp25_residual_listwise")
    for epoch in range(epochs):
        model.train()
        total_mse, total_rank, n = 0.0, 0.0, 0
        for x, y, gid in train_loader:
            x, y, gid = x.to(device), y.to(device), gid.to(device)
            optimizer.zero_grad()
            pred = model(x)
            mse_loss = mse_fn(pred, y)
            rank_loss = listwise_ranking_loss(pred, y, gid)
            loss = mse_loss + rank_weight * rank_loss
            loss.backward()
            optimizer.step()
            total_mse += mse_loss.item()
            total_rank += rank_loss.item()
            n += 1

        train_mse = total_mse / max(n, 1)
        train_rank = total_rank / max(n, 1)
        metrics = evaluate(model, val_loader, device, val_df)
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info("epoch %d/%d train_mse=%.6f train_rank=%.6f val_rmse=%.6f val_spearman=%.6f lr=%.6f",
                     epoch + 1, epochs, train_mse, train_rank,
                     metrics["val_rmse"], metrics["mean_within_gene_spearman"], cur_lr)
        elog.record(epoch + 1, train_mse=train_mse, train_rank=train_rank,
                    val_rmse=metrics["val_rmse"],
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
