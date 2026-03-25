"""exp08 — MSE + within-gene listwise ranking auxiliary loss.

Motivation: MSE alone only matches absolute fitness values. By adding a
ranking loss computed *within* each gene across different conditions, the model
is explicitly incentivised to preserve the correct condition-level ordering
for each gene — directly targeting the within-gene Spearman metric.

Requires ``build_loaders_with_gene_idx()`` from prepare.py so each train
batch includes integer ``gene_idx`` for grouping.
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autoresearch_regression.early_stop import early_stopper_from_env
from autoresearch_regression.prepare import (
    EpochLog,
    build_loaders_with_gene_idx,
    evaluate,
    print_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_mlp(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 2048), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, 1),
    )


def _listwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor,
                           gene_idx: torch.Tensor) -> torch.Tensor:
    """KL-div between softmax(target) and softmax(pred) within each gene group.

    Pre-filters to genes with >= 2 rows to avoid iterating over singletons.
    Returns scalar 0 if no gene has >= 2 rows in the batch.
    """
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    unique_genes, inverse, counts = torch.unique(gene_idx, return_inverse=True, return_counts=True)
    multi_mask = counts >= 2
    if not multi_mask.any():
        return torch.tensor(0.0, device=pred.device)

    multi_genes = unique_genes[multi_mask]
    total_loss = torch.tensor(0.0, device=pred.device)
    n_groups = 0
    for g in multi_genes:
        mask = gene_idx == g
        p = pred[mask]
        t = target[mask]
        log_q = F.log_softmax(p, dim=0)
        p_target = F.softmax(t, dim=0)
        total_loss = total_loss + F.kl_div(log_q, p_target, reduction="batchmean")
        n_groups += 1
    return total_loss / n_groups


def train() -> dict[str, Any]:
    epochs = int(os.environ.get("AUTORESEARCH_EPOCHS", "8"))
    lr = float(os.environ.get("LR", "5e-4"))
    wd = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    rank_weight = float(os.environ.get("RANK_WEIGHT", "0.1"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _test_loader, val_df = build_loaders_with_gene_idx()

    sample = next(iter(train_loader))
    input_dim = int(sample[0].shape[1])
    model = _build_mlp(input_dim).to(device)
    mse_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stop = early_stopper_from_env()

    elog = EpochLog("exp08_ranking_mse")
    for epoch in range(epochs):
        model.train()
        total_mse, total_rank, n = 0.0, 0.0, 0
        for x, y, gid in train_loader:
            x, y, gid = x.to(device), y.to(device), gid.to(device)
            optimizer.zero_grad()
            pred = model(x)
            mse_loss = mse_fn(pred, y)
            rank_loss = _listwise_ranking_loss(pred, y, gid)
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
