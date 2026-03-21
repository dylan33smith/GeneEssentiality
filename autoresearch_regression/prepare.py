"""Fixed data loaders and evaluation (not edited by the autoresearch agent).

Run from repository root so ``src`` imports resolve, e.g.:

    python -m autoresearch_regression.train
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.regression_dataset import create_regression_dataloaders, load_regression_split_dataframes
from src.model.regression_metrics import mean_within_gene_spearman, val_rmse

logger = logging.getLogger(__name__)


def harness_config() -> dict[str, Any]:
    """Env-tunable defaults (human-edited; agent does not change this file)."""
    return {
        "subset": os.environ.get("DATA_SUBSET", "processed"),
        "batch_size": int(os.environ.get("BATCH_SIZE", "2048")),
        "num_workers": int(os.environ.get("NUM_WORKERS", "4")),
    }


def build_loaders_and_val_frame() -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """Return train/val/test loaders and the val split DataFrame (aligned row order with val loader)."""
    cfg = harness_config()
    split_dfs = load_regression_split_dataframes(subset=cfg["subset"])
    train_loader, val_loader, test_loader = create_regression_dataloaders(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        subset=cfg["subset"],
        split_dfs=split_dfs,
    )
    val_df = split_dfs["val"].reset_index(drop=True)
    return train_loader, val_loader, test_loader, val_df


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    val_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute validation RMSE and mean within-gene Spearman (primary / secondary harness metrics)."""
    model.eval()
    preds: list[np.ndarray] = []
    for x_batch, _y_batch in val_loader:
        x_batch = x_batch.to(device)
        out = model(x_batch)
        preds.append(out.detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0).ravel()
    y_true = val_df["fit"].to_numpy(dtype=np.float64)
    gene_keys = val_df["gene_key"].to_numpy()
    if len(y_pred) != len(y_true):
        raise RuntimeError(
            f"Val predictions length {len(y_pred)} != val_df rows {len(y_true)} "
            "(check val DataLoader vs val_df alignment)."
        )
    rmse = val_rmse(y_true, y_pred)
    sp = mean_within_gene_spearman(y_true, y_pred, gene_keys)
    return {
        "val_rmse": rmse,
        "mean_within_gene_spearman": sp["mean_rho"],
        "n_genes_used_for_spearman": sp["n_genes_used"],
        "n_genes_single_row": sp["n_genes_single_row"],
        "n_genes_nan_rho": sp["n_genes_nan_rho"],
    }
