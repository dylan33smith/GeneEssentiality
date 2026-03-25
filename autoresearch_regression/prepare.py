"""Fixed data loaders and evaluation (not edited by the autoresearch agent).

Run from repository root so ``src`` imports resolve, e.g.:

    python -m autoresearch_regression.experiments.exp01_baseline
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model.regression_dataset import (
    FitnessRegressionDataset,
    create_regression_dataloaders,
    load_regression_split_dataframes,
)
from src.model.regression_metrics import mean_within_gene_spearman, val_rmse

from .early_stop import EarlyStopper, early_stopper_from_env

logger = logging.getLogger(__name__)

GENE_EMBED_DIM = 1152


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


# ---------------------------------------------------------------------------
# Gene-index-aware loader (for ranking experiments exp08, exp09)
# ---------------------------------------------------------------------------

class _GeneIdxDataset(Dataset):
    """Wraps FitnessRegressionDataset to also return a per-row integer gene id."""

    def __init__(self, inner: FitnessRegressionDataset, gene_int_ids: np.ndarray) -> None:
        self._inner = inner
        self._gene_int_ids = gene_int_ids

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self._inner[idx]
        gid = torch.tensor(self._gene_int_ids[idx], dtype=torch.long)
        return x, y, gid

    @property
    def input_dim(self) -> int:
        return self._inner.input_dim


def build_loaders_with_gene_idx() -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """Like build_loaders_and_val_frame but train batches yield (x, y, gene_idx).

    ``gene_idx`` is a dense integer id (0 .. n_unique_genes-1) for grouping
    within a batch.  Val/test loaders are standard (x, y) — evaluation uses
    ``evaluate()`` which aligns via ``val_df``.
    """
    import json

    from src.data_io import get_data_dir
    from src.model.regression_dataset import _load_and_split_rows, load_embedding_store

    cfg = harness_config()
    data_dir = get_data_dir() / cfg["subset"]

    emb_store = load_embedding_store(data_dir / "ProtLM_embeddings_layer8")
    vocab_path = data_dir / "condition_vocab_regression.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_sizes: dict[str, int] = vocab["sizes"]

    split_dfs = _load_and_split_rows(
        regression_parquet=data_dir / "regression_dataset.parquet",
        splits_csv=data_dir / "mmseqs_splits.csv",
    )

    logger.info("Building gene_key -> int mapping ...")
    all_gene_keys = pd.concat([split_dfs[s]["gene_key"] for s in ("train", "val", "test")]).unique()
    gk_to_int: dict[str, int] = {gk: i for i, gk in enumerate(sorted(all_gene_keys))}
    logger.info("  %d unique genes", len(gk_to_int))

    logger.info("Constructing train dataset (%d rows) ...", len(split_dfs["train"]))
    train_ds_inner = FitnessRegressionDataset(split_dfs["train"], emb_store, vocab_sizes)
    train_gene_ints = split_dfs["train"]["gene_key"].map(gk_to_int).values.astype(np.int64)
    train_ds = _GeneIdxDataset(train_ds_inner, train_gene_ints)
    logger.info("  train dataset ready (input_dim=%d)", train_ds.input_dim)

    val_ds = FitnessRegressionDataset(split_dfs["val"], emb_store, vocab_sizes)
    test_ds = FitnessRegressionDataset(split_dfs["test"], emb_store, vocab_sizes)
    logger.info("  val/test datasets ready")

    bs = cfg["batch_size"]
    nw = cfg["num_workers"]
    logger.info("Creating DataLoaders (bs=%d, workers=%d) ...", bs, nw)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=True, persistent_workers=nw > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=True, persistent_workers=nw > 0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw,
                             pin_memory=True, persistent_workers=nw > 0)
    logger.info("  loaders created: train=%d val=%d test=%d batches",
                len(train_loader), len(val_loader), len(test_loader))

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
    if isinstance(model, nn.Module):
        model.eval()
    preds: list[np.ndarray] = []
    for batch in val_loader:
        x_batch = batch[0].to(device)
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


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print harness-standard metric lines (for grep and run_all.py)."""
    print(f"val_rmse:          {metrics['val_rmse']:.6f}")
    print(f"mean_within_gene_spearman: {metrics['mean_within_gene_spearman']:.6f}")
    print(f"n_genes_used_for_spearman: {metrics['n_genes_used_for_spearman']}")
    print(f"n_genes_single_row: {metrics['n_genes_single_row']}")
    print(f"n_genes_nan_rho: {metrics['n_genes_nan_rho']}")


# Early stopping: implementation in early_stop.py; re-exported here for imports
# from autoresearch_regression.prepare import early_stopper_from_env

# ---------------------------------------------------------------------------
# Per-epoch curve logging
# ---------------------------------------------------------------------------

CURVES_DIR = Path(__file__).resolve().parent / "curves"


class EpochLog:
    """Collects per-epoch metrics and saves to JSON for plotting."""

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.rows: list[dict[str, Any]] = []

    def record(self, epoch: int, **kwargs: float) -> None:
        row = {"epoch": epoch}
        row.update(kwargs)
        self.rows.append(row)

    def save(self, final_metrics: dict[str, Any] | None = None) -> Path:
        import json as _json

        CURVES_DIR.mkdir(exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "config": {
                "data_subset": os.environ.get("DATA_SUBSET", "processed"),
                "epochs_max": os.environ.get("AUTORESEARCH_EPOCHS", "8"),
                "batch_size": os.environ.get("BATCH_SIZE", "2048"),
                "early_stop_patience": os.environ.get("EARLY_STOP_PATIENCE", "3"),
                "early_stop_min_delta": os.environ.get("EARLY_STOP_MIN_DELTA", "1e-4"),
            },
            "epoch_metrics": self.rows,
        }
        if final_metrics is not None:
            out["final_metrics"] = final_metrics
        path = CURVES_DIR / f"{self.experiment_name}.json"
        path.write_text(_json.dumps(out, indent=2))
        logger.info("Saved epoch curves to %s", path)
        return path
