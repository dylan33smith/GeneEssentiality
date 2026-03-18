"""PyTorch Dataset and DataLoader for fitness regression.

Combines ProteomeLM gene embeddings (1152-dim) with nutrient-profile one-hot
vectors (530-dim) to predict per-gene per-experiment fitness scores.

The design keeps memory low (~1.5 GB total) by storing compact index arrays
and looking up the full embedding + one-hot on the fly in ``__getitem__``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_io import get_data_dir

logger = logging.getLogger(__name__)

CONDITION_COLS = ("media", "condition_1", "condition_2", "condition_3")
IDX_COLS = ("media_idx", "cond1_idx", "cond2_idx", "cond3_idx")


class EmbeddingStore(NamedTuple):
    """Pre-loaded embedding tensor and gene_key-to-row-index mapping."""

    tensor: torch.Tensor  # [N_genes, 1152], float32
    gene_key_to_idx: dict[str, int]


def load_embedding_store(
    embedding_dir: Path | str,
) -> EmbeddingStore:
    """Load all per-organism .pt files into a single tensor and index dict.

    Args:
        embedding_dir: Directory containing ``*_proteomelm.pt`` files, each
            with keys ``embeddings`` (Tensor [N, 1152]) and ``group_labels``
            (list of gene_key strings).

    Returns:
        EmbeddingStore with a stacked tensor and a gene_key -> row index dict.
    """
    embedding_dir = Path(embedding_dir)
    pt_files = sorted(embedding_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {embedding_dir}")

    all_emb: list[torch.Tensor] = []
    gene_key_to_idx: dict[str, int] = {}
    offset = 0

    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        emb = data["embeddings"]
        labels = data["group_labels"]
        for i, gk in enumerate(labels):
            gene_key_to_idx[gk] = offset + i
        all_emb.append(emb)
        offset += len(labels)

    tensor = torch.cat(all_emb, dim=0).float()
    logger.info(
        "Loaded embedding store: %d gene_keys, tensor %s",
        len(gene_key_to_idx),
        tuple(tensor.shape),
    )
    return EmbeddingStore(tensor=tensor, gene_key_to_idx=gene_key_to_idx)


class FitnessRegressionDataset(Dataset):
    """Dataset of (embedding || nutrient_onehot, fitness) pairs.

    Each sample concatenates a 1152-dim gene embedding with a 530-dim
    nutrient-profile one-hot to produce a 1682-dim input vector, paired
    with a scalar fitness target.

    The embedding tensor and gene_key_to_idx dict are passed in (shared
    across train/val/test) to avoid loading .pt files multiple times.
    Nutrient one-hot vectors are built on the fly from four int16 indices.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        embedding_store: EmbeddingStore,
        vocab_sizes: dict[str, int],
    ) -> None:
        """Build dataset from pre-filtered rows.

        Args:
            rows: DataFrame with columns gene_key, media_idx, cond1_idx,
                cond2_idx, cond3_idx, fit, t. Must already be filtered to the
                desired split.
            embedding_store: Shared EmbeddingStore (tensor + dict).
            vocab_sizes: Dict mapping each condition column name to its vocab
                size (e.g. ``{"media": 113, ...}``).
        """
        emb_tensor, gk_to_idx = embedding_store

        gene_idx = rows["gene_key"].map(gk_to_idx)
        if gene_idx.isna().any():
            n_miss = int(gene_idx.isna().sum())
            raise ValueError(
                f"{n_miss} gene_keys in rows have no embedding. "
                "Filter rows to genes present in the embedding store."
            )

        self._emb_tensor = emb_tensor
        self._gene_idx = gene_idx.values.astype(np.int32)
        self._media_idx = rows["media_idx"].values.astype(np.int16)
        self._cond1_idx = rows["cond1_idx"].values.astype(np.int16)
        self._cond2_idx = rows["cond2_idx"].values.astype(np.int16)
        self._cond3_idx = rows["cond3_idx"].values.astype(np.int16)
        self._fit = rows["fit"].values.astype(np.float32)
        self._t = rows["t"].values.astype(np.float32)

        self._sizes = [vocab_sizes[c] for c in CONDITION_COLS]
        self._onehot_dim = sum(self._sizes)

    def __len__(self) -> int:
        return len(self._fit)

    @property
    def input_dim(self) -> int:
        """Total input dimension (embedding_dim + onehot_dim)."""
        return self._emb_tensor.shape[1] + self._onehot_dim

    @property
    def embedding_dim(self) -> int:
        return self._emb_tensor.shape[1]

    @property
    def onehot_dim(self) -> int:
        return self._onehot_dim

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._emb_tensor[self._gene_idx[idx]]

        onehot = torch.zeros(self._onehot_dim, dtype=torch.float32)
        offset = 0
        for size, val in zip(
            self._sizes,
            [
                self._media_idx[idx],
                self._cond1_idx[idx],
                self._cond2_idx[idx],
                self._cond3_idx[idx],
            ],
        ):
            onehot[offset + int(val)] = 1.0
            offset += size

        x = torch.cat([emb, onehot])
        y = torch.tensor([self._fit[idx]], dtype=torch.float32)
        return x, y


def _load_and_split_rows(
    regression_parquet: Path,
    splits_csv: Path,
) -> dict[str, pd.DataFrame]:
    """Load regression parquet, join splits, return dict of per-split DataFrames."""
    logger.info("Loading regression parquet %s ...", regression_parquet)
    reg = pd.read_parquet(regression_parquet)
    logger.info("  rows: %d", len(reg))

    logger.info("Loading splits from %s ...", splits_csv)
    splits = pd.read_csv(splits_csv, usecols=["gene_key", "split"])
    reg = reg.merge(splits, on="gene_key", how="inner")
    logger.info("  rows after split join: %d", len(reg))

    result: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        subset = reg[reg["split"] == split_name].reset_index(drop=True)
        result[split_name] = subset
        logger.info("  %s: %d rows, %d genes", split_name, len(subset), subset["gene_key"].nunique())
    return result


def create_regression_dataloaders(
    batch_size: int = 2048,
    num_workers: int = 4,
    subset: str = "processed",
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test DataLoaders for fitness regression.

    Loads the embedding store once and shares it across all three datasets.

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader worker processes.
        subset: Data subset directory name ("processed" or "mvp").
        pin_memory: Pin memory for GPU transfer.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = get_data_dir() / subset

    emb_store = load_embedding_store(data_dir / "ProtLM_embeddings_layer8")

    vocab_path = data_dir / "condition_vocab_regression.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_sizes: dict[str, int] = vocab["sizes"]
    logger.info("Vocab sizes: %s, total_onehot_dim: %d", vocab_sizes, sum(vocab_sizes.values()))

    split_dfs = _load_and_split_rows(
        regression_parquet=data_dir / "regression_dataset.parquet",
        splits_csv=data_dir / "mmseqs_splits.csv",
    )

    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        ds = FitnessRegressionDataset(
            rows=split_dfs[split_name],
            embedding_store=emb_store,
            vocab_sizes=vocab_sizes,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=(split_name == "train"),
        )
        logger.info(
            "  %s loader: %d batches (batch_size=%d)",
            split_name,
            len(loaders[split_name]),
            batch_size,
        )

    return loaders["train"], loaders["val"], loaders["test"]
