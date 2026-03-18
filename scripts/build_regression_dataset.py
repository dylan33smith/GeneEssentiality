#!/usr/bin/env python3
"""Build a gene-experiment regression dataset for fitness prediction.

Merges fitness scores with experiment metadata (media, conditions),
filters to genes that have ProteomeLM embeddings, builds integer-indexed
vocabularies for one-hot encoding at training time, and saves a compact
parquet plus a vocabulary JSON.

Output:
    data/processed/regression_dataset.parquet
    data/processed/condition_vocab_regression.json

Usage:
    python scripts/build_regression_dataset.py
    python scripts/build_regression_dataset.py --subset processed   # default
    python scripts/build_regression_dataset.py --embedding-dir data/processed/ProtLM_embeddings_layer8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_io import load_experiments, load_fitness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITION_COLS = ("media", "condition_1", "condition_2", "condition_3")
IDX_COLS = ("media_idx", "cond1_idx", "cond2_idx", "cond3_idx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build gene-experiment regression dataset"
    )
    parser.add_argument(
        "--subset",
        choices=["processed", "mvp"],
        default="processed",
        help="Data subset (default: processed)",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Directory with ProtLM embedding .pt files. "
        "Default: data/<subset>/ProtLM_embeddings_layer8",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: data/<subset>/",
    )
    args = parser.parse_args()

    data_dir = project_root / "data" / args.subset
    if args.embedding_dir is None:
        args.embedding_dir = data_dir / "ProtLM_embeddings_layer8"
    if args.output_dir is None:
        args.output_dir = data_dir
    return args


def collect_embedding_gene_keys(embedding_dir: Path) -> set[str]:
    """Scan .pt files and return the set of gene_keys that have embeddings."""
    gene_keys: set[str] = set()
    pt_files = sorted(embedding_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {embedding_dir}")
    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        gene_keys.update(data["group_labels"])
    logger.info("Found %d gene_keys across %d .pt files", len(gene_keys), len(pt_files))
    return gene_keys


def build_vocab(series: pd.Series) -> dict[str, int]:
    """Build a string-to-index mapping. Empty string gets index 0."""
    unique_vals = sorted(series.unique(), key=lambda x: (x != "", x))
    return {val: idx for idx, val in enumerate(unique_vals)}


def main() -> int:
    args = parse_args()

    logger.info("Loading fitness (%s) ...", args.subset)
    fitness = load_fitness(args.subset)
    logger.info("  fitness shape: %s", fitness.shape)

    logger.info("Loading experiments (%s) ...", args.subset)
    experiments = load_experiments(args.subset)
    logger.info("  experiments shape: %s", experiments.shape)

    exp_cols = ["orgId", "expName"] + list(CONDITION_COLS)
    missing = [c for c in exp_cols if c not in experiments.columns]
    if missing:
        raise ValueError(f"Experiments table missing columns: {missing}")

    logger.info("Merging fitness with experiment metadata ...")
    df = fitness.merge(experiments[exp_cols], on=["orgId", "expName"], how="left")
    logger.info("  merged shape: %s", df.shape)

    for col in CONDITION_COLS:
        df[col] = df[col].fillna("").astype(str)

    logger.info("Building gene_key ...")
    df["gene_key"] = df["orgId"].str.cat(df["locusId"], sep=":")

    logger.info("Collecting embedding gene_keys from %s ...", args.embedding_dir)
    emb_keys = collect_embedding_gene_keys(args.embedding_dir)

    n_before = len(df)
    df = df[df["gene_key"].isin(emb_keys)]
    n_after = len(df)
    n_dropped = n_before - n_after
    logger.info(
        "Filtered to genes with embeddings: %d -> %d (dropped %d rows, %.1f%%)",
        n_before, n_after, n_dropped, 100 * n_dropped / n_before if n_before else 0,
    )

    logger.info("Building vocabularies ...")
    vocabs: dict[str, dict[str, int]] = {}
    for col in CONDITION_COLS:
        vocabs[col] = build_vocab(df[col])
        logger.info("  %s: %d unique values", col, len(vocabs[col]))

    for col, idx_col in zip(CONDITION_COLS, IDX_COLS):
        df[idx_col] = df[col].map(vocabs[col]).astype(np.int16)

    df["fit"] = df["fit"].astype(np.float32)
    df["t"] = df["t"].astype(np.float32)

    output_cols = [
        "gene_key", "orgId", "locusId", "expName",
        *CONDITION_COLS,
        *IDX_COLS,
        "fit", "t",
    ]
    df = df[list(output_cols)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "regression_dataset.parquet"
    logger.info("Saving parquet to %s ...", parquet_path)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    parquet_mb = parquet_path.stat().st_size / (1024 * 1024)
    logger.info("  parquet size: %.1f MB", parquet_mb)

    vocab_out = {
        col: vocab for col, vocab in vocabs.items()
    }
    sizes = {col: len(vocab) for col, vocab in vocabs.items()}
    vocab_out["sizes"] = sizes
    vocab_out["total_onehot_dim"] = sum(sizes.values())

    vocab_path = args.output_dir / "condition_vocab_regression.json"
    logger.info("Saving vocab to %s ...", vocab_path)
    with open(vocab_path, "w") as f:
        json.dump(vocab_out, f, indent=2, ensure_ascii=False)

    logger.info("=== Summary ===")
    logger.info("  Rows:         %d", len(df))
    logger.info("  Unique genes: %d", df["gene_key"].nunique())
    logger.info("  Unique exps:  %d", df["expName"].nunique())
    logger.info("  Unique orgs:  %d", df["orgId"].nunique())
    logger.info("  Vocab sizes:  %s", sizes)
    logger.info("  Total one-hot dim: %d", vocab_out["total_onehot_dim"])
    logger.info("  fit range:    [%.2f, %.2f]", df["fit"].min(), df["fit"].max())
    logger.info("  fit mean:     %.4f", df["fit"].mean())
    logger.info("  Parquet:      %s (%.1f MB)", parquet_path, parquet_mb)
    logger.info("  Vocab JSON:   %s", vocab_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
