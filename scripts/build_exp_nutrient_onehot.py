#!/usr/bin/env python3
"""Build one-hot nutrient profile per experiment and tie to expName (and orgId).

Reads regression_dataset.parquet and condition_vocab_regression.json,
builds the 530-dim one-hot vector (media + condition_1 + condition_2 + condition_3)
for each unique (orgId, expName), and saves a table that can be joined to the
regression dataset on (orgId, expName).

Output:
    data/processed/exp_nutrient_onehot.parquet
        Columns: orgId, expName, plus one-hot columns named by condition value
        (e.g. media_LB, condition_1_Glucose, ...; empty string -> media_empty, etc.).
        (uint8 0/1). Row key: (orgId, expName).

Usage:
    python scripts/build_exp_nutrient_onehot.py
    python scripts/build_exp_nutrient_onehot.py --subset processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
        description="Build experiment-level nutrient profile one-hot and tie to expName"
    )
    parser.add_argument(
        "--subset",
        choices=["processed", "mvp"],
        default="processed",
        help="Data subset (default: processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: data/<subset>/",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / args.subset
    if args.output_dir is None:
        args.output_dir = data_dir
    return args


def onehot_from_indices(
    media_idx: int,
    cond1_idx: int,
    cond2_idx: int,
    cond3_idx: int,
    sizes: dict[str, int],
) -> np.ndarray:
    """Build 530-dim one-hot from the four index columns (order: media, c1, c2, c3)."""
    cols = list(CONDITION_COLS)
    indices = [media_idx, cond1_idx, cond2_idx, cond3_idx]
    parts = []
    for col, idx in zip(cols, indices):
        n = sizes[col]
        vec = np.zeros(n, dtype=np.uint8)
        if 0 <= idx < n:
            vec[idx] = 1
        parts.append(vec)
    return np.concatenate(parts)


def main() -> int:
    args = parse_args()

    reg_path = args.output_dir / "regression_dataset.parquet"
    vocab_path = args.output_dir / "condition_vocab_regression.json"

    if not reg_path.exists():
        logger.error("Regression dataset not found: %s", reg_path)
        return 1
    if not vocab_path.exists():
        logger.error("Vocab not found: %s", vocab_path)
        return 1

    logger.info("Loading regression dataset from %s ...", reg_path)
    reg = pd.read_parquet(reg_path)
    logger.info("  shape: %s", reg.shape)

    logger.info("Loading condition vocab from %s ...", vocab_path)
    with open(vocab_path) as f:
        vocab = json.load(f)
    sizes = vocab["sizes"]
    total_dim = vocab["total_onehot_dim"]
    logger.info("  sizes: %s, total_onehot_dim: %d", sizes, total_dim)

    # Build column names: index -> name per vocab, then "{col}_{name}" (empty -> "empty")
    onehot_column_names: list[str] = []
    for col in CONDITION_COLS:
        str_to_idx = vocab[col]
        idx_to_str = [""] * sizes[col]
        for s, idx in str_to_idx.items():
            idx_to_str[idx] = s
        for name in idx_to_str:
            label = name if name else "empty"
            onehot_column_names.append(f"{col}_{label}")

    # One row per (orgId, expName) with first occurrence's indices
    exp_profile = (
        reg.groupby(["orgId", "expName"], as_index=False)
        .agg(
            media_idx=("media_idx", "first"),
            cond1_idx=("cond1_idx", "first"),
            cond2_idx=("cond2_idx", "first"),
            cond3_idx=("cond3_idx", "first"),
        )
    )

    n_exp = len(exp_profile)
    logger.info("Building one-hot for %d unique (orgId, expName) ...", n_exp)

    onehot = np.zeros((n_exp, total_dim), dtype=np.uint8)
    for i in range(n_exp):
        row = exp_profile.iloc[i]
        onehot[i] = onehot_from_indices(
            int(row["media_idx"]),
            int(row["cond1_idx"]),
            int(row["cond2_idx"]),
            int(row["cond3_idx"]),
            sizes,
        )

    # Table: orgId, expName, plus one-hot columns named by condition value
    key_df = exp_profile[["orgId", "expName"]].copy()
    nutrient_df = pd.DataFrame(
        onehot,
        columns=onehot_column_names,
        dtype=np.uint8,
    )
    out_df = pd.concat([key_df, nutrient_df], axis=1)

    out_path = args.output_dir / "exp_nutrient_onehot.parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %s ...", out_path)
    out_df.to_parquet(out_path, index=False, engine="pyarrow")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("  shape: %s, %.2f MB", out_df.shape, size_mb)

    logger.info("Done. Join to regression_dataset on (orgId, expName) to get one-hot per row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
