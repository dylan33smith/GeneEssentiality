#!/usr/bin/env python3
"""Verify that regression_dataset.parquet and the dataloader associate fitness with the correct (gene, experiment).

Spot-checks:
1. Sample rows from regression_dataset.parquet; for each, look up (orgId, locusId, expName) in the
   source fitness.parquet and confirm fit values match.
2. Load embedding store and FitnessRegressionDataset; for a few indices, reconstruct (gene_key, fit)
   from the dataset and confirm the same (gene_key, expName) in the parquet has the same fit.

Run from project root: python scripts/verify_regression_data_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data_io import load_fitness, get_processed_dir
from src.model.regression_dataset import (
    load_embedding_store,
    FitnessRegressionDataset,
    _load_and_split_rows,
)


def main() -> int:
    data_dir = get_processed_dir()
    reg_path = data_dir / "regression_dataset.parquet"
    splits_path = data_dir / "mmseqs_splits.csv"
    vocab_path = data_dir / "condition_vocab_regression.json"

    print("=== 1. Load regression parquet and source fitness ===")
    reg = pd.read_parquet(reg_path)
    fitness = load_fitness("processed")

    print("Sample 500 random rows from regression_dataset ...")
    sample = reg.sample(n=min(500, len(reg)), random_state=42)

    merged = sample.merge(
        fitness,
        on=["orgId", "locusId", "expName"],
        how="left",
        suffixes=("_reg", "_src"),
    )
    fit_match = (merged["fit_reg"] - merged["fit_src"]).abs() < 1e-5
    n_match = fit_match.sum()
    n_total = len(merged)
    print(f"  Rows checked: {n_total}. Fit match (reg vs source fitness): {n_match}. Mismatch: {n_total - n_match}")

    if n_match != n_total:
        bad = merged[~fit_match][["orgId", "locusId", "expName", "fit_reg", "fit_src"]].head()
        print("  Example mismatches:")
        print(bad)
        return 1
    print("  OK: regression_dataset fit values match source fitness.parquet for sampled rows.\n")

    print("=== 2. Dataloader alignment: same row => same gene_key, fit ===")
    import json
    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_sizes = vocab["sizes"]
    emb_store = load_embedding_store(data_dir / "ProtLM_embeddings_layer8")
    split_dfs = _load_and_split_rows(reg_path, splits_path)
    ds = FitnessRegressionDataset(rows=split_dfs["train"], embedding_store=emb_store, vocab_sizes=vocab_sizes)

    idx_to_gene_key = split_dfs["train"].reset_index(drop=True)["gene_key"]
    idx_to_fit = split_dfs["train"].reset_index(drop=True)["fit"]

    indices_to_check = [0, 100, len(ds) // 2, len(ds) - 1]
    for idx in indices_to_check:
        x, y = ds[idx]
        expected_gene_key = idx_to_gene_key.iloc[idx]
        expected_fit = idx_to_fit.iloc[idx]
        assert abs(y.item() - expected_fit) < 1e-5, f"idx {idx}: fit mismatch {y.item()} vs {expected_fit}"
        emb_idx = ds._gene_idx[idx]
        reverse_lookup = {v: k for k, v in emb_store.gene_key_to_idx.items()}
        actual_gene_key = reverse_lookup.get(emb_idx, "?")
        assert actual_gene_key == expected_gene_key, f"idx {idx}: gene_key mismatch {actual_gene_key} vs {expected_gene_key}"
        print(f"  idx {idx}: gene_key={expected_gene_key}, fit={expected_fit:.4f} -> dataset returns fit={y.item():.4f} OK")
    print("  OK: Dataset __getitem__ returns the same gene (via embedding index) and fit as the parquet row.\n")

    print("=== 3. Embedding order: group_labels[i] matches embeddings[i] in .pt ===")
    pt_path = sorted((data_dir / "ProtLM_embeddings_layer8").glob("*.pt"))[0]
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    for i in [0, 100]:
        gk = data["group_labels"][i]
        emb_pt = data["embeddings"][i]
        j = emb_store.gene_key_to_idx[gk]
        emb_store_vec = emb_store.tensor[j]
        diff = (emb_pt.float() - emb_store_vec).abs().max().item()
        print(f"  {gk}: max diff store vs .pt = {diff:.6f}")
        assert diff < 1e-5, "Embedding store row should match .pt file"
    print("  OK: Embedding store indexing matches per-file (group_labels, embeddings) order.\n")

    print("All alignment checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
