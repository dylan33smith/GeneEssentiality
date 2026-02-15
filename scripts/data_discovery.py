#!/usr/bin/env python3
"""
Data discovery script for GeneEssentiality.

Loads all data stages via src.data_io, computes real metrics (row counts,
dtypes, nulls, value ranges, essentiality class counts, experiment distributions),
and optionally inspects embeddings and FASTA. Output is printed for use when
writing notes/05_DATA_ANALYSIS.md. No invented numbers; missing files are reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

from src.data_io import (
    get_data_dir,
    get_mvp_dir,
    load_experiments,
    load_fitness,
    load_genes,
    load_media_experiments,
    load_organisms,
)

ESSENTIALITY_CLASSES = ["always_essential", "conditional", "non_essential", "no_data"]


def numeric_stats(series: pd.Series) -> dict | None:
    if not pd.api.types.is_numeric_dtype(series):
        return None
    s = series.dropna()
    if len(s) == 0:
        return {"min": None, "max": None, "mean": None, "count": 0}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "count": int(len(s)),
    }


def table_schema(df: pd.DataFrame) -> list[dict]:
    return [
        {"column": col, "dtype": str(df[col].dtype)}
        for col in df.columns
    ]


def null_counts(df: pd.DataFrame) -> dict[str, int]:
    return df.isna().sum().astype(int).to_dict()


def run_subset(subset: str) -> dict:
    genes = load_genes(subset)
    experiments = load_experiments(subset)
    fitness = load_fitness(subset)
    organisms = load_organisms(subset)

    n_genes = len(genes)
    n_experiments = len(experiments)
    n_fitness = len(fitness)
    n_organisms = len(organisms)
    unique_genes = fitness.groupby(["orgId", "locusId"]).ngroups if len(fitness) else 0
    unique_experiments = fitness[["orgId", "expName"]].drop_duplicates().shape[0] if len(fitness) else 0

    stats = {
        "subset": subset,
        "row_counts": {
            "genes": n_genes,
            "experiments": n_experiments,
            "fitness": n_fitness,
            "organisms": n_organisms,
        },
        "unique_genes_in_fitness": unique_genes,
        "unique_experiments_in_fitness": unique_experiments,
        "schema": {
            "genes": table_schema(genes),
            "experiments": table_schema(experiments),
            "fitness": table_schema(fitness),
            "organisms": table_schema(organisms),
        },
        "null_counts": {
            "genes": null_counts(genes),
            "experiments": null_counts(experiments),
            "fitness": null_counts(fitness),
            "organisms": null_counts(organisms),
        },
    }

    if "fit" in fitness.columns:
        stats["fitness_fit"] = numeric_stats(fitness["fit"])
    if "t" in fitness.columns:
        stats["fitness_t"] = numeric_stats(fitness["t"])

    if "cor12" in experiments.columns:
        stats["experiments_cor12"] = numeric_stats(experiments["cor12"])
        cor12_ge_02 = (experiments["cor12"] >= 0.2).sum()
        stats["experiments_cor12_pct_ge_0_2"] = 100.0 * cor12_ge_02 / len(experiments) if len(experiments) else 0

    gene_numeric = ["frac_essential_confident", "n_confident_experiments"]
    for col in gene_numeric:
        if col in genes.columns:
            stats[f"genes_{col}"] = numeric_stats(genes[col])

    if "essentiality_class" in genes.columns:
        ec = genes["essentiality_class"].value_counts()
        total = len(genes)
        stats["essentiality_class_counts"] = ec.to_dict()
        stats["essentiality_class_pct"] = {k: 100.0 * ec.get(k, 0) / total for k in ESSENTIALITY_CLASSES}

    if "media" in experiments.columns:
        stats["experiments_by_media"] = experiments["media"].value_counts().head(20).to_dict()
    if "condition_1" in experiments.columns:
        stats["experiments_by_condition_1_top10"] = experiments["condition_1"].value_counts().head(10).to_dict()

    stats["snapshot_genes"] = genes.head(3).to_dict(orient="records") if len(genes) else []
    stats["snapshot_experiments"] = experiments.head(3).to_dict(orient="records") if len(experiments) else []
    stats["snapshot_fitness"] = fitness.head(3).to_dict(orient="records") if len(fitness) else []
    stats["snapshot_organisms"] = organisms.head(3).to_dict(orient="records") if len(organisms) else []

    return stats


def check_embeddings() -> dict:
    path = get_mvp_dir() / "embeddings" / "protein_embeddings.pt"
    if not path.exists():
        return {"present": False, "path": str(path), "error": "File missing"}
    try:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        out = {"present": True, "path": str(path), "keys": list(data.keys())}
        if "inputs_embeds" in data:
            out["inputs_embeds_shape"] = list(data["inputs_embeds"].shape)
        if "group_embeds" in data:
            out["group_embeds_shape"] = list(data["group_embeds"].shape)
        if "group_labels" in data:
            out["group_labels_len"] = len(data["group_labels"])
        return out
    except Exception as e:
        return {"present": True, "path": str(path), "error": str(e)}


def check_fasta() -> dict:
    data_dir = get_mvp_dir()
    out = {}
    proteins = data_dir / "proteins.fasta"
    out["proteins_fasta_exists"] = proteins.exists()
    if proteins.exists():
        try:
            from Bio import SeqIO
            count = sum(1 for _ in SeqIO.parse(proteins, "fasta"))
            out["proteins_fasta_sequences"] = count
        except Exception as e:
            out["proteins_fasta_error"] = str(e)
    org_fastas = data_dir / "organism_fastas"
    out["organism_fastas_dir_exists"] = org_fastas.is_dir() if org_fastas.exists() else False
    if org_fastas.is_dir():
        out["organism_fastas_files"] = len(list(org_fastas.glob("*.fasta")))
    return out


def main() -> int:
    print("=== Data discovery (GeneEssentiality) ===\n")

    results = {}

    for subset in ("processed", "mvp"):
        print(f"--- Subset: {subset} ---")
        try:
            stats = run_subset(subset)
            results[subset] = stats

            rc = stats["row_counts"]
            print(f"Row counts: genes={rc['genes']}, experiments={rc['experiments']}, fitness={rc['fitness']}, organisms={rc['organisms']}")
            if "fitness_fit" in stats and stats["fitness_fit"]:
                print(f"  fit: min={stats['fitness_fit']['min']:.4f}, max={stats['fitness_fit']['max']:.4f}, mean={stats['fitness_fit']['mean']:.4f}")
            if "fitness_t" in stats and stats["fitness_t"]:
                print(f"  t:   min={stats['fitness_t']['min']:.4f}, max={stats['fitness_t']['max']:.4f}, mean={stats['fitness_t']['mean']:.4f}")
            if "experiments_cor12_pct_ge_0_2" in stats:
                print(f"  cor12 >= 0.2: {stats['experiments_cor12_pct_ge_0_2']:.2f}% of experiments")
            if "essentiality_class_counts" in stats:
                print("  Essentiality classes:", stats["essentiality_class_counts"])
            if "essentiality_class_pct" in stats:
                print("  Essentiality %:", stats["essentiality_class_pct"])
            print()
        except FileNotFoundError as e:
            print(f"  MISSING FILE: {e}")
            results[subset] = {"error": str(e)}
        except Exception as e:
            print(f"  ERROR: {e}")
            results[subset] = {"error": str(e)}
            raise

    print("--- Media experiments (Excel) ---")
    try:
        media_df = load_media_experiments()
        print(f"  Loaded: {len(media_df)} rows, columns: {list(media_df.columns)}")
        results["media_experiments"] = {"rows": len(media_df), "columns": list(media_df.columns)}
    except Exception as e:
        print(f"  ERROR or MISSING: {e}")
        results["media_experiments"] = {"error": str(e)}

    print("\n--- Embeddings (protein_embeddings.pt) ---")
    emb = check_embeddings()
    results["embeddings"] = emb
    if emb.get("present"):
        if "error" in emb:
            print(f"  Error loading: {emb['error']}")
        else:
            print(f"  Path: {emb['path']}")
            print(f"  Keys: {emb.get('keys', [])}")
            if "inputs_embeds_shape" in emb:
                print(f"  inputs_embeds.shape: {emb['inputs_embeds_shape']}")
            if "group_labels_len" in emb:
                print(f"  group_labels length: {emb['group_labels_len']}")
    else:
        print(f"  MISSING: {emb.get('path', 'unknown')}")

    print("\n--- FASTA ---")
    fasta_info = check_fasta()
    results["fasta"] = fasta_info
    print(f"  proteins.fasta exists: {fasta_info.get('proteins_fasta_exists')}")
    if fasta_info.get("proteins_fasta_sequences") is not None:
        print(f"  proteins.fasta sequences: {fasta_info['proteins_fasta_sequences']}")
    print(f"  organism_fastas/ exists: {fasta_info.get('organism_fastas_dir_exists')}")
    if fasta_info.get("organism_fastas_files") is not None:
        print(f"  organism_fastas/*.fasta count: {fasta_info['organism_fastas_files']}")

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if pd.isna(obj):
            return None
        return str(obj)

    out_path = Path(__file__).resolve().parent / "data_discovery_output.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nFull metrics and snapshots saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
