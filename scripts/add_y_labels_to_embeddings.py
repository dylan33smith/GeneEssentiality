#!/usr/bin/env python3
"""
Add y label vectors to ProteomeLM embedding .pt files.

Reads .pt files from data/mvp/ProtLM_embedddings/, joins to genes.parquet
to get essentiality_class, maps to integers (0=always_essential, 1=conditional,
2=non_essential, -1=no_data), and saves new .pt files with embeddings,
group_labels, and y to data/mvp/ProtLM_embeddings_with_labels/.

Run from repo root:
  python scripts/add_y_labels_to_embeddings.py
  python scripts/add_y_labels_to_embeddings.py --input data/mvp/ProtLM_embedddings --output data/mvp/ProtLM_embeddings_with_labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import torch

from src.data_io import get_mvp_dir, load_genes

CLASS_TO_INT = {
    "always_essential": 0,
    "conditional": 1,
    "non_essential": 2,
    "no_data": -1,
}


def main() -> int:
    mvp_dir = get_mvp_dir()
    parser = argparse.ArgumentParser(
        description="Add y label vectors to ProteomeLM embedding .pt files",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=mvp_dir / "ProtLM_embedddings",
        help="Input directory containing .pt files (default: data/mvp/ProtLM_embedddings)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=mvp_dir / "ProtLM_embeddings_with_labels",
        help="Output directory for .pt files with y (default: data/mvp/ProtLM_embeddings_with_labels)",
    )
    args = parser.parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading genes.parquet (MVP)...")
    genes = load_genes("mvp")
    genes["gene_key"] = genes["orgId"].astype(str) + ":" + genes["locusId"].astype(str)
    class_to_int = genes["essentiality_class"].map(CLASS_TO_INT)
    genes["y"] = class_to_int
    gene_lookup = genes.set_index("gene_key")["y"].to_dict()

    pt_files = sorted(input_dir.glob("*.pt"))
    if not pt_files:
        print(f"Error: no .pt files found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(pt_files)} .pt files. Processing...")

    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            print(f"  Skip {pt_path.name}: unexpected format (not a dict)", file=sys.stderr)
            continue
        if "embeddings" not in data or "group_labels" not in data:
            print(f"  Skip {pt_path.name}: missing embeddings or group_labels", file=sys.stderr)
            continue

        group_labels = data["group_labels"]
        y_list = []
        for label in group_labels:
            y_val = gene_lookup.get(label, -1)
            if pd.isna(y_val):
                y_val = -1
            y_list.append(int(y_val))

        y_tensor = torch.tensor(y_list, dtype=torch.long)

        result = {
            "embeddings": data["embeddings"],
            "group_labels": group_labels,
            "y": y_tensor,
        }
        out_path = output_dir / pt_path.name
        torch.save(result, out_path)

        n_labeled = (y_tensor >= 0).sum().item()
        n_total = len(y_tensor)
        print(f"  {pt_path.name}: {n_total} genes, {n_labeled} with labels (y != -1) -> {out_path}")

    print(f"Done. Output written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
