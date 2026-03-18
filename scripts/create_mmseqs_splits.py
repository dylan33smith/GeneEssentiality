#!/usr/bin/env python3
"""Create train/val/test splits based on MMseqs2 sequence-similarity clustering.

Reproduces the ProteomeLM-Ess split strategy (ProteomeLM paper: bioRxiv 2025.08.01.668221).
Methods, "Supervised gene essentiality prediction: ProteomeLM-Ess" -> "Data and training":

  "We split the proteins into training, validation and test sets by clustering proteins
   across all genomes according to sequence similarity. Specifically, we clustered
   sequences using MMSeqs2 [123] with a 40% similarity threshold. We designed our split
   so that if two labeled proteins belong to the same cluster, then they are either
   both in the training set, in the validation set, or in the test set. The data split
   is performed at the protein level and not at the genome level."

Implementation alignment:
  - 40% sequence identity: --min-seq-id 0.4 (default). Matches paper.
  - Cluster-level assignment: entire clusters assigned to one of train/val/test. Matches.
  - Protein-level split: we cluster all proteins from the combined FASTA (all genomes).
    Matches "across all genomes" and "at the protein level".
  - Val/test fractions and assignment order: not specified in the paper for essentiality.
    We use 10% val, 10% test and assign clusters by shuffled order + greedy protein
    count to approximate those fractions. (The paper uses 50/20/30 + min-cut only for a
    different task, PPI sequence-similarity-controlled splits in Supplementary 3.3.)
  - MMseqs2 coverage: paper does not specify; we use coverage=0.8, cov-mode=1.

Output: CSV mapping each protein (orgId:locusId) to its split for use with
ProtLM_embeddings_with_labels .pt files.

Requirements:
  - mmseqs2 installed and on PATH (conda install -c bioconda mmseqs2)

Usage:
  python scripts/create_mmseqs_splits.py                    # MVP (default)
  python scripts/create_mmseqs_splits.py --subset processed # Processed (48 organisms)
  python scripts/create_mmseqs_splits.py --subset mvp       # MVP explicitly
  python scripts/create_mmseqs_splits.py --min-seq-id 0.4 --val-frac 0.1 --test-frac 0.1
  python scripts/create_mmseqs_splits.py --fasta data/processed/proteins.fasta --output data/processed/mmseqs_splits.csv
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MMseqs2-based train/val/test splits"
    )
    parser.add_argument(
        "--subset", choices=["mvp", "processed"], default=None,
        help="Use preset paths for this data subset (mvp or processed). Overrides --fasta, --output, and default --embeddings-dir.",
    )
    parser.add_argument(
        "--fasta", type=Path,
        default=None,
        help="Input FASTA (headers: >orgId:locusId). Default: data/<subset>/proteins.fasta",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Output CSV. Default: data/<subset>/mmseqs_splits.csv",
    )
    parser.add_argument(
        "--min-seq-id", type=float, default=0.4,
        help="MMseqs2 minimum sequence identity for clustering (default: 0.4 = 40%%)",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.8,
        help="MMseqs2 coverage threshold (default: 0.8)",
    )
    parser.add_argument(
        "--cov-mode", type=int, default=1,
        help="MMseqs2 coverage mode (default: 1 = target coverage)",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of clusters for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.1,
        help="Fraction of clusters for test (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for cluster assignment (default: 42)",
    )
    parser.add_argument(
        "--exclude-no-data", action="store_true",
        help="If set, also load .pt labels and exclude genes with y == -1 from the output",
    )
    parser.add_argument(
        "--embeddings-dir", type=Path,
        default=None,
        help="Directory with labeled .pt files (used only with --exclude-no-data). Default: data/<subset>/ProtLM_embeddings_with_labels_layer8",
    )
    args = parser.parse_args()

    data_dir = project_root / "data"
    if args.subset == "processed":
        base = data_dir / "processed"
    elif args.subset == "mvp":
        base = data_dir / "mvp"
    else:
        base = data_dir / "mvp"

    if args.fasta is None:
        args.fasta = base / "proteins.fasta"
    if args.output is None:
        args.output = base / "mmseqs_splits.csv"
    if args.embeddings_dir is None:
        args.embeddings_dir = base / "ProtLM_embeddings_with_labels_layer8"

    return args


def check_mmseqs() -> str:
    """Check that mmseqs is installed and return its path."""
    path = shutil.which("mmseqs")
    if path is None:
        logger.error(
            "mmseqs not found on PATH. Install with: conda install -c bioconda mmseqs2"
        )
        sys.exit(1)
    version = subprocess.run(
        [path, "version"], capture_output=True, text=True
    ).stdout.strip()
    logger.info("Found mmseqs: %s (version %s)", path, version)
    return path


def run_mmseqs_cluster(
    mmseqs_bin: str,
    fasta_path: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
) -> Path:
    """Run mmseqs easy-cluster and return path to the cluster TSV."""
    result_prefix = tmp_dir / "clusterRes"
    mmseqs_tmp = tmp_dir / "mmseqs_tmp"
    mmseqs_tmp.mkdir(exist_ok=True)

    cmd = [
        mmseqs_bin, "easy-cluster",
        str(fasta_path),
        str(result_prefix),
        str(mmseqs_tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--threads", "4",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("mmseqs failed:\n%s\n%s", result.stdout, result.stderr)
        sys.exit(1)

    tsv_path = Path(f"{result_prefix}_cluster.tsv")
    if not tsv_path.exists():
        logger.error("Expected cluster TSV not found: %s", tsv_path)
        sys.exit(1)

    logger.info("Clustering complete: %s", tsv_path)
    return tsv_path


def parse_cluster_tsv(tsv_path: Path) -> dict[str, str]:
    """Parse MMseqs2 cluster TSV (representative\tmember) into {member: representative}."""
    member_to_rep: dict[str, str] = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                member_to_rep[member] = rep
    return member_to_rep


def assign_clusters_to_splits(
    member_to_rep: dict[str, str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Assign each cluster to train/val/test, then map to all members.

    Clusters are shuffled randomly and assigned greedily by protein count
    to approximate the target fractions while keeping clusters intact.
    """
    import numpy as np

    rng = np.random.RandomState(seed)

    rep_to_members: dict[str, list[str]] = defaultdict(list)
    for member, rep in member_to_rep.items():
        rep_to_members[rep].append(member)

    cluster_ids = list(rep_to_members.keys())
    cluster_sizes = [len(rep_to_members[c]) for c in cluster_ids]
    total_proteins = sum(cluster_sizes)

    target_val = int(total_proteins * val_frac)
    target_test = int(total_proteins * test_frac)

    order = rng.permutation(len(cluster_ids))

    val_count, test_count = 0, 0
    cluster_split: dict[str, str] = {}

    for idx in order:
        cid = cluster_ids[idx]
        size = cluster_sizes[idx]
        if val_count < target_val:
            cluster_split[cid] = "val"
            val_count += size
        elif test_count < target_test:
            cluster_split[cid] = "test"
            test_count += size
        else:
            cluster_split[cid] = "train"

    train_count = total_proteins - val_count - test_count
    logger.info(
        "Split assignment: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        train_count, 100 * train_count / total_proteins,
        val_count, 100 * val_count / total_proteins,
        test_count, 100 * test_count / total_proteins,
    )
    logger.info(
        "Clusters: %d total, train=%d, val=%d, test=%d",
        len(cluster_ids),
        sum(1 for s in cluster_split.values() if s == "train"),
        sum(1 for s in cluster_split.values() if s == "val"),
        sum(1 for s in cluster_split.values() if s == "test"),
    )

    rows = []
    for member, rep in member_to_rep.items():
        rows.append({
            "gene_key": member,
            "cluster_rep": rep,
            "split": cluster_split[rep],
        })

    return pd.DataFrame(rows)


def load_labeled_genes(embeddings_dir: Path) -> set[str]:
    """Load gene_keys that have y >= 0 from labeled .pt files."""
    import torch

    labeled = set()
    for pt_path in sorted(embeddings_dir.glob("*.pt")):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            continue
        labels = data.get("group_labels", [])
        y = data.get("y")
        if y is None:
            continue
        for gene_key, yi in zip(labels, y):
            if (isinstance(yi, int) and yi >= 0) or (hasattr(yi, "item") and yi.item() >= 0):
                labeled.add(gene_key)
    return labeled


def main() -> int:
    args = parse_args()

    if not args.fasta.is_file():
        logger.error("FASTA not found: %s", args.fasta)
        return 1

    mmseqs_bin = check_mmseqs()

    with tempfile.TemporaryDirectory(prefix="mmseqs_split_") as tmp_str:
        tmp_dir = Path(tmp_str)
        tsv_path = run_mmseqs_cluster(
            mmseqs_bin, args.fasta, tmp_dir,
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
        )
        member_to_rep = parse_cluster_tsv(tsv_path)

    logger.info("Parsed %d protein-to-cluster mappings", len(member_to_rep))

    df = assign_clusters_to_splits(
        member_to_rep,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    if args.exclude_no_data:
        if not args.embeddings_dir.is_dir():
            logger.warning("Embeddings dir not found: %s", args.embeddings_dir)
        else:
            labeled_genes = load_labeled_genes(args.embeddings_dir)
            before = len(df)
            df = df[df["gene_key"].isin(labeled_genes)]
            logger.info(
                "Filtered to labeled genes (y >= 0): %d -> %d", before, len(df)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved splits to %s (%d rows)", args.output, len(df))

    for split_name in ["train", "val", "test"]:
        count = (df["split"] == split_name).sum()
        logger.info("  %s: %d proteins", split_name, count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
