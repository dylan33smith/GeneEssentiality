"""Data contract tests — validate actual data on disk.

All tests require real Parquet/pt data and are marked with @pytest.mark.data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data_io import get_mvp_dir, load_experiments, load_fitness, load_genes, load_organisms
from src.model.dataset import EssentialityDataset

pytestmark = pytest.mark.data

VALID_CLASSES = {"always_essential", "conditional", "non_essential", "no_data"}
EXPECTED_GENE_COLUMNS = {
    "orgId", "locusId", "sysName", "scaffoldId", "begin", "end",
    "gene_length", "type", "strand", "gene", "desc", "GC",
    "n_total_experiments", "n_confident_experiments", "frac_not_confident",
    "mean_fit_confident", "frac_essential_all", "frac_essential_confident",
    "essentiality_class",
}


def test_mvp_genes_has_required_columns():
    genes = load_genes("mvp")
    missing = EXPECTED_GENE_COLUMNS - set(genes.columns)
    assert not missing, f"Missing columns: {missing}"


def test_mvp_genes_essentiality_classes_valid():
    genes = load_genes("mvp")
    classes = set(genes["essentiality_class"].unique())
    assert classes.issubset(VALID_CLASSES), f"Unexpected classes: {classes - VALID_CLASSES}"


def test_mvp_genes_no_duplicate_locus_per_org():
    genes = load_genes("mvp")
    dupes = genes.duplicated(subset=["orgId", "locusId"], keep=False)
    assert dupes.sum() == 0, f"{dupes.sum()} duplicate (orgId, locusId) pairs"


def test_mvp_experiments_cor12_above_threshold():
    exps = load_experiments("mvp")
    below = exps[exps["cor12"] < 0.2]
    assert len(below) == 0, f"{len(below)} experiments have cor12 < 0.2"


def test_mvp_fitness_no_null_fit_or_t():
    fitness = load_fitness("mvp")
    assert fitness["fit"].isna().sum() == 0
    assert fitness["t"].isna().sum() == 0


def test_mvp_organisms_count():
    orgs = load_organisms("mvp")
    assert len(orgs) == 27


def test_embeddings_pt_files_match_organisms():
    orgs = load_organisms("mvp")
    org_ids = set(orgs["orgId"])
    emb_dir = get_mvp_dir() / "ProtLM_embeddings_with_labels"
    if not emb_dir.is_dir():
        pytest.skip(f"Embeddings dir not found: {emb_dir}")
    pt_orgs = set()
    for f in emb_dir.glob("*.pt"):
        stem = f.stem
        if stem.endswith("_proteomelm"):
            stem = stem[: -len("_proteomelm")]
        pt_orgs.add(stem)
    missing = org_ids - pt_orgs
    assert not missing, f"Missing .pt files for organisms: {missing}"


def test_embeddings_shape_consistent():
    emb_dir = get_mvp_dir() / "ProtLM_embeddings_with_labels"
    if not emb_dir.is_dir():
        pytest.skip(f"Embeddings dir not found: {emb_dir}")
    dims = set()
    for pt_path in emb_dir.glob("*.pt"):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        dims.add(data["embeddings"].shape[1])
    assert len(dims) == 1, f"Inconsistent embedding dims: {dims}"


def test_embeddings_labels_match_genes():
    genes = load_genes("mvp")
    emb_dir = get_mvp_dir() / "ProtLM_embeddings_with_labels"
    if not emb_dir.is_dir():
        pytest.skip(f"Embeddings dir not found: {emb_dir}")

    class_to_int = {"always_essential": 0, "conditional": 1, "non_essential": 2, "no_data": -1}
    genes["expected_y"] = genes["essentiality_class"].map(class_to_int)
    gene_lookup = dict(zip(
        genes["orgId"].astype(str) + ":" + genes["locusId"].astype(str),
        genes["expected_y"],
    ))

    mismatches = 0
    for pt_path in sorted(emb_dir.glob("*.pt"))[:3]:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        for label, y_val in zip(data["group_labels"], data["y"].tolist()):
            expected = gene_lookup.get(label, -1)
            if y_val != expected:
                mismatches += 1
    assert mismatches == 0, f"{mismatches} label mismatches in first 3 organisms"


def test_no_data_filtered_in_labeled_embeddings():
    emb_dir = get_mvp_dir() / "ProtLM_embeddings_with_labels"
    if not emb_dir.is_dir():
        pytest.skip(f"Embeddings dir not found: {emb_dir}")
    orgs = load_organisms("mvp")
    org_id = orgs["orgId"].iloc[0]
    ds = EssentialityDataset(emb_dir, organism_ids=[org_id], exclude_no_data=True)
    labels = ds.get_labels()
    assert (labels == -1).sum() == 0
