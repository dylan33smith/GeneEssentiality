"""Tests for src/pipeline/label_embeddings.py — labeling .pt files."""

from __future__ import annotations

from pathlib import Path

import torch

from src.pipeline.label_embeddings import label_single_file


def _make_pt_file(tmp_path: Path, org_id: str = "TestOrg") -> Path:
    """Create a minimal .pt file with embeddings and group_labels."""
    pt_path = tmp_path / f"{org_id}_proteomelm.pt"
    data = {
        "embeddings": torch.randn(3, 8),
        "group_labels": [f"{org_id}:g1", f"{org_id}:g2", f"{org_id}:g3"],
    }
    torch.save(data, pt_path)
    return pt_path


def test_label_single_file(tmp_path: Path):
    pt_path = _make_pt_file(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    lookup = {"TestOrg:g1": 0, "TestOrg:g2": 1, "TestOrg:g3": 2}

    n_total, n_labeled = label_single_file(pt_path, output_dir, lookup)
    assert n_total == 3
    assert n_labeled == 3

    result = torch.load(output_dir / pt_path.name, map_location="cpu", weights_only=False)
    assert "y" in result
    assert result["y"].tolist() == [0, 1, 2]


def test_label_single_file_unknown_gene(tmp_path: Path):
    pt_path = _make_pt_file(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    lookup = {"TestOrg:g1": 0}

    n_total, n_labeled = label_single_file(pt_path, output_dir, lookup)
    assert n_total == 3
    assert n_labeled == 1

    result = torch.load(output_dir / pt_path.name, map_location="cpu", weights_only=False)
    assert result["y"][0] == 0
    assert result["y"][1] == -1
    assert result["y"][2] == -1


def test_label_single_file_preserves_embeddings(tmp_path: Path):
    pt_path = _make_pt_file(tmp_path)
    original = torch.load(pt_path, map_location="cpu", weights_only=False)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    label_single_file(pt_path, output_dir, {})
    result = torch.load(output_dir / pt_path.name, map_location="cpu", weights_only=False)
    assert torch.equal(original["embeddings"], result["embeddings"])


def test_build_gene_lookup_with_dataframe():
    """build_gene_lookup accepts an explicit genes DataFrame."""
    import pandas as pd
    from src.pipeline.label_embeddings import build_gene_lookup

    genes = pd.DataFrame({
        "orgId": ["Org1", "Org1"],
        "locusId": ["g1", "g2"],
        "essentiality_class": ["always_essential", "non_essential"],
    })
    class_to_int = {"always_essential": 0, "conditional": 1, "non_essential": 2, "no_data": -1}
    lookup = build_gene_lookup(class_to_int, genes=genes)
    assert lookup["Org1:g1"] == 0
    assert lookup["Org1:g2"] == 2
