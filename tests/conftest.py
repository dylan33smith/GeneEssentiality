"""Shared fixtures and utilities for the GeneEssentiality test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.pipeline.create_fastas import parse_fasta_header


def parse_fasta_to_seqs(path: Path) -> dict[tuple[str, str], str]:
    """Parse a FASTA file into a dict mapping (orgId, locusId) -> sequence text."""
    seqs: dict[tuple[str, str], str] = {}
    current_key: tuple[str, str] | None = None
    current_seq: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if current_key is not None:
                    seqs[current_key] = "".join(current_seq)
                org_id, locus_id = parse_fasta_header(line.strip())
                if org_id and locus_id:
                    current_key = (str(org_id), str(locus_id))
                else:
                    current_key = None
                current_seq = []
            else:
                current_seq.append(line)
        if current_key is not None:
            seqs[current_key] = "".join(current_seq)
    return seqs


@pytest.fixture
def tmp_project_root(tmp_path: Path) -> Path:
    """Temp directory mimicking the project layout."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "mvp").mkdir(parents=True)
    (tmp_path / "config").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_fitness_df() -> pd.DataFrame:
    """Small fitness DataFrame spanning all 4 class outcomes.

    Gene layout (all in org "TestOrg"):
      geneA: 10 experiments, all confident (|t|>=2), all essential (fit<-1) -> always_essential
      geneB: 10 experiments, all confident, 5 essential -> conditional (50%)
      geneC: 10 experiments, all confident, 0 essential -> non_essential (0%)
      geneD: 10 experiments, none confident (|t|<2) -> no_data
    """
    rows = []
    for i in range(10):
        rows.append({"orgId": "TestOrg", "locusId": "geneA", "expName": f"expA{i}", "fit": -2.0, "t": 3.0})
    for i in range(5):
        rows.append({"orgId": "TestOrg", "locusId": "geneB", "expName": f"expB{i}", "fit": -2.0, "t": 3.0})
    for i in range(5, 10):
        rows.append({"orgId": "TestOrg", "locusId": "geneB", "expName": f"expB{i}", "fit": 0.0, "t": 3.0})
    for i in range(10):
        rows.append({"orgId": "TestOrg", "locusId": "geneC", "expName": f"expC{i}", "fit": 0.5, "t": 3.0})
    for i in range(10):
        rows.append({"orgId": "TestOrg", "locusId": "geneD", "expName": f"expD{i}", "fit": -2.0, "t": 0.5})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_genes_df() -> pd.DataFrame:
    """Small genes DataFrame with pre-computed essentiality columns."""
    return pd.DataFrame({
        "orgId": ["TestOrg"] * 4,
        "locusId": ["geneA", "geneB", "geneC", "geneD"],
        "sysName": ["sysA", "sysB", "sysC", "sysD"],
        "gene_length": [1000, 2000, 1500, 800],
        "n_total_experiments": [10, 10, 10, 10],
        "n_confident_experiments": [10, 10, 10, 0],
        "frac_essential_confident": [1.0, 0.5, 0.0, np.nan],
        "essentiality_class": ["always_essential", "conditional", "non_essential", "no_data"],
    })


@pytest.fixture
def sample_embeddings_dir(tmp_path: Path) -> Path:
    """Temp dir with 3 fake .pt files for organisms OrgA, OrgB, OrgC.

    OrgA: 10 genes, dim=8, y=[0,0, 1,1,1, 2,2,2,2, -1]
    OrgB: 8 genes, dim=8, y=[0, 1,1, 2,2,2,2, -1]
    OrgC: 5 genes, dim=8, y=[1, 2,2,2, -1]
    """
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    dim = 8

    for org_id, y_list in [
        ("OrgA", [0, 0, 1, 1, 1, 2, 2, 2, 2, -1]),
        ("OrgB", [0, 1, 1, 2, 2, 2, 2, -1]),
        ("OrgC", [1, 2, 2, 2, -1]),
    ]:
        n = len(y_list)
        data = {
            "embeddings": torch.randn(n, dim),
            "group_labels": [f"{org_id}:gene{i}" for i in range(n)],
            "y": torch.tensor(y_list, dtype=torch.long),
        }
        torch.save(data, emb_dir / f"{org_id}_proteomelm.pt")

    return emb_dir


@pytest.fixture
def sample_config(tmp_project_root: Path) -> "PipelineConfig":
    """A PipelineConfig pointing at the temp directories."""
    from src.config import (
        ClassifyGenesConfig,
        CreateFastasConfig,
        DataPaths,
        EmbeddingsConfig,
        ExtractDataConfig,
        LabelsConfig,
        MvpFilterConfig,
        PipelineConfig,
    )

    return PipelineConfig(
        project_root=tmp_project_root,
        data=DataPaths(
            raw_dir=tmp_project_root / "data" / "raw",
            processed_dir=tmp_project_root / "data" / "processed",
            mvp_dir=tmp_project_root / "data" / "mvp",
        ),
        extract_data=ExtractDataConfig(),
        classify_genes=ClassifyGenesConfig(),
        mvp_filter=MvpFilterConfig(),
        create_fastas=CreateFastasConfig(),
        embeddings=EmbeddingsConfig(),
        labels=LabelsConfig(),
    )


@pytest.fixture
def sample_model_config_yaml(sample_embeddings_dir: Path, tmp_path: Path) -> Path:
    """Write a minimal model.yaml to tmp dir and return its path."""
    config = {
        "data": {"input_dir": str(sample_embeddings_dir)},
        "split": {
            "train_organisms": ["OrgA"],
            "val_organisms": ["OrgB"],
            "test_organisms": ["OrgC"],
        },
    }
    config_path = tmp_path / "model.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path
