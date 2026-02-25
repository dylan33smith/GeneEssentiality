"""Tests for src/pipeline/filter_mvp.py — MVP filtering functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.filter_mvp import (
    build_condition_vocab,
    filter_experiments,
    filter_fitness_vectorized,
    filter_mvp_sequences,
    get_mvp_organisms,
)


def _make_experiments() -> pd.DataFrame:
    return pd.DataFrame({
        "orgId": ["Org1", "Org1", "Org2", "Org3", "Org3"],
        "expName": ["e1", "e2", "e3", "e4", "e5"],
        "media": ["LB", "M9_glucose", "RCH2_defined", "Custom", "LB_salt"],
        "cor12": [0.3, 0.1, 0.25, 0.4, 0.35],
        "expGroup": ["group1", "group1", "group1", "plant", "group2"],
        "condition_1": ["c1", "c2", "c3", "c4", "c5"],
        "units_1": ["mg/L", "mg/L", "mM", "mg/L", "mM"],
        "aerobic": ["Yes", "Yes", "No", "Yes", "No"],
        "temperature": ["30", "30", "37", "30", "37"],
    })


def test_get_mvp_organisms():
    exps = _make_experiments()
    orgs = get_mvp_organisms(exps, media_prefixes=["LB", "RCH2", "M9"])
    assert "Org1" in orgs
    assert "Org2" in orgs
    assert "Org3" in orgs


def test_filter_experiments_cor12():
    exps = _make_experiments()
    mvp_orgs = {"Org1", "Org2", "Org3"}
    filtered = filter_experiments(exps, mvp_orgs, min_cor12=0.2, excluded_groups=set(), excluded_media=set())
    assert all(filtered["cor12"] >= 0.2)
    assert "e2" not in filtered["expName"].values


def test_filter_experiments_exclusions():
    exps = _make_experiments()
    mvp_orgs = {"Org1", "Org2", "Org3"}
    filtered = filter_experiments(
        exps, mvp_orgs, min_cor12=0.0, excluded_groups={"plant"}, excluded_media=set()
    )
    assert "e4" not in filtered["expName"].values


def test_filter_fitness_vectorized():
    fitness = pd.DataFrame({
        "orgId": ["Org1", "Org1", "Org2"],
        "locusId": ["g1", "g1", "g2"],
        "expName": ["e1", "e2", "e3"],
        "fit": [-1.0, 0.5, -2.0],
        "t": [3.0, 1.0, 4.0],
    })
    mvp_exps = pd.DataFrame({"orgId": ["Org1"], "expName": ["e1"]})
    result = filter_fitness_vectorized(fitness, mvp_exps)
    assert len(result) == 1
    assert result.iloc[0]["expName"] == "e1"


def test_build_condition_vocab():
    exps = _make_experiments()
    vocab = build_condition_vocab(exps)
    assert "media" in vocab
    assert "condition_1" in vocab
    assert "metadata" in vocab
    assert vocab["metadata"]["n_experiments"] == len(exps)


def test_filter_mvp_sequences(tmp_path: Path):
    fasta_content = ">Org1:g1\nACGT\n>Org1:g2\nTGCA\n>Org2:g3\nAAAA\n"
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(fasta_content)

    mvp_genes = pd.DataFrame({"orgId": ["Org1", "Org1"], "locusId": ["g1", "g2"]})
    output_path = tmp_path / "mvp_proteins.fasta"
    included, skipped = filter_mvp_sequences(fasta_path, mvp_genes, output_path)

    assert included == 2
    assert skipped == 1
    assert output_path.exists()
    text = output_path.read_text()
    assert ">Org1:g1" in text
    assert ">Org2:g3" not in text
