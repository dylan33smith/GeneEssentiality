"""Tests for src/pipeline/classify_genes.py — gene essentiality classification step."""

from __future__ import annotations

import pandas as pd

from src.config import ClassifyGenesConfig
from src.pipeline.classify_genes import (
    ClassifyGenesStep,
    compute_gene_essentiality,
    merge_essentiality_into_genes,
)


def test_compute_gene_essentiality(sample_fitness_df: pd.DataFrame):
    cfg = ClassifyGenesConfig()
    result = compute_gene_essentiality(sample_fitness_df, cfg)
    assert len(result) == 4
    classes = dict(zip(result["locusId"], result["essentiality_class"]))
    assert classes["geneA"] == "always_essential"
    assert classes["geneB"] == "conditional"
    assert classes["geneC"] == "non_essential"
    assert classes["geneD"] == "no_data"


def test_compute_gene_essentiality_custom_thresholds(sample_fitness_df: pd.DataFrame):
    cfg = ClassifyGenesConfig(confident_t_threshold=5.0)
    result = compute_gene_essentiality(sample_fitness_df, cfg)
    for _, row in result.iterrows():
        assert row["essentiality_class"] == "no_data"


def test_merge_essentiality_into_genes(sample_fitness_df: pd.DataFrame):
    genes_df = pd.DataFrame({
        "orgId": ["TestOrg"] * 5,
        "locusId": ["geneA", "geneB", "geneC", "geneD", "geneX"],
        "gene": ["a", "b", "c", "d", "x"],
    })
    cfg = ClassifyGenesConfig()
    gene_stats = compute_gene_essentiality(sample_fitness_df, cfg)
    merged = merge_essentiality_into_genes(genes_df, gene_stats)
    assert "essentiality_class" in merged.columns
    gene_x = merged[merged["locusId"] == "geneX"].iloc[0]
    assert gene_x["essentiality_class"] == "no_data"


def test_classify_step_name():
    step = ClassifyGenesStep("processed")
    assert step.name == "classify_genes_processed"
    step_mvp = ClassifyGenesStep("mvp")
    assert step_mvp.name == "classify_genes_mvp"


def test_classify_step_check_inputs_missing(sample_config):
    step = ClassifyGenesStep("processed")
    assert step.check_inputs(sample_config) is False
