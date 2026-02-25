"""Tests for src/data_io.py — data loading functions.

All tests require actual Parquet data on disk and are marked with @pytest.mark.data.
"""

from __future__ import annotations

import pytest

from src.data_io import (
    get_data_dir,
    load_experiments,
    load_fitness,
    load_genes,
    load_organisms,
)

pytestmark = pytest.mark.data


def test_get_data_dir_exists():
    d = get_data_dir()
    assert d.name == "data"
    assert d.is_dir()


def test_load_genes_mvp():
    genes = load_genes("mvp")
    assert len(genes) > 0
    for col in ("orgId", "locusId", "essentiality_class", "n_confident_experiments"):
        assert col in genes.columns


def test_load_genes_processed():
    genes = load_genes("processed")
    assert len(genes) > 0
    assert "orgId" in genes.columns


def test_load_genes_invalid_subset():
    """Invalid subset raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="subset must be"):
        load_genes("invalid")  # type: ignore[arg-type]


def test_load_experiments_mvp():
    exps = load_experiments("mvp")
    assert len(exps) > 0
    for col in ("orgId", "expName", "media", "cor12"):
        assert col in exps.columns


def test_load_fitness_mvp():
    fitness = load_fitness("mvp")
    assert len(fitness) > 0
    for col in ("orgId", "locusId", "fit", "t"):
        assert col in fitness.columns


def test_load_organisms_mvp():
    orgs = load_organisms("mvp")
    assert len(orgs) > 0
    for col in ("orgId", "genus", "species"):
        assert col in orgs.columns
