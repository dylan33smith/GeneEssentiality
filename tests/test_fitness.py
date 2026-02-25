"""Tests for src/fitness.py — essentiality classification logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fitness import aggregate_fitness_to_genes, get_essentiality_class


class TestGetEssentialityClass:
    def test_always_essential(self):
        assert get_essentiality_class(0.9, 10) == "always_essential"

    def test_conditional(self):
        assert get_essentiality_class(0.5, 10) == "conditional"

    def test_non_essential(self):
        assert get_essentiality_class(0.05, 10) == "non_essential"

    def test_no_data(self):
        assert get_essentiality_class(0.9, 0) == "no_data"
        assert get_essentiality_class(0.0, 0) == "no_data"

    def test_boundary_80_is_conditional(self):
        assert get_essentiality_class(0.8, 10) == "conditional"

    def test_boundary_10_is_conditional(self):
        assert get_essentiality_class(0.1, 10) == "conditional"

    def test_boundary_above_80(self):
        assert get_essentiality_class(0.81, 10) == "always_essential"

    def test_boundary_below_10(self):
        assert get_essentiality_class(0.09, 10) == "non_essential"

    def test_vectorized(self):
        fracs = np.array([0.9, 0.5, 0.05, 0.5])
        ns = np.array([10, 10, 10, 0])
        result = get_essentiality_class(fracs, ns)
        expected = np.array(["always_essential", "conditional", "non_essential", "no_data"])
        np.testing.assert_array_equal(result, expected)


class TestAggregateFitnessToGenes:
    def test_basic(self, sample_fitness_df: pd.DataFrame):
        result = aggregate_fitness_to_genes(sample_fitness_df)
        assert len(result) == 4

        gene_a = result[result["locusId"] == "geneA"].iloc[0]
        assert gene_a["essentiality_class"] == "always_essential"
        assert gene_a["n_confident_experiments"] == 10

        gene_b = result[result["locusId"] == "geneB"].iloc[0]
        assert gene_b["essentiality_class"] == "conditional"
        assert abs(gene_b["frac_essential_confident"] - 0.5) < 1e-9

        gene_c = result[result["locusId"] == "geneC"].iloc[0]
        assert gene_c["essentiality_class"] == "non_essential"

        gene_d = result[result["locusId"] == "geneD"].iloc[0]
        assert gene_d["essentiality_class"] == "no_data"
        assert gene_d["n_confident_experiments"] == 0

    def test_all_below_threshold(self):
        """All |t| < 2 -> n_confident=0 -> no_data."""
        df = pd.DataFrame({
            "orgId": ["Org"] * 5,
            "locusId": ["g1"] * 5,
            "expName": [f"e{i}" for i in range(5)],
            "fit": [-2.0] * 5,
            "t": [1.0] * 5,
        })
        result = aggregate_fitness_to_genes(df)
        assert result.iloc[0]["essentiality_class"] == "no_data"

    def test_missing_columns(self):
        df = pd.DataFrame({"orgId": ["A"], "locusId": ["g1"]})
        with pytest.raises(ValueError, match="fitness must have columns"):
            aggregate_fitness_to_genes(df)

    def test_custom_t_threshold(self, sample_fitness_df: pd.DataFrame):
        """With t_threshold=5.0, no experiments are confident -> all no_data."""
        result = aggregate_fitness_to_genes(
            sample_fitness_df, confident_t_threshold=5.0
        )
        for _, row in result.iterrows():
            assert row["essentiality_class"] == "no_data"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["orgId", "locusId", "expName", "fit", "t"])
        result = aggregate_fitness_to_genes(df)
        assert len(result) == 0
        assert "essentiality_class" in result.columns
