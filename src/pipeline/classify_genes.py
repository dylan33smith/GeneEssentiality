"""Steps 2 & 4: Compute essentiality classification on genes.parquet.

Adds per-gene essentiality statistics and classification to the genes table
using fitness data. The t-score threshold for "confident" measurements is
configurable (default 1.0, matching the legacy pipeline that built the
current Parquet files).

Uses src.fitness.get_essentiality_class() for the final 4-class assignment
(always_essential / conditional / non_essential / no_data) -- that function
depends only on frac and n_confident, not on the t-threshold.

This step class is instantiated twice in the pipeline:
    ClassifyGenesStep("processed")  -> classify_genes_processed
    ClassifyGenesStep("mvp")        -> classify_genes_mvp

Refactored from: src/data/add_essentiality_classification.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ClassifyGenesConfig, PipelineConfig
from src.fitness import aggregate_fitness_to_genes

logger = logging.getLogger(__name__)


def compute_gene_essentiality(
    fitness_df: pd.DataFrame,
    cfg: ClassifyGenesConfig,
) -> pd.DataFrame:
    """Compute per-gene essentiality statistics from a fitness table.

    Delegates to src.fitness.aggregate_fitness_to_genes with config thresholds.

    Args:
        fitness_df: DataFrame with columns orgId, locusId, fit, t.
        cfg: Classification config with threshold parameters.

    Returns:
        DataFrame with one row per (orgId, locusId) and columns:
        n_total_experiments, n_confident_experiments, frac_not_confident,
        mean_fit_confident, frac_essential_all, frac_essential_confident,
        essentiality_class.
    """
    return aggregate_fitness_to_genes(
        fitness_df,
        confident_t_threshold=cfg.confident_t_threshold,
        essentiality_fit_threshold=cfg.essentiality_fit_threshold,
    )


def merge_essentiality_into_genes(
    genes_df: pd.DataFrame,
    gene_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Merge essentiality statistics into genes DataFrame.

    Drops any existing essentiality columns first, then left-joins the
    new stats. Genes with no fitness data get essentiality_class='no_data'.

    Args:
        genes_df: Original genes DataFrame.
        gene_stats: Output of compute_gene_essentiality().

    Returns:
        Updated genes DataFrame with essentiality columns.
    """
    essentiality_cols = [
        "n_total_experiments",
        "n_confident_experiments",
        "frac_not_confident",
        "mean_fit_confident",
        "frac_essential_all",
        "frac_essential_confident",
        "frac_essential",
        "essentiality_class",
    ]
    existing = [c for c in essentiality_cols if c in genes_df.columns]
    if existing:
        genes_df = genes_df.drop(columns=existing)

    merge_cols = [
        "orgId",
        "locusId",
        "n_total_experiments",
        "n_confident_experiments",
        "frac_not_confident",
        "mean_fit_confident",
        "frac_essential_all",
        "frac_essential_confident",
        "essentiality_class",
    ]

    genes_updated = genes_df.merge(gene_stats[merge_cols], on=["orgId", "locusId"], how="left")

    genes_updated["essentiality_class"] = genes_updated["essentiality_class"].fillna("no_data")
    genes_updated["n_total_experiments"] = (
        genes_updated["n_total_experiments"].fillna(0).astype(int)
    )
    genes_updated["n_confident_experiments"] = (
        genes_updated["n_confident_experiments"].fillna(0).astype(int)
    )

    return genes_updated


class ClassifyGenesStep:
    """Compute essentiality classification on a subset's genes.parquet.

    Instantiated with subset="processed" or "mvp" to produce two separate
    pipeline steps sharing the same logic.
    """

    def __init__(self, subset: str) -> None:
        if subset not in ("processed", "mvp"):
            raise ValueError(f"subset must be 'processed' or 'mvp', got '{subset}'")
        self._subset = subset

    @property
    def name(self) -> str:
        return f"classify_genes_{self._subset}"

    def _data_dir(self, config: PipelineConfig) -> Path:
        if self._subset == "mvp":
            return config.data.mvp_dir
        return config.data.processed_dir

    def check_inputs(self, config: PipelineConfig) -> bool:
        data_dir = self._data_dir(config)
        ok = True
        for f in ("genes.parquet", "fitness.parquet"):
            if not (data_dir / f).is_file():
                logger.warning("%s not found: %s", f, data_dir / f)
                ok = False
        return ok

    def run(self, config: PipelineConfig) -> None:
        data_dir = self._data_dir(config)
        genes_path = data_dir / "genes.parquet"
        fitness_path = data_dir / "fitness.parquet"

        if not genes_path.is_file() or not fitness_path.is_file():
            logger.warning(
                "Skipping %s: genes.parquet or fitness.parquet not found in %s",
                self.name,
                data_dir,
            )
            return

        cfg = config.classify_genes
        logger.info("Parameters: |t| >= %.1f, fit < %.1f", cfg.confident_t_threshold, cfg.essentiality_fit_threshold)
        logger.info("  always_essential: > %.0f%%", cfg.always_essential_frac * 100)
        logger.info("  conditional: %.0f%%-%.0f%%", cfg.conditional_min_frac * 100, cfg.always_essential_frac * 100)

        logger.info("Loading %s genes.parquet ...", self._subset)
        genes_df = pd.read_parquet(genes_path)
        logger.info("  %s genes", f"{len(genes_df):,}")

        logger.info("Loading %s fitness.parquet ...", self._subset)
        fitness_df = pd.read_parquet(fitness_path)
        logger.info("  %s fitness records", f"{len(fitness_df):,}")

        total = len(fitness_df)
        n_confident = int((np.abs(fitness_df["t"]) >= cfg.confident_t_threshold).sum())
        logger.info(
            "  Confident (|t| >= %.1f): %s (%.1f%%)",
            cfg.confident_t_threshold,
            f"{n_confident:,}",
            n_confident / total * 100 if total > 0 else 0,
        )

        gene_stats = compute_gene_essentiality(fitness_df, cfg)
        genes_with_confident = int((gene_stats["n_confident_experiments"] > 0).sum())
        logger.info("  Genes with fitness data: %s", f"{len(gene_stats):,}")
        logger.info("  Genes with confident data: %s", f"{genes_with_confident:,}")

        if len(gene_stats) > 0:
            for cls, count in gene_stats["essentiality_class"].value_counts().items():
                logger.info("    %s: %s", cls, f"{count:,}")

        genes_updated = merge_essentiality_into_genes(genes_df, gene_stats)

        logger.info("Final classification (all %s genes):", self._subset)
        for cls, count in genes_updated["essentiality_class"].value_counts().items():
            logger.info("    %s: %s", cls, f"{count:,}")

        genes_updated.to_parquet(genes_path, index=False)
        logger.info("Saved %s", genes_path)
