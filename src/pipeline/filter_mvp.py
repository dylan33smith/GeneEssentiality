"""Step 3: Create MVP subset from processed data.

Filters the processed Parquet tables to the "Terrestrial" MVP subset
(organisms using LB/RCH2/M9 media, quality-filtered experiments). Also
creates the MVP proteins.fasta and condition_vocab.json.

Refactored from:
    src/data/create_mvp_subset.py
    src/data/process_sequences.py   (filter_mvp_sequences part)
    src/data/build_condition_vocab.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

EXPECTED_PARQUETS = [
    "genes.parquet",
    "experiments.parquet",
    "fitness.parquet",
    "organisms.parquet",
]
EXPECTED_EXTRAS = [
    "proteins.fasta",
    "condition_vocab.json",
]


def get_mvp_organisms(
    experiments_df: pd.DataFrame,
    media_prefixes: list[str],
) -> set[str]:
    """Identify organisms that use any of the target media prefixes.

    Args:
        experiments_df: Full experiments table.
        media_prefixes: List of media prefix strings (e.g. ["LB", "RCH2", "M9"]).

    Returns:
        Set of orgId strings for MVP organisms.
    """
    mask = pd.Series(False, index=experiments_df.index)
    for prefix in media_prefixes:
        mask = mask | experiments_df["media"].str.startswith(prefix, na=False)
    return set(experiments_df.loc[mask, "orgId"].unique())


def filter_experiments(
    experiments_df: pd.DataFrame,
    mvp_orgs: set[str],
    min_cor12: float,
    excluded_groups: set[str],
    excluded_media: set[str],
) -> pd.DataFrame:
    """Filter experiments for MVP subset."""
    mask = (
        experiments_df["orgId"].isin(mvp_orgs)
        & (experiments_df["cor12"] >= min_cor12)
        & (~experiments_df["expGroup"].isin(excluded_groups))
        & (~experiments_df["media"].isin(excluded_media))
    )
    return experiments_df[mask].copy()


def filter_fitness_vectorized(
    fitness_df: pd.DataFrame,
    mvp_experiments: pd.DataFrame,
) -> pd.DataFrame:
    """Filter fitness records to MVP experiments using a vectorized merge.

    This replaces the legacy list-comprehension approach that iterated
    27M rows in Python, providing a major speedup.
    """
    valid_keys = mvp_experiments[["orgId", "expName"]].drop_duplicates()
    return fitness_df.merge(valid_keys, on=["orgId", "expName"], how="inner")


def filter_mvp_sequences(
    processed_fasta: Path,
    mvp_genes_df: pd.DataFrame,
    output_path: Path,
) -> tuple[int, int]:
    """Filter processed proteins.fasta to only MVP genes.

    Args:
        processed_fasta: Path to data/processed/proteins.fasta.
        mvp_genes_df: MVP genes DataFrame (must have orgId, locusId).
        output_path: Path to write data/mvp/proteins.fasta.

    Returns:
        Tuple of (included_count, skipped_count).
    """
    mvp_gene_ids = set(zip(mvp_genes_df["orgId"], mvp_genes_df["locusId"]))

    included = 0
    skipped = 0

    with open(processed_fasta) as fin, open(output_path, "w") as fout:
        include_seq = False
        for line in fin:
            if line.startswith(">"):
                header = line.strip().lstrip(">")
                parts = header.split(":")
                if len(parts) >= 2:
                    org_id, locus_id = parts[0], parts[1]
                else:
                    org_id, locus_id = None, None
                include_seq = (org_id, locus_id) in mvp_gene_ids
                if include_seq:
                    included += 1
                    fout.write(line)
                else:
                    skipped += 1
            elif include_seq:
                fout.write(line)

    return included, skipped


def build_condition_vocab(experiments_df: pd.DataFrame) -> dict:
    """Build label-encoding vocabularies from MVP experiments.

    Returns a dict suitable for JSON serialization with keys:
    media, condition_1, expGroup, units_1, composite, aerobic, temperature, metadata.
    """

    def _vocab(values: pd.Series, name: str) -> dict:
        unique_vals = sorted(values.dropna().unique())
        vocab = {val: idx for idx, val in enumerate(unique_vals)}
        inverse = {idx: val for idx, val in enumerate(unique_vals)}
        logger.info("  %s: %d unique values", name, len(vocab))
        return {"vocab": vocab, "inverse": inverse, "size": len(vocab)}

    media_vocab = _vocab(experiments_df["media"], "media")
    condition_vocab = _vocab(experiments_df["condition_1"], "condition_1")
    expgroup_vocab = _vocab(experiments_df["expGroup"], "expGroup")
    units_vocab = _vocab(experiments_df["units_1"], "units_1")

    composite = (
        experiments_df["media"].fillna("") + "_" + experiments_df["condition_1"].fillna("")
    )
    composite_vocab = _vocab(composite, "composite")

    aerobic_vocab = _vocab(experiments_df["aerobic"], "aerobic")
    temperature_vocab = _vocab(experiments_df["temperature"], "temperature")

    return {
        "media": media_vocab,
        "condition_1": condition_vocab,
        "expGroup": expgroup_vocab,
        "units_1": units_vocab,
        "composite": composite_vocab,
        "aerobic": aerobic_vocab,
        "temperature": temperature_vocab,
        "metadata": {
            "n_experiments": len(experiments_df),
            "n_organisms": int(experiments_df["orgId"].nunique()),
            "description": "Label encodings for MVP experiment conditions",
        },
    }


class FilterMvpStep:
    """Create MVP subset from processed data."""

    @property
    def name(self) -> str:
        return "filter_mvp"

    def check_inputs(self, config: PipelineConfig) -> bool:
        processed_dir = config.data.processed_dir
        required = ["genes.parquet", "experiments.parquet", "fitness.parquet", "organisms.parquet"]
        missing = [f for f in required if not (processed_dir / f).is_file()]
        if missing:
            logger.warning("Processed inputs missing for MVP filter: %s", missing)
            return False
        return True

    def run(self, config: PipelineConfig) -> None:
        processed_dir = config.data.processed_dir
        mvp_dir = config.data.mvp_dir

        all_expected = EXPECTED_PARQUETS + EXPECTED_EXTRAS
        missing = [f for f in all_expected if not (mvp_dir / f).exists()]
        if not missing:
            logger.info("All MVP outputs already exist in %s. Skipping.", mvp_dir)
            return

        mvp_dir.mkdir(parents=True, exist_ok=True)
        mvp_cfg = config.mvp_filter

        logger.info("Loading processed data ...")
        experiments_df = pd.read_parquet(processed_dir / "experiments.parquet")
        genes_df = pd.read_parquet(processed_dir / "genes.parquet")
        organisms_df = pd.read_parquet(processed_dir / "organisms.parquet")
        fitness_df = pd.read_parquet(processed_dir / "fitness.parquet")
        logger.info(
            "  %s experiments, %s genes, %s fitness records",
            f"{len(experiments_df):,}",
            f"{len(genes_df):,}",
            f"{len(fitness_df):,}",
        )

        logger.info("Identifying MVP organisms (media prefixes: %s) ...", mvp_cfg.media_prefixes)
        mvp_orgs = get_mvp_organisms(experiments_df, mvp_cfg.media_prefixes)
        logger.info("  Found %d MVP organisms", len(mvp_orgs))

        excluded_groups = set(mvp_cfg.excluded_exp_groups)
        excluded_media = set(mvp_cfg.excluded_media)
        logger.info(
            "Filtering experiments (cor12 >= %.2f, exclude groups=%s, media=%s) ...",
            mvp_cfg.min_cor12,
            excluded_groups,
            excluded_media,
        )
        mvp_experiments = filter_experiments(
            experiments_df, mvp_orgs, mvp_cfg.min_cor12, excluded_groups, excluded_media
        )
        logger.info(
            "  Retained %s experiments (%.1f%%)",
            f"{len(mvp_experiments):,}",
            len(mvp_experiments) / len(experiments_df) * 100,
        )

        mvp_genes = genes_df[genes_df["orgId"].isin(mvp_orgs)].copy()
        logger.info("  Retained %s genes", f"{len(mvp_genes):,}")

        mvp_organisms = organisms_df[organisms_df["orgId"].isin(mvp_orgs)].copy()

        logger.info("Filtering fitness records (vectorized merge) ...")
        mvp_fitness = filter_fitness_vectorized(fitness_df, mvp_experiments)
        logger.info(
            "  Retained %s fitness records (%.1f%%)",
            f"{len(mvp_fitness):,}",
            len(mvp_fitness) / len(fitness_df) * 100 if len(fitness_df) > 0 else 0,
        )

        logger.info("Saving MVP Parquet files ...")
        mvp_genes.to_parquet(mvp_dir / "genes.parquet", index=False)
        mvp_experiments.to_parquet(mvp_dir / "experiments.parquet", index=False)
        mvp_organisms.to_parquet(mvp_dir / "organisms.parquet", index=False)
        mvp_fitness.to_parquet(mvp_dir / "fitness.parquet", index=False)

        processed_fasta = processed_dir / "proteins.fasta"
        mvp_fasta = mvp_dir / "proteins.fasta"
        if processed_fasta.is_file():
            logger.info("Filtering sequences for MVP subset ...")
            included, skipped = filter_mvp_sequences(processed_fasta, mvp_genes, mvp_fasta)
            logger.info("  Included: %s, Skipped: %s", f"{included:,}", f"{skipped:,}")
        else:
            logger.warning("processed/proteins.fasta not found; skipping sequence filter")

        logger.info("Building condition vocabulary ...")
        vocab = build_condition_vocab(mvp_experiments)
        vocab_path = mvp_dir / "condition_vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab, f, indent=2)
        logger.info("  Saved %s", vocab_path)

        logger.info("MVP subset complete: %d organisms, %s genes, %s experiments, %s fitness",
                     len(mvp_organisms), f"{len(mvp_genes):,}",
                     f"{len(mvp_experiments):,}", f"{len(mvp_fitness):,}")
