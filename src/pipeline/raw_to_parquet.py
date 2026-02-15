"""Step 1: Convert raw Fitness Browser exports to processed Parquet tables.

This step is a STUB. The raw-to-parquet transformation logic has not yet been
implemented in this repository (the processed Parquet files were built upstream).

When the expected output files already exist the step will skip silently. If
they are missing and the step is required, it raises NotImplementedError.
"""

from __future__ import annotations

import logging

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

EXPECTED_OUTPUTS = [
    "genes.parquet",
    "experiments.parquet",
    "fitness.parquet",
    "organisms.parquet",
    "proteins.fasta",
]


class RawToParquetStep:
    """Convert data/raw/ into data/processed/ Parquet tables + proteins.fasta."""

    @property
    def name(self) -> str:
        return "raw_to_parquet"

    def check_inputs(self, config: PipelineConfig) -> bool:
        raw_dir = config.data.raw_dir
        if not raw_dir.is_dir():
            logger.warning("Raw data directory does not exist: %s", raw_dir)
            return False
        return True

    def run(self, config: PipelineConfig) -> None:
        processed_dir = config.data.processed_dir
        missing = [
            f for f in EXPECTED_OUTPUTS if not (processed_dir / f).exists()
        ]

        if not missing:
            logger.info(
                "All processed outputs already exist in %s. Skipping.",
                processed_dir,
            )
            return

        raise NotImplementedError(
            f"raw_to_parquet step is not yet implemented. "
            f"Missing outputs in {processed_dir}: {missing}. "
            f"Please place pre-built Parquet files there or implement this step."
        )
