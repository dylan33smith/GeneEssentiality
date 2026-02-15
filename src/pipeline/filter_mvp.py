"""Step 2: Filter processed data into the MVP subset.

This step is a STUB. The MVP filtering logic (terrestrial organisms with
shared media, cor12 >= 0.2) has not yet been implemented in this repository.
The MVP Parquet files were built upstream.

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


class FilterMvpStep:
    """Filter data/processed/ into data/mvp/ (quality-filtered subset)."""

    @property
    def name(self) -> str:
        return "filter_mvp"

    def check_inputs(self, config: PipelineConfig) -> bool:
        processed_dir = config.data.processed_dir
        required = ["genes.parquet", "experiments.parquet", "fitness.parquet"]
        missing = [f for f in required if not (processed_dir / f).exists()]
        if missing:
            logger.warning(
                "Processed inputs missing for MVP filter: %s", missing
            )
            return False
        return True

    def run(self, config: PipelineConfig) -> None:
        mvp_dir = config.data.mvp_dir
        missing = [
            f for f in EXPECTED_OUTPUTS if not (mvp_dir / f).exists()
        ]

        if not missing:
            logger.info(
                "All MVP outputs already exist in %s. Skipping.", mvp_dir
            )
            return

        raise NotImplementedError(
            f"filter_mvp step is not yet implemented. "
            f"Missing outputs in {mvp_dir}: {missing}. "
            f"Please place pre-built MVP Parquet files there or implement this step. "
            f"Filter config: media={config.mvp_filter.media}, "
            f"min_cor12={config.mvp_filter.min_cor12}."
        )
