"""Data pipeline: orchestrates all steps from raw data to labeled embeddings."""

from __future__ import annotations

import logging
from typing import Sequence

from src.config import PipelineConfig
from src.pipeline.steps import PipelineStep, run_step_with_logging

logger = logging.getLogger(__name__)

STEP_ORDER: list[str] = [
    "raw_to_parquet",
    "filter_mvp",
    "create_fastas",
    "generate_embeddings",
    "label_embeddings",
]


def _build_default_steps() -> list[PipelineStep]:
    """Lazily import and instantiate all steps in canonical order."""
    from src.pipeline.create_fastas import CreateFastasStep
    from src.pipeline.filter_mvp import FilterMvpStep
    from src.pipeline.generate_embeddings import GenerateEmbeddingsStep
    from src.pipeline.label_embeddings import LabelEmbeddingsStep
    from src.pipeline.raw_to_parquet import RawToParquetStep

    return [
        RawToParquetStep(),
        FilterMvpStep(),
        CreateFastasStep(),
        GenerateEmbeddingsStep(),
        LabelEmbeddingsStep(),
    ]


class DataPipeline:
    """Orchestrates the full data processing pipeline.

    Supports running all steps, a single step, or a contiguous range.
    """

    def __init__(
        self,
        config: PipelineConfig,
        steps: Sequence[PipelineStep] | None = None,
    ) -> None:
        self.config = config
        self.steps = list(steps) if steps is not None else _build_default_steps()
        self._step_map = {s.name: s for s in self.steps}

    @property
    def step_names(self) -> list[str]:
        """Return ordered list of step names."""
        return [s.name for s in self.steps]

    def _resolve_range(
        self,
        start: str | None = None,
        end: str | None = None,
        single: str | None = None,
    ) -> list[PipelineStep]:
        """Select which steps to run based on CLI arguments.

        Args:
            start: Name of first step (inclusive). Defaults to first step.
            end: Name of last step (inclusive). Defaults to last step.
            single: If set, run only this step (overrides start/end).

        Returns:
            Ordered list of steps to execute.
        """
        names = self.step_names

        if single is not None:
            if single not in self._step_map:
                raise ValueError(
                    f"Unknown step '{single}'. Available: {names}"
                )
            return [self._step_map[single]]

        start_idx = 0
        end_idx = len(self.steps) - 1

        if start is not None:
            if start not in self._step_map:
                raise ValueError(
                    f"Unknown step '{start}'. Available: {names}"
                )
            start_idx = names.index(start)

        if end is not None:
            if end not in self._step_map:
                raise ValueError(
                    f"Unknown step '{end}'. Available: {names}"
                )
            end_idx = names.index(end)

        if start_idx > end_idx:
            raise ValueError(
                f"Start step '{start}' comes after end step '{end}'. "
                f"Order: {names}"
            )

        return self.steps[start_idx : end_idx + 1]

    def run(
        self,
        start: str | None = None,
        end: str | None = None,
        single: str | None = None,
    ) -> None:
        """Execute pipeline steps.

        Args:
            start: First step to run (inclusive). Defaults to first step.
            end: Last step to run (inclusive). Defaults to last step.
            single: If set, run only this one step (overrides start/end).
        """
        selected = self._resolve_range(start=start, end=end, single=single)
        step_names = [s.name for s in selected]
        logger.info("Pipeline will run %d step(s): %s", len(selected), step_names)

        for step in selected:
            run_step_with_logging(step, self.config)

        logger.info("Pipeline finished.")


__all__ = ["DataPipeline", "STEP_ORDER"]
