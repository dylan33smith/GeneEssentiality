"""PipelineStep protocol and step registry."""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineStep(Protocol):
    """Interface that every pipeline step must implement."""

    @property
    def name(self) -> str:
        """Short identifier used for CLI selection (e.g. 'create_fastas')."""
        ...

    def check_inputs(self, config: PipelineConfig) -> bool:
        """Return True if all required inputs for this step exist.

        Should log warnings for missing inputs but not raise.
        """
        ...

    def run(self, config: PipelineConfig) -> None:
        """Execute the step. Raise on fatal errors."""
        ...


def run_step_with_logging(step: PipelineStep, config: PipelineConfig) -> None:
    """Execute a pipeline step with timing and structured logging."""
    logger.info("=" * 60)
    logger.info("STEP: %s", step.name)
    logger.info("=" * 60)

    if not step.check_inputs(config):
        logger.warning("Input check failed for step '%s'. Attempting to run anyway.", step.name)

    t0 = time.perf_counter()
    step.run(config)
    elapsed = time.perf_counter() - t0

    logger.info("Step '%s' completed in %.1f s", step.name, elapsed)
    logger.info("")
