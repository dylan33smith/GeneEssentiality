"""Tests for src/pipeline/__init__.py — DataPipeline orchestration."""

from __future__ import annotations

import pytest

from src.pipeline import STEP_ORDER, DataPipeline


def test_step_order_length():
    assert len(STEP_ORDER) == 7


def test_pipeline_step_names(sample_config):
    pipeline = DataPipeline(sample_config)
    assert pipeline.step_names == STEP_ORDER


def test_pipeline_resolve_single(sample_config):
    pipeline = DataPipeline(sample_config)
    steps = pipeline._resolve_range(single="filter_mvp")
    assert len(steps) == 1
    assert steps[0].name == "filter_mvp"


def test_pipeline_resolve_range(sample_config):
    pipeline = DataPipeline(sample_config)
    steps = pipeline._resolve_range(start="filter_mvp", end="create_fastas")
    names = [s.name for s in steps]
    assert names[0] == "filter_mvp"
    assert names[-1] == "create_fastas"
    assert len(names) >= 2


def test_pipeline_unknown_step_raises(sample_config):
    pipeline = DataPipeline(sample_config)
    with pytest.raises(ValueError, match="Unknown step"):
        pipeline._resolve_range(single="nonexistent")


def test_pipeline_reversed_range_raises(sample_config):
    pipeline = DataPipeline(sample_config)
    with pytest.raises(ValueError, match="comes after"):
        pipeline._resolve_range(start="create_fastas", end="filter_mvp")
