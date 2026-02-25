"""Tests for train/val/test split integrity from config/model.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = PROJECT_ROOT / "config" / "model.yaml"


def _load_splits() -> tuple[set[str], set[str], set[str]]:
    with open(MODEL_CONFIG) as f:
        cfg = yaml.safe_load(f)
    split = cfg.get("split", {})
    return (
        set(split.get("train_organisms", [])),
        set(split.get("val_organisms", [])),
        set(split.get("test_organisms", [])),
    )


def test_no_organism_overlap():
    train, val, test = _load_splits()
    assert not (train & val), f"Overlap train/val: {train & val}"
    assert not (train & test), f"Overlap train/test: {train & test}"
    assert not (val & test), f"Overlap val/test: {val & test}"


def test_all_mvp_organisms_assigned():
    train, val, test = _load_splits()
    assigned = train | val | test
    assert len(assigned) == 27, f"Expected 27 organisms, got {len(assigned)}"


def test_split_sizes_reasonable():
    train, val, test = _load_splits()
    assert len(train) >= 15, f"Train has only {len(train)} organisms"
    assert len(val) >= 3, f"Val has only {len(val)} organisms"
    assert len(test) >= 3, f"Test has only {len(test)} organisms"


def test_each_split_has_all_classes():
    """Each split should have >0 organisms (proxy for having all classes).
    Full class-level check requires data and is in test_data_validation.py.
    """
    train, val, test = _load_splits()
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_class_distribution_not_degenerate():
    """Verify no split is a single organism (which could have degenerate distribution)."""
    train, val, test = _load_splits()
    assert len(train) >= 3
    assert len(val) >= 2
    assert len(test) >= 2
