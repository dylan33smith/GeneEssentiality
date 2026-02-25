"""Tests for src/config.py — YAML loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import PipelineConfig, _resolve_path, load_config


def test_load_config_default_path():
    cfg = load_config()
    assert isinstance(cfg, PipelineConfig)
    assert cfg.data.mvp_dir.is_absolute()


def test_load_config_custom_yaml(tmp_path: Path):
    config_dict = {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "mvp_dir": "data/mvp",
        },
        "embeddings": {
            "proteomelm_model": "Bitbol-Lab/ProteomeLM-S",
            "device": "cpu",
            "hidden_layer": 12,
        },
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    cfg = load_config(config_path)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.embeddings.device == "cpu"
    assert cfg.embeddings.hidden_layer == 12


def test_load_config_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_hidden_layer_default(tmp_path: Path):
    config_dict = {"data": {}, "embeddings": {"device": "cpu"}}
    config_path = tmp_path / "no_layer.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    cfg = load_config(config_path)
    assert cfg.embeddings.hidden_layer == 8


def test_load_config_hidden_layer_override(tmp_path: Path):
    config_dict = {"data": {}, "embeddings": {"hidden_layer": -1}}
    config_path = tmp_path / "layer_override.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    cfg = load_config(config_path)
    assert cfg.embeddings.hidden_layer == -1


def test_resolve_path_relative(tmp_path: Path):
    result = _resolve_path("data/raw", tmp_path)
    assert result == tmp_path / "data" / "raw"
    assert result.is_absolute()


def test_resolve_path_absolute(tmp_path: Path):
    abs_path = "/tmp/absolute/path"
    result = _resolve_path(abs_path, tmp_path)
    assert result == Path(abs_path)
