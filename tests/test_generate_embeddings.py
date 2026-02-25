"""Tests for src/pipeline/generate_embeddings.py helper functions.

Tests the parts that don't require GPU or ProteomeLM installation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from src.pipeline.generate_embeddings import _ensure_proteomelm_on_path, _resolve_device


class TestEnsureProteomlmOnPath:
    def test_adds_sibling_dir_to_path(self, tmp_path: Path):
        proteomelm_dir = tmp_path / "ProteomeLM"
        proteomelm_dir.mkdir()
        project_root = tmp_path / "GeneEssentiality"
        project_root.mkdir()

        original_path = sys.path.copy()
        try:
            _ensure_proteomelm_on_path(project_root)
            assert str(proteomelm_dir) in sys.path
        finally:
            sys.path[:] = original_path

    def test_noop_when_dir_missing(self, tmp_path: Path):
        project_root = tmp_path / "GeneEssentiality"
        project_root.mkdir()

        original_path = sys.path.copy()
        _ensure_proteomelm_on_path(project_root)
        assert sys.path == original_path

    def test_no_duplicate_entries(self, tmp_path: Path):
        proteomelm_dir = tmp_path / "ProteomeLM"
        proteomelm_dir.mkdir()
        project_root = tmp_path / "GeneEssentiality"
        project_root.mkdir()

        original_path = sys.path.copy()
        try:
            _ensure_proteomelm_on_path(project_root)
            _ensure_proteomelm_on_path(project_root)
            count = sys.path.count(str(proteomelm_dir))
            assert count == 1
        finally:
            sys.path[:] = original_path


class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_fallback_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert _resolve_device("cuda:0") == "cpu"

    def test_cuda_kept_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert _resolve_device("cuda:0") == "cuda:0"
