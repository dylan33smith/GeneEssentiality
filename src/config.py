"""Pipeline configuration: YAML loading and typed dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Resolve project root (parent of src/)."""
    import os

    root = os.environ.get("GENEESSENTIALITY_ROOT")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[1]


DEFAULT_CONFIG_PATH = _project_root() / "config" / "pipeline.yaml"


@dataclass
class DataPaths:
    """Resolved absolute paths for the three data stages."""

    raw_dir: Path
    processed_dir: Path
    mvp_dir: Path


@dataclass
class CreateFastasConfig:
    """Parameters for the create_fastas step."""

    subsets: list[str] = field(default_factory=lambda: ["processed", "mvp"])


@dataclass
class EmbeddingsConfig:
    """Parameters for the generate_embeddings step."""

    proteomelm_model: str = "Bitbol-Lab/ProteomeLM-S"
    device: str = "cuda:0"
    input_subdir: str = "organism_fastas"
    output_subdir: str = "ProtLM_embedddings"


@dataclass
class LabelsConfig:
    """Parameters for the label_embeddings step."""

    input_subdir: str = "ProtLM_embedddings"
    output_subdir: str = "ProtLM_embeddings_with_labels"
    class_to_int: dict[str, int] = field(
        default_factory=lambda: {
            "always_essential": 0,
            "conditional": 1,
            "non_essential": 2,
            "no_data": -1,
        }
    )


@dataclass
class MvpFilterConfig:
    """Parameters for the filter_mvp step."""

    media: list[str] = field(default_factory=lambda: ["LB", "RCH2_defined", "M9"])
    min_cor12: float = 0.2


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire data pipeline."""

    project_root: Path
    data: DataPaths
    create_fastas: CreateFastasConfig
    embeddings: EmbeddingsConfig
    labels: LabelsConfig
    mvp_filter: MvpFilterConfig


def _resolve_path(raw: str, project_root: Path) -> Path:
    """Resolve a path string relative to project_root if not absolute."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return project_root / p


def load_config(path: Path | str | None = None) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        path: Path to YAML config. Defaults to config/pipeline.yaml.

    Returns:
        Fully resolved PipelineConfig dataclass.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    root = _project_root()

    data_raw = raw.get("data", {})
    data_paths = DataPaths(
        raw_dir=_resolve_path(data_raw.get("raw_dir", "data/raw"), root),
        processed_dir=_resolve_path(data_raw.get("processed_dir", "data/processed"), root),
        mvp_dir=_resolve_path(data_raw.get("mvp_dir", "data/mvp"), root),
    )

    cf_raw = raw.get("create_fastas", {})
    create_fastas = CreateFastasConfig(
        subsets=cf_raw.get("subsets", ["processed", "mvp"]),
    )

    emb_raw = raw.get("embeddings", {})
    embeddings = EmbeddingsConfig(
        proteomelm_model=emb_raw.get("proteomelm_model", "Bitbol-Lab/ProteomeLM-S"),
        device=emb_raw.get("device", "cuda:0"),
        input_subdir=emb_raw.get("input_subdir", "organism_fastas"),
        output_subdir=emb_raw.get("output_subdir", "ProtLM_embedddings"),
    )

    default_class_to_int = {
        "always_essential": 0,
        "conditional": 1,
        "non_essential": 2,
        "no_data": -1,
    }
    lab_raw = raw.get("labels", {})
    labels = LabelsConfig(
        input_subdir=lab_raw.get("input_subdir", "ProtLM_embedddings"),
        output_subdir=lab_raw.get("output_subdir", "ProtLM_embeddings_with_labels"),
        class_to_int=lab_raw.get("class_to_int", default_class_to_int),
    )

    mvp_raw = raw.get("mvp_filter", {})
    mvp_filter = MvpFilterConfig(
        media=mvp_raw.get("media", ["LB", "RCH2_defined", "M9"]),
        min_cor12=float(mvp_raw.get("min_cor12", 0.2)),
    )

    return PipelineConfig(
        project_root=root,
        data=data_paths,
        create_fastas=create_fastas,
        embeddings=embeddings,
        labels=labels,
        mvp_filter=mvp_filter,
    )
