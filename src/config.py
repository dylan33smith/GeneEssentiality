"""Pipeline configuration: YAML loading and typed dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.paths import project_root

DEFAULT_CONFIG_PATH = project_root() / "config" / "pipeline.yaml"


@dataclass
class DataPaths:
    """Resolved absolute paths for the three data stages."""

    raw_dir: Path
    processed_dir: Path
    mvp_dir: Path


@dataclass
class ExtractDataConfig:
    """Parameters for the extract_data step."""

    db_filename: str = "feba.db"
    raw_sequences_filename: str = "aaseqs"


@dataclass
class ClassifyGenesConfig:
    """Parameters for the classify_genes step.

    Thresholds default to the values used by the legacy
    add_essentiality_classification.py script.
    """

    confident_t_threshold: float = 2.0
    essentiality_fit_threshold: float = -1.0
    always_essential_frac: float = 0.8
    conditional_min_frac: float = 0.1


@dataclass
class MvpFilterConfig:
    """Parameters for the filter_mvp step."""

    media_prefixes: list[str] = field(default_factory=lambda: ["LB", "RCH2", "M9"])
    min_cor12: float = 0.2
    excluded_exp_groups: list[str] = field(default_factory=lambda: ["plant"])
    excluded_media: list[str] = field(default_factory=lambda: ["Potato Dextrose Broth"])


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
    hidden_layer: int = 8
    # Data subset: "mvp" (default) or "processed". FASTAs from data/<subset>/organism_fastas.
    subset: str = "mvp"


@dataclass
class LabelsConfig:
    """Parameters for the label_embeddings step."""

    input_subdir: str = "ProtLM_embedddings"
    output_subdir: str = "ProtLM_embeddings_with_labels"
    # Data subset: "mvp" (default) or "processed". Must match embeddings.subset.
    subset: str = "mvp"
    class_to_int: dict[str, int] = field(
        default_factory=lambda: {
            "always_essential": 0,
            "conditional": 1,
            "non_essential": 2,
            "no_data": -1,
        }
    )


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire data pipeline."""

    project_root: Path
    data: DataPaths
    extract_data: ExtractDataConfig
    classify_genes: ClassifyGenesConfig
    mvp_filter: MvpFilterConfig
    create_fastas: CreateFastasConfig
    embeddings: EmbeddingsConfig
    labels: LabelsConfig


def _resolve_path(raw: str, root: Path) -> Path:
    """Resolve a path string relative to root if not absolute."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return root / p


def _build_from_yaml(cls: type, yaml_section: dict[str, Any] | None) -> Any:
    """Construct a dataclass from a YAML dict section.

    Reads only keys that correspond to dataclass fields. Coerces numeric
    types (int, float) from YAML values. Missing keys use field defaults.
    """
    import dataclasses

    section = yaml_section or {}
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in section:
            continue
        val = section[f.name]
        if val is not None and f.type == "float":
            val = float(val)
        elif val is not None and f.type == "int":
            val = int(val)
        kwargs[f.name] = val
    return cls(**kwargs)


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

    root = project_root()

    data_raw = raw.get("data", {})
    data_paths = DataPaths(
        raw_dir=_resolve_path(data_raw.get("raw_dir", "data/raw"), root),
        processed_dir=_resolve_path(data_raw.get("processed_dir", "data/processed"), root),
        mvp_dir=_resolve_path(data_raw.get("mvp_dir", "data/mvp"), root),
    )

    return PipelineConfig(
        project_root=root,
        data=data_paths,
        extract_data=_build_from_yaml(ExtractDataConfig, raw.get("extract_data")),
        classify_genes=_build_from_yaml(ClassifyGenesConfig, raw.get("classify_genes")),
        mvp_filter=_build_from_yaml(MvpFilterConfig, raw.get("mvp_filter")),
        create_fastas=_build_from_yaml(CreateFastasConfig, raw.get("create_fastas")),
        embeddings=_build_from_yaml(EmbeddingsConfig, raw.get("embeddings")),
        labels=_build_from_yaml(LabelsConfig, raw.get("labels")),
    )
