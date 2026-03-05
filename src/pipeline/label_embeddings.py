"""Step 5: Attach y-label vectors to ProteomeLM embedding .pt files.

Reads .pt files from the embeddings output directory, joins each gene's
group_label (orgId:locusId) to genes.parquet to look up essentiality_class,
maps classes to integers, and saves new .pt files with keys: embeddings,
group_labels, y.

Label mapping (configurable via pipeline.yaml, defaults preserved):
    always_essential -> 0
    conditional      -> 1
    non_essential    -> 2
    no_data          -> -1
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

from src.config import PipelineConfig
from src.data_io import load_genes

logger = logging.getLogger(__name__)


def build_gene_lookup(
    class_to_int: dict[str, int],
    genes: pd.DataFrame | None = None,
    subset: str = "mvp",
) -> dict[str, int]:
    """Build a dict mapping 'orgId:locusId' -> integer label.

    Args:
        class_to_int: Mapping from essentiality_class string to integer.
        genes: Gene DataFrame with orgId, locusId, essentiality_class columns.
            If None, loads genes from disk for the given subset.
        subset: Data subset to load from if genes is None: "mvp" or "processed".

    Returns:
        Dict mapping gene_key -> y integer.
    """
    if genes is None:
        genes = load_genes(subset)
    genes = genes.copy()
    genes["gene_key"] = genes["orgId"].astype(str) + ":" + genes["locusId"].astype(str)
    genes["y"] = genes["essentiality_class"].map(class_to_int)
    return genes.set_index("gene_key")["y"].to_dict()


def label_single_file(
    pt_path: Path,
    output_dir: Path,
    gene_lookup: dict[str, int],
) -> tuple[int, int]:
    """Add y labels to a single .pt file and save to output_dir.

    Args:
        pt_path: Source .pt file (must contain embeddings and group_labels).
        output_dir: Destination directory.
        gene_lookup: Mapping from gene_key to integer label.

    Returns:
        Tuple of (total_genes, labeled_genes) where labeled means y != -1.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        logger.warning("Skip %s: unexpected format (not a dict)", pt_path.name)
        return 0, 0

    if "embeddings" not in data or "group_labels" not in data:
        logger.warning("Skip %s: missing embeddings or group_labels", pt_path.name)
        return 0, 0

    group_labels = data["group_labels"]
    y_list: list[int] = []
    for label in group_labels:
        y_val = gene_lookup.get(label, -1)
        if pd.isna(y_val):
            y_val = -1
        y_list.append(int(y_val))

    y_tensor = torch.tensor(y_list, dtype=torch.long)

    result = {
        "embeddings": data["embeddings"],
        "group_labels": group_labels,
        "y": y_tensor,
    }
    out_path = output_dir / pt_path.name
    torch.save(result, out_path)

    n_total = len(y_tensor)
    n_labeled = int((y_tensor >= 0).sum().item())
    logger.info(
        "  %s: %d genes, %d with labels (y != -1) -> %s",
        pt_path.name,
        n_total,
        n_labeled,
        out_path,
    )
    return n_total, n_labeled


def _labels_base_dir(config: PipelineConfig) -> Path:
    """Resolve data directory for labels from config.labels.subset."""
    subset = getattr(config.labels, "subset", "mvp")
    if subset not in ("processed", "mvp"):
        raise ValueError(f"labels.subset must be 'processed' or 'mvp', got {subset!r}")
    return config.data.processed_dir if subset == "processed" else config.data.mvp_dir


class LabelEmbeddingsStep:
    """Attach essentiality y-labels to ProteomeLM .pt files."""

    @property
    def name(self) -> str:
        return "label_embeddings"

    def check_inputs(self, config: PipelineConfig) -> bool:
        base_dir = _labels_base_dir(config)
        input_dir = base_dir / config.labels.input_subdir
        if not input_dir.is_dir():
            logger.warning("Embedding input directory not found: %s", input_dir)
            return False
        pt_files = sorted(input_dir.glob("*.pt"))
        if not pt_files:
            logger.warning("No .pt files in %s", input_dir)
            return False

        genes_path = base_dir / "genes.parquet"
        if not genes_path.is_file():
            logger.warning("genes.parquet not found: %s", genes_path)
            return False
        return True

    def run(self, config: PipelineConfig) -> None:
        base_dir = _labels_base_dir(config)
        input_dir = base_dir / config.labels.input_subdir
        output_dir = base_dir / config.labels.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        subset = getattr(config.labels, "subset", "mvp")
        class_to_int = config.labels.class_to_int
        logger.info("Subset:      %s", subset)
        logger.info("Label mapping: %s", class_to_int)
        logger.info("Input dir:  %s", input_dir)
        logger.info("Output dir: %s", output_dir)

        logger.info("Loading genes.parquet (%s) ...", subset)
        gene_lookup = build_gene_lookup(class_to_int, subset=subset)

        pt_files = sorted(input_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {input_dir}")

        logger.info("Found %d .pt files. Processing ...", len(pt_files))

        grand_total = 0
        grand_labeled = 0
        for pt_path in pt_files:
            n_total, n_labeled = label_single_file(pt_path, output_dir, gene_lookup)
            grand_total += n_total
            grand_labeled += n_labeled

        logger.info(
            "Labeling complete. %d total genes, %d with labels (y != -1).",
            grand_total,
            grand_labeled,
        )
