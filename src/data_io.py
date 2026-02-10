"""Data loading and path resolution for the GeneEssentiality pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd

DataSubset = Literal["processed", "mvp"]


def _project_root() -> Path:
    """Resolve project root: parent of src/ or GENEESSENTIALITY_ROOT env."""
    root = os.environ.get("GENEESSENTIALITY_ROOT")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Return the data directory (project_root/data)."""
    return _project_root() / "data"


def get_processed_dir() -> Path:
    """Return the processed data directory (full database, all organisms)."""
    return get_data_dir() / "processed"


def get_mvp_dir() -> Path:
    """Return the MVP data directory (quality-filtered subset)."""
    return get_data_dir() / "mvp"


def load_genes(subset: DataSubset = "mvp") -> pd.DataFrame:
    """Load the gene table (metadata and essentiality class).

    Expected columns include: orgId, locusId, sysName, scaffoldId, begin, end,
    gene_length, type, strand, gene, desc, GC, n_total_experiments,
    n_confident_experiments, frac_not_confident, mean_fit_confident,
    frac_essential_all, frac_essential_confident, essentiality_class.

    Args:
        subset: "processed" (all 48 organisms) or "mvp" (27 organisms).

    Returns:
        Gene DataFrame with one row per gene.
    """
    base = get_mvp_dir() if subset == "mvp" else get_processed_dir()
    return pd.read_parquet(base / "genes.parquet")


def load_experiments(subset: DataSubset = "mvp") -> pd.DataFrame:
    """Load the experiments table (conditions and metadata).

    Expected columns include: orgId, expName, expDesc, expGroup, mutantLibrary,
    media, mediaStrength, condition_1, concentration_1, units_1, temperature,
    aerobic, cor12, maxFit, nGenerations, etc.

    Args:
        subset: "processed" or "mvp".

    Returns:
        Experiments DataFrame with one row per experiment.
    """
    base = get_mvp_dir() if subset == "mvp" else get_processed_dir()
    return pd.read_parquet(base / "experiments.parquet")


def load_fitness(subset: DataSubset = "mvp") -> pd.DataFrame:
    """Load the fitness table (gene-experiment fitness scores).

    Expected columns: orgId, locusId, expName, fit (log2 fitness), t (t-stat).

    Args:
        subset: "processed" or "mvp".

    Returns:
        Fitness DataFrame with one row per (gene, experiment).
    """
    base = get_mvp_dir() if subset == "mvp" else get_processed_dir()
    return pd.read_parquet(base / "fitness.parquet")


def load_organisms(subset: DataSubset = "mvp") -> pd.DataFrame:
    """Load the organisms table (taxonomy and metadata).

    Expected columns: orgId, division, genus, species, strain, taxonomyId.

    Args:
        subset: "processed" or "mvp".

    Returns:
        Organisms DataFrame with one row per organism.
    """
    base = get_mvp_dir() if subset == "mvp" else get_processed_dir()
    return pd.read_parquet(base / "organisms.parquet")


def load_media_experiments() -> pd.DataFrame:
    """Load media/experiment metadata from Excel (sheet index 2).

    Returns:
        DataFrame from data/media_composition.xlsx, sheet_name=2.
    """
    path = get_data_dir() / "media_composition.xlsx"
    return pd.read_excel(path, sheet_name=2)


def inspect_table(df: pd.DataFrame, name: str = "") -> None:
    """Print column names and row count for a table.

    Args:
        df: DataFrame to inspect.
        name: Optional label (e.g. "genes", "fitness"); if set, prints row count.
    """
    for col in df.columns:
        print(col)
    if name:
        print(f"Number of {name}: {len(df)}")
