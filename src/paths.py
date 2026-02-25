"""Shared path resolution for the GeneEssentiality project."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Resolve project root: GENEESSENTIALITY_ROOT env var or parent of src/."""
    root = os.environ.get("GENEESSENTIALITY_ROOT")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[1]
