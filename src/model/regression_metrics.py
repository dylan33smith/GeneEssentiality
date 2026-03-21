"""Metrics for fitness regression (val RMSE, within-gene Spearman).

Usable from notebooks, scripts, and ``autoresearch_regression`` harness.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr


def val_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error on aligned 1D arrays."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true {len(y_true)} vs y_pred {len(y_pred)}")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mean_within_gene_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_keys: np.ndarray,
) -> dict[str, Any]:
    """Mean Spearman correlation within each gene (same gene, different conditions).

    For each ``gene_key``, computes Spearman between predictions and true ``fit``
    across all validation rows for that gene. Genes with fewer than two rows are
    skipped. Genes where Spearman is undefined (constant targets or preds) are
    skipped and counted in ``n_genes_nan_rho``.

    Args:
        y_true: True fitness values (same length as ``y_pred``).
        y_pred: Predicted fitness values.
        gene_keys: Same length as ``y_true`` / ``y_pred`` (object or string array).

    Returns:
        Dict with ``mean_rho``, ``n_genes_used``, ``n_genes_single_row``,
        ``n_genes_nan_rho``, and ``per_gene_rho`` (gene_key str -> float).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    gene_keys = np.asarray(gene_keys)
    if not (len(y_true) == len(y_pred) == len(gene_keys)):
        raise ValueError("y_true, y_pred, and gene_keys must have the same length")

    per_gene_rho: dict[str, float] = {}
    n_genes_single_row = 0
    n_genes_nan_rho = 0

    for g in np.unique(gene_keys):
        mask = gene_keys == g
        n = int(mask.sum())
        if n < 2:
            n_genes_single_row += 1
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        if np.std(yt) < 1e-15 or np.std(yp) < 1e-15:
            n_genes_nan_rho += 1
            continue
        rho, _ = spearmanr(yp, yt)
        if np.isnan(rho):
            n_genes_nan_rho += 1
            continue
        per_gene_rho[str(g)] = float(rho)

    if not per_gene_rho:
        return {
            "mean_rho": float("nan"),
            "n_genes_used": 0,
            "n_genes_single_row": n_genes_single_row,
            "n_genes_nan_rho": n_genes_nan_rho,
            "per_gene_rho": {},
        }

    mean_rho = float(np.mean(list(per_gene_rho.values())))
    return {
        "mean_rho": mean_rho,
        "n_genes_used": len(per_gene_rho),
        "n_genes_single_row": n_genes_single_row,
        "n_genes_nan_rho": n_genes_nan_rho,
        "per_gene_rho": per_gene_rho,
    }
