"""Tests for src.model.regression_metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.model.regression_metrics import mean_within_gene_spearman, val_rmse


def test_val_rmse_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert val_rmse(y, y) == pytest.approx(0.0)


def test_val_rmse_simple() -> None:
    # errors [0, 2] -> MSE 2 -> RMSE sqrt(2)
    assert val_rmse(np.array([0.0, 2.0]), np.array([0.0, 0.0])) == pytest.approx(np.sqrt(2.0))


def test_within_gene_spearman_two_genes() -> None:
    # gene a: (1,3) vs (1,2) — same rank order -> rho 1
    # gene b: (1,2) vs (1,3) — same rank order -> rho 1
    y_true = np.array([1.0, 3.0, 1.0, 2.0])
    y_pred = np.array([1.0, 2.0, 1.0, 3.0])
    gk = np.array(["a", "a", "b", "b"])
    out = mean_within_gene_spearman(y_true, y_pred, gk)
    assert out["mean_rho"] == pytest.approx(1.0)
    assert out["n_genes_used"] == 2
    assert out["n_genes_single_row"] == 0


def test_within_gene_skips_single_row_genes() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    gk = np.array(["x", "y", "y"])
    out = mean_within_gene_spearman(y_true, y_pred, gk)
    assert out["n_genes_single_row"] == 1
    assert out["n_genes_used"] == 1
    assert out["mean_rho"] == pytest.approx(1.0)
