"""Fitness and essentiality classification (80%/10% rules preserved)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Classification thresholds (do not change) ---
# Confident measurement: |t| >= CONFIDENT_T_THRESHOLD
CONFIDENT_T_THRESHOLD = 2.0
# Essential in a single experiment: fit < ESSENTIALITY_FIT_THRESHOLD
ESSENTIALITY_FIT_THRESHOLD = -1.0
# Gene-level class (based on frac_essential_confident over confident experiments):
#   always_essential: frac > ALWAYS_ESSENTIAL_FRAC (>80%)
#   conditional:      CONDITIONAL_ESSENTIAL_FRAC_MIN <= frac <= CONDITIONAL_ESSENTIAL_FRAC_MAX (10-80%)
#   non_essential:    frac < CONDITIONAL_ESSENTIAL_FRAC_MIN (<10%)
#   no_data:           no confident experiments (n_confident == 0)
ALWAYS_ESSENTIAL_FRAC = 0.8
CONDITIONAL_ESSENTIAL_FRAC_MIN = 0.1
CONDITIONAL_ESSENTIAL_FRAC_MAX = 0.8


def get_essentiality_class(
    frac_essential_confident: float | np.ndarray,
    n_confident: int | np.ndarray,
) -> str | np.ndarray:
    """Assign essentiality class from fraction essential (confident exps only).

    Rules (preserved):
    - no_data: n_confident == 0
    - always_essential: frac_essential_confident > 0.8
    - conditional: 0.1 <= frac_essential_confident <= 0.8
    - non_essential: frac_essential_confident < 0.1 (and n_confident > 0)

    Confident = |t| >= 2; essential in an experiment = fit < -1.

    Args:
        frac_essential_confident: Fraction of confident experiments where fit < -1.
            Can be scalar or array; NaN for genes with no confident exps.
        n_confident: Number of confident experiments. Scalar or array.

    Returns:
        Class label(s): "always_essential", "conditional", "non_essential", "no_data".
        Same shape as inputs if array.
    """
    frac = np.asarray(frac_essential_confident)
    n = np.asarray(n_confident)
    scalar = frac.ndim == 0 and n.ndim == 0
    if scalar:
        frac = np.atleast_1d(frac)
        n = np.atleast_1d(n)

    out = np.select(
        [n == 0, frac > ALWAYS_ESSENTIAL_FRAC, frac >= CONDITIONAL_ESSENTIAL_FRAC_MIN],
        ["no_data", "always_essential", "conditional"],
        default="non_essential",
    )
    if scalar:
        return str(out[0])
    return out


def aggregate_fitness_to_genes(
    fitness: pd.DataFrame,
    confident_t_threshold: float = CONFIDENT_T_THRESHOLD,
    essentiality_fit_threshold: float = ESSENTIALITY_FIT_THRESHOLD,
) -> pd.DataFrame:
    """Compute gene-level stats and essentiality_class from raw fitness table.

    Uses vectorized groupby/agg (two passes). Thresholds are configurable
    and default to the module-level constants.

    Args:
        fitness: DataFrame with columns orgId, locusId, expName, fit, t.
        confident_t_threshold: |t| threshold for confident measurements.
        essentiality_fit_threshold: Fitness threshold for essential calls.

    Returns:
        DataFrame with one row per (orgId, locusId) and columns:
        n_total_experiments, n_confident_experiments, frac_not_confident,
        mean_fit_confident, frac_essential_all, frac_essential_confident,
        essentiality_class.
    """
    required = {"orgId", "locusId", "expName", "fit", "t"}
    if not required.issubset(fitness.columns):
        raise ValueError(f"fitness must have columns {required}")

    if len(fitness) == 0:
        return pd.DataFrame(
            columns=[
                "orgId", "locusId", "n_total_experiments",
                "n_confident_experiments", "frac_not_confident",
                "mean_fit_confident", "frac_essential_all",
                "frac_essential_confident", "essentiality_class",
            ]
        )

    all_stats = fitness.groupby(["orgId", "locusId"]).agg(
        n_total_experiments=("fit", "count"),
        n_essential_all=("fit", lambda x: (x < essentiality_fit_threshold).sum()),
    ).reset_index()

    confident = fitness[np.abs(fitness["t"]) >= confident_t_threshold]

    if len(confident) > 0:
        confident_stats = confident.groupby(["orgId", "locusId"]).agg(
            n_confident_experiments=("fit", "count"),
            mean_fit_confident=("fit", "mean"),
            n_essential_confident=(
                "fit", lambda x: (x < essentiality_fit_threshold).sum()
            ),
        ).reset_index()
    else:
        confident_stats = pd.DataFrame(
            columns=[
                "orgId", "locusId", "n_confident_experiments",
                "mean_fit_confident", "n_essential_confident",
            ]
        )

    gene_stats = all_stats.merge(
        confident_stats, on=["orgId", "locusId"], how="left"
    )
    gene_stats["n_confident_experiments"] = (
        gene_stats["n_confident_experiments"].fillna(0).infer_objects(copy=False).astype(int)
    )
    gene_stats["n_essential_confident"] = (
        gene_stats["n_essential_confident"].fillna(0).infer_objects(copy=False).astype(int)
    )
    gene_stats["frac_not_confident"] = 1 - (
        gene_stats["n_confident_experiments"] / gene_stats["n_total_experiments"]
    )
    gene_stats["frac_essential_all"] = (
        gene_stats["n_essential_all"] / gene_stats["n_total_experiments"]
    )
    gene_stats["frac_essential_confident"] = np.where(
        gene_stats["n_confident_experiments"] > 0,
        gene_stats["n_essential_confident"] / gene_stats["n_confident_experiments"],
        np.nan,
    )
    gene_stats["essentiality_class"] = get_essentiality_class(
        gene_stats["frac_essential_confident"].values,
        gene_stats["n_confident_experiments"].values,
    )
    gene_stats = gene_stats.drop(columns=["n_essential_all", "n_essential_confident"])
    return gene_stats
