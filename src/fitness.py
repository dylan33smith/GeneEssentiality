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

    out = np.where(n == 0, "no_data", np.where(
        frac > ALWAYS_ESSENTIAL_FRAC, "always_essential",
        np.where(
            frac >= CONDITIONAL_ESSENTIAL_FRAC_MIN, "conditional",
            "non_essential",
        ),
    ))
    if scalar:
        return str(out[0])
    return out


def aggregate_fitness_to_genes(fitness: pd.DataFrame) -> pd.DataFrame:
    """Compute gene-level stats and essentiality_class from raw fitness table.

    Uses vectorized groupby/agg. No row-wise loops.
    Thresholds: confident = |t| >= 2, essential = fit < -1;
    classes: >80% always_essential, 10-80% conditional, <10% non_essential,
    no confident exps = no_data.

    Args:
        fitness: DataFrame with columns orgId, locusId, expName, fit, t.

    Returns:
        DataFrame with one row per (orgId, locusId) and columns:
        n_total_experiments, n_confident_experiments, frac_not_confident,
        mean_fit_confident, frac_essential_all, frac_essential_confident,
        essentiality_class. Other gene metadata (scaffoldId, begin, end, etc.)
        must be merged from the genes table if needed.
    """
    required = {"orgId", "locusId", "expName", "fit", "t"}
    if not required.issubset(fitness.columns):
        raise ValueError(f"fitness must have columns {required}")

    t_abs = fitness["t"].abs()
    confident_mask = t_abs >= CONFIDENT_T_THRESHOLD
    essential_mask = fitness["fit"] < ESSENTIALITY_FIT_THRESHOLD

    idx = fitness.groupby(["orgId", "locusId"]).size().index
    n_total = fitness.groupby(["orgId", "locusId"]).size().reindex(idx).fillna(0)
    n_confident = (
        confident_mask.groupby([fitness["orgId"], fitness["locusId"]]).sum().reindex(idx)
    ).fillna(0).astype(int)
    n_total = n_total.astype(int)

    frac_not_confident = 1 - (n_confident / n_total.replace(0, np.nan))

    fit_conf = fitness.loc[confident_mask]
    mean_fit_confident = (
        fit_conf.groupby(["orgId", "locusId"])["fit"].mean().reindex(idx)
    )
    essential_in_confident = fit_conf["fit"] < ESSENTIALITY_FIT_THRESHOLD
    n_essential_confident = (
        essential_in_confident.groupby([fit_conf["orgId"], fit_conf["locusId"]])
        .sum()
        .reindex(idx)
        .fillna(0)
        .astype(int)
    )
    frac_essential_all = (
        essential_mask.groupby([fitness["orgId"], fitness["locusId"]]).mean().reindex(idx)
    )
    frac_essential_confident = n_essential_confident / n_confident.replace(0, np.nan)

    essentiality_class = get_essentiality_class(
        frac_essential_confident.values, n_confident.values
    )

    result = pd.DataFrame(
        {
            "n_total_experiments": n_total.values,
            "n_confident_experiments": n_confident.values,
            "frac_not_confident": frac_not_confident.values,
            "mean_fit_confident": mean_fit_confident.values,
            "frac_essential_all": frac_essential_all.values,
            "frac_essential_confident": frac_essential_confident.values,
            "essentiality_class": essentiality_class,
        },
        index=idx,
    ).reset_index()
    return result
