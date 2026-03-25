#!/usr/bin/env python
"""Run all (or selected) experiments and append results to results.tsv.

Usage (from repository root):

    python -m autoresearch_regression.run_all
    python -m autoresearch_regression.run_all --experiments exp01_baseline,exp06_huber
    python -m autoresearch_regression.run_all --start-from exp05_deep   # resume queue from exp05 onward
    DATA_SUBSET=mvp AUTORESEARCH_EPOCHS=8 python -m autoresearch_regression.run_all

The ``epochs`` column in ``results.tsv`` is the configured **max** (``AUTORESEARCH_EPOCHS``);
runs may end earlier if early stopping triggers.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
RESULTS_TSV = Path(__file__).resolve().parent / "results.tsv"

ALL_EXPERIMENTS = [
    "exp01_baseline",
    "exp02_wider",
    "exp03_layernorm_gelu",
    "exp04_batchnorm",
    "exp05_deep",
    "exp06_huber",
    "exp07_residual",
    "exp08_ranking_mse",
    "exp09_pairwise",
    "exp10_lr_sweep",
]

TSV_HEADER = "timestamp\texperiment\tval_rmse\tmean_within_gene_spearman\tn_genes_used\tstatus\tdata_subset\tepochs\tdescription\n"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ensure_results_file() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)
    elif RESULTS_TSV.stat().st_size == 0:
        RESULTS_TSV.write_text(TSV_HEADER)


def _append_result(experiment: str, metrics: dict[str, Any] | None,
                   status: str, description: str = "") -> None:
    _ensure_results_file()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subset = os.environ.get("DATA_SUBSET", "processed")
    epochs = os.environ.get("AUTORESEARCH_EPOCHS", "8")
    if metrics:
        rmse = f"{metrics['val_rmse']:.6f}"
        spearman = f"{metrics['mean_within_gene_spearman']:.6f}"
        n_genes = str(metrics.get("n_genes_used_for_spearman", ""))
    else:
        rmse = spearman = n_genes = ""
    row = f"{ts}\t{experiment}\t{rmse}\t{spearman}\t{n_genes}\t{status}\t{subset}\t{epochs}\t{description}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)


def run_experiment(name: str) -> dict[str, Any] | None:
    module_name = f"autoresearch_regression.experiments.{name}"
    logger.info("=" * 60)
    logger.info("STARTING: %s", name)
    logger.info("=" * 60)
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        metrics = mod.train()
        elapsed = time.time() - t0
        logger.info("FINISHED: %s in %.1fs", name, elapsed)
        _append_result(name, metrics, "ok", f"elapsed={elapsed:.0f}s")
        return metrics
    except Exception:
        elapsed = time.time() - t0
        logger.error("CRASHED: %s after %.1fs", name, elapsed)
        traceback.print_exc()
        _append_result(name, None, "crash", traceback.format_exc().splitlines()[-1][:120])
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoresearch experiments")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment names (mutually exclusive with --start-from)")
    parser.add_argument("--start-from", type=str, default=None, metavar="EXP_NAME",
                        help="Run ALL_EXPERIMENTS from this name onward (inclusive); "
                             "mutually exclusive with --experiments")
    args = parser.parse_args()

    if args.experiments and args.start_from:
        parser.error("Use either --experiments or --start-from, not both.")

    if args.start_from:
        if args.start_from not in ALL_EXPERIMENTS:
            parser.error(f"Unknown experiment: {args.start_from}. Choose from {ALL_EXPERIMENTS}")
        idx = ALL_EXPERIMENTS.index(args.start_from)
        selected = ALL_EXPERIMENTS[idx:]
        logger.info("--start-from %s → running %d experiment(s) from index %d",
                    args.start_from, len(selected), idx)
    elif args.experiments:
        selected = [e.strip() for e in args.experiments.split(",")]
        for s in selected:
            if s not in ALL_EXPERIMENTS:
                parser.error(f"Unknown experiment: {s}. Choose from {ALL_EXPERIMENTS}")
    else:
        selected = ALL_EXPERIMENTS

    _ensure_results_file()
    logger.info("Running %d experiment(s): %s", len(selected), ", ".join(selected))
    logger.info(
        "DATA_SUBSET=%s  AUTORESEARCH_EPOCHS=%s (max)  EARLY_STOP_PATIENCE=%s  EARLY_STOP_MIN_DELTA=%s",
        os.environ.get("DATA_SUBSET", "processed"),
        os.environ.get("AUTORESEARCH_EPOCHS", "8"),
        os.environ.get("EARLY_STOP_PATIENCE", "3"),
        os.environ.get("EARLY_STOP_MIN_DELTA", "1e-4"),
    )

    for name in selected:
        run_experiment(name)

    logger.info("=" * 60)
    logger.info("ALL DONE. Results written to %s", RESULTS_TSV)


if __name__ == "__main__":
    main()
