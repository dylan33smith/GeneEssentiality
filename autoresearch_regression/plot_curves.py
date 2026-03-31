#!/usr/bin/env python
"""Plot training curves from saved epoch logs.

Reads JSON files from ``curves/`` (written by each experiment's EpochLog)
and generates comparison figures. Processed runs use ``exp01_*.json``–``exp10_*.json``;
MVP runs use ``exp11_*.json``–``exp20_*.json`` so both can coexist.

Usage (from repo root):

    python -m autoresearch_regression.plot_curves
    python -m autoresearch_regression.plot_curves --experiments exp01_baseline,exp06_huber
    python -m autoresearch_regression.plot_curves --save-dir autoresearch_regression/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

CURVES_DIR = Path(__file__).resolve().parent / "curves"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def load_all_curves(selected: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Load epoch-log JSONs from curves/. Returns {experiment_name: data}."""
    if not CURVES_DIR.exists():
        print(f"No curves directory found at {CURVES_DIR}")
        return {}
    results = {}
    for p in sorted(CURVES_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        name = data["experiment"]
        if selected and name not in selected:
            continue
        results[name] = data
    return results


def _extract_series(data: dict[str, Any], key: str) -> tuple[list[int], list[float]]:
    """Extract (epochs, values) for a given metric key from epoch_metrics."""
    epochs = []
    values = []
    for row in data["epoch_metrics"]:
        if key in row:
            epochs.append(row["epoch"])
            values.append(row[key])
    return epochs, values


def _train_series_rmse_comparable(data: dict[str, Any]) -> tuple[list[int], list[float], bool]:
    """Training curve in the same units as validation RMSE when possible.

    ``train_mse`` is mean squared error on training batches; ``sqrt(mse)`` is
    training-set RMSE (same units as ``val_rmse``). If only ``train_loss`` is
    present (e.g. Huber), values are left as-is and ``rmselike`` is False.
    """
    epochs, vals = _extract_series(data, "train_mse")
    if vals:
        return epochs, [float(np.sqrt(v)) for v in vals], True
    epochs, vals = _extract_series(data, "train_loss")
    if vals:
        return epochs, vals, False
    return [], [], False


def plot_comparison(all_data: dict[str, dict[str, Any]], save_dir: Path | None = None) -> None:
    """Generate multi-experiment comparison plots."""
    if not all_data:
        print("No curve data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Experiment Training Curves", fontsize=16, fontweight="bold")

    n = len(all_data)
    if n <= 10:
        cmap = plt.cm.tab10
    elif n <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.hsv
    colors = cmap(np.linspace(0, 1, n))

    any_non_rmse_train = any(
        not _train_series_rmse_comparable(d)[2] for d in all_data.values()
    )

    # --- Panel 1: Train RMSE = sqrt(MSE), same units as val RMSE ---
    ax = axes[0, 0]
    for i, (name, data) in enumerate(all_data.items()):
        epochs, vals, rmselike = _train_series_rmse_comparable(data)
        if vals:
            ax.plot(epochs, vals, label=name, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Epoch")
    if any_non_rmse_train:
        ax.set_ylabel("Train error")
        ax.set_title("Training (√MSE → RMSE, else raw loss)")
    else:
        ax.set_ylabel("Train RMSE (√ of mean train MSE)")
        ax.set_title("Training RMSE")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Val RMSE (share y with train when both are RMSE-like) ---
    ax = axes[0, 1]
    for i, (name, data) in enumerate(all_data.items()):
        epochs, vals = _extract_series(data, "val_rmse")
        if vals:
            ax.plot(epochs, vals, label=name, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    if not any_non_rmse_train:
        ax.sharey(axes[0, 0])

    # --- Panel 3: Val Spearman ---
    ax = axes[1, 0]
    for i, (name, data) in enumerate(all_data.items()):
        epochs, vals = _extract_series(data, "val_spearman")
        if vals:
            ax.plot(epochs, vals, label=name, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Within-Gene Spearman")
    ax.set_title("Within-Gene Spearman")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Learning rate ---
    ax = axes[1, 1]
    for i, (name, data) in enumerate(all_data.items()):
        epochs, vals = _extract_series(data, "lr")
        if vals:
            ax.plot(epochs, vals, label=name, color=colors[i], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    # --- Individual experiment plots ---
    for name, data in all_data.items():
        fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
        fig2.suptitle(name, fontsize=14, fontweight="bold")

        tr_ep, tr_y, tr_rmse_like = _train_series_rmse_comparable(data)
        if tr_ep:
            lbl = "train RMSE (√MSE)" if tr_rmse_like else "train loss (not RMSE)"
            ax2[0].plot(tr_ep, tr_y, "b-", linewidth=1.5, label=lbl)
        ax2[0].set_xlabel("Epoch")
        ax2[0].set_ylabel("Train RMSE" if tr_rmse_like else "Train loss")
        ax2[0].set_title("Training (aligned with val when √MSE)")
        ax2[0].legend(fontsize=8)
        ax2[0].grid(True, alpha=0.3)

        epochs, rmse = _extract_series(data, "val_rmse")
        if rmse:
            ax2[1].plot(epochs, rmse, "r-", linewidth=1.5)
        ax2[1].set_xlabel("Epoch")
        ax2[1].set_ylabel("RMSE")
        ax2[1].set_title("Validation RMSE")
        ax2[1].grid(True, alpha=0.3)
        if tr_ep and tr_rmse_like and rmse:
            ax2[1].sharey(ax2[0])

        epochs, sp = _extract_series(data, "val_spearman")
        if sp:
            ax2[2].plot(epochs, sp, "g-", linewidth=1.5)
        ax2[2].set_xlabel("Epoch")
        ax2[2].set_ylabel("Spearman")
        ax2[2].set_title("Within-Gene Spearman")
        ax2[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            out = save_dir / f"{name}.png"
            fig2.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved: {out}")

    if not save_dir:
        plt.show()
    else:
        plt.close("all")


def print_summary_table(all_data: dict[str, dict[str, Any]]) -> None:
    """Print a text summary table of final metrics."""
    print("\n" + "=" * 80)
    print(f"{'Experiment':<25} {'Val RMSE':>10} {'Spearman':>10} {'Epochs':>8} {'Subset':>10}")
    print("-" * 80)
    for name, data in all_data.items():
        fm = data.get("final_metrics", {})
        cfg = data.get("config", {})
        rmse = fm.get("val_rmse", float("nan"))
        sp = fm.get("mean_within_gene_spearman", float("nan"))
        ep = cfg.get("epochs_max", cfg.get("epochs", "?"))
        subset = cfg.get("data_subset", "?")
        print(f"{name:<25} {rmse:>10.6f} {sp:>10.6f} {ep:>8} {subset:>10}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot autoresearch training curves")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment names (default: all in curves/)")
    parser.add_argument("--save-dir", type=str, default=str(FIGURES_DIR),
                        help="Directory to save figures (default: autoresearch_regression/figures/)")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively instead of saving")
    args = parser.parse_args()

    selected = [s.strip() for s in args.experiments.split(",")] if args.experiments else None
    all_data = load_all_curves(selected)

    if not all_data:
        print("No curve data found. Run experiments first.")
        return

    print(f"Loaded curves for {len(all_data)} experiment(s): {', '.join(all_data.keys())}")
    print_summary_table(all_data)

    save_dir = None if args.show else Path(args.save_dir)
    plot_comparison(all_data, save_dir=save_dir)


if __name__ == "__main__":
    main()
