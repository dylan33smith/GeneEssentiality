#!/usr/bin/env python3
"""
Create data visualizations for GeneEssentiality processed and MVP datasets.

Generates 16 visualization types for both processed and mvp subsets.
Output: figures/{processed,mvp}/*.png and figures/comparison/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_io import load_experiments, load_fitness, load_genes
from src.fitness import (
    ALWAYS_ESSENTIAL_FRAC,
    CONDITIONAL_ESSENTIAL_FRAC_MIN,
    CONFIDENT_T_THRESHOLD,
    ESSENTIALITY_FIT_THRESHOLD,
)

ESSENTIALITY_CLASSES = ["always_essential", "conditional", "non_essential", "no_data"]
HIST_BINS_EXPERIMENTS = [0, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf]


def _threshold_text() -> str:
    return (
        f"Confident: |t| >= {CONFIDENT_T_THRESHOLD}; essential per experiment: fit < {ESSENTIALITY_FIT_THRESHOLD}. "
        f"Classes: >{int(ALWAYS_ESSENTIAL_FRAC*100)}% always_essential; "
        f"{int(CONDITIONAL_ESSENTIAL_FRAC_MIN*100)}-{int(ALWAYS_ESSENTIAL_FRAC*100)}% conditional; "
        f"<{int(CONDITIONAL_ESSENTIAL_FRAC_MIN*100)}% non_essential; no confident experiments = no_data"
    )


def plot_essentiality_distribution(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    counts = genes["essentiality_class"].value_counts()
    ordered = [c for c in ESSENTIALITY_CLASSES if c in counts.index]
    vals = [counts.get(c, 0) for c in ordered]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ordered, vals, color=sns.color_palette("husl", 4))
    ax.set_xlabel("Essentiality class")
    ax.set_ylabel("Gene count")
    ax.set_title(f"Essentiality class distribution ({subset})")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=20, ha="right")
    fig.suptitle(_threshold_text(), fontsize=8, y=-0.02, wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap_condition_organism(
    experiments: pd.DataFrame, subset: str, output_path: Path, top_n: int = 30
) -> None:
    exp = experiments.copy()
    exp["condition_1"] = exp["condition_1"].fillna("unknown")
    top_conds = exp["condition_1"].value_counts().head(top_n).index
    exp = exp[exp["condition_1"].isin(top_conds)]
    pivot = exp.groupby(["condition_1", "orgId"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, annot=False, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Experiments: condition vs organism ({subset})")
    ax.set_xlabel("Organism")
    ax.set_ylabel("Condition")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap_media_organism(
    experiments: pd.DataFrame, subset: str, output_path: Path, top_n: int = 40
) -> None:
    exp = experiments.copy()
    exp["media"] = exp["media"].fillna("unknown")
    top_media = exp["media"].value_counts().head(top_n).index
    exp = exp[exp["media"].isin(top_media)]
    pivot = exp.groupby(["media", "orgId"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, annot=False, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Experiments: media vs organism ({subset})")
    ax.set_xlabel("Organism")
    ax.set_ylabel("Media")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histogram_genes_vs_experiments(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    x = genes["n_total_experiments"].values
    bins = HIST_BINS_EXPERIMENTS
    hist, _ = np.histogram(x, bins=bins)
    labels = [f"{int(b)}–{int(bins[i+1])}" if bins[i+1] != np.inf else f"{int(b)}+" for i, b in enumerate(bins[:-1])]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, hist, color="steelblue", edgecolor="black")
    ax.set_xlabel("Number of experiments per gene")
    ax.set_ylabel("Gene count")
    ax.set_title(f"Genes vs number of experiments ({subset})")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histogram_genes_vs_fitness_all(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    x = genes["n_total_experiments"].values
    bins = HIST_BINS_EXPERIMENTS
    hist, _ = np.histogram(x, bins=bins)
    labels = [f"{int(b)}–{int(bins[i+1])}" if bins[i+1] != np.inf else f"{int(b)}+" for i, b in enumerate(bins[:-1])]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, hist, color="steelblue", edgecolor="black")
    ax.set_xlabel("Number of fitness scores per gene (all)")
    ax.set_ylabel("Gene count")
    ax.set_title(f"Genes vs number of fitness scores — all ({subset})")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histogram_genes_vs_fitness_confident(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    x = genes["n_confident_experiments"].values
    bins = HIST_BINS_EXPERIMENTS
    hist, _ = np.histogram(x, bins=bins)
    labels = [f"{int(b)}–{int(bins[i+1])}" if bins[i+1] != np.inf else f"{int(b)}+" for i, b in enumerate(bins[:-1])]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, hist, color="darkgreen", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Number of confident fitness scores per gene (|t| >= 2)")
    ax.set_ylabel("Gene count")
    ax.set_title(f"Genes vs number of fitness scores — confident only ({subset})")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histogram_fitness_scores(fitness: pd.DataFrame, subset: str, output_path: Path) -> None:
    fit = fitness["fit"].dropna()
    if len(fit) > 1_000_000:
        fit = fit.sample(frac=0.1, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(fit, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(x=ESSENTIALITY_FIT_THRESHOLD, color="red", linestyle="--", label=f"Threshold ({ESSENTIALITY_FIT_THRESHOLD})")
    ax.set_xlabel("Fitness (log₂)")
    ax.set_ylabel("Count")
    ax.set_title(f"Fitness score distribution ({subset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histogram_t_statistic(fitness: pd.DataFrame, subset: str, output_path: Path) -> None:
    t = fitness["t"].dropna()
    if len(t) > 1_000_000:
        t = t.sample(frac=0.1, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(t, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(x=-CONFIDENT_T_THRESHOLD, color="red", linestyle="--", label=f"Confident threshold (±{CONFIDENT_T_THRESHOLD})")
    ax.axvline(x=CONFIDENT_T_THRESHOLD, color="red", linestyle="--")
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Count")
    ax.set_title(f"t-statistic distribution ({subset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histogram_cor12(experiments: pd.DataFrame, subset: str, output_path: Path) -> None:
    cor = experiments["cor12"].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cor, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(x=0.2, color="red", linestyle="--", label="MVP threshold (0.2)")
    ax.set_xlabel("Replicate correlation (cor12)")
    ax.set_ylabel("Experiment count")
    ax.set_title(f"cor12 distribution ({subset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_experiments_per_organism(experiments: pd.DataFrame, subset: str, output_path: Path) -> None:
    counts = experiments.groupby("orgId").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(counts.index, counts.values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Number of experiments")
    ax.set_ylabel("Organism")
    ax.set_title(f"Experiments per organism ({subset})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_essentiality_class_by_organism(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    counts = genes.groupby(["orgId", "essentiality_class"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=ESSENTIALITY_CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    counts.plot(kind="barh", stacked=True, ax=ax, color=sns.color_palette("husl", 4))
    ax.set_xlabel("Gene count")
    ax.set_ylabel("Organism")
    ax.set_title(f"Essentiality class by organism ({subset})")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_genes_per_organism(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    counts = genes.groupby("orgId").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(counts.index, counts.values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("Organism")
    ax.set_title(f"Genes per organism ({subset})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap_media_condition(
    experiments: pd.DataFrame,
    subset: str,
    output_path: Path,
    top_media: int = 30,
    top_conditions: int = 30,
) -> None:
    exp = experiments.copy()
    exp["media"] = exp["media"].fillna("unknown")
    exp["condition_1"] = exp["condition_1"].fillna("unknown")
    top_m = exp["media"].value_counts().head(top_media).index
    top_c = exp["condition_1"].value_counts().head(top_conditions).index
    exp = exp[exp["media"].isin(top_m) & exp["condition_1"].isin(top_c)]
    pivot = exp.groupby(["media", "condition_1"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, annot=False, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Experiments: media vs condition ({subset})")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Media")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_conditions_by_count(
    experiments: pd.DataFrame, subset: str, output_path: Path, top_n: int = 20
) -> None:
    exp = experiments.copy()
    exp["condition_1"] = exp["condition_1"].fillna("unknown")
    counts = exp["condition_1"].value_counts().head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(counts.index, counts.values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Number of experiments")
    ax.set_ylabel("Condition")
    ax.set_title(f"Top conditions by experiment count ({subset})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter_n_confident_vs_frac_essential(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    df = genes[genes["n_confident_experiments"] > 0].copy()
    df = df.dropna(subset=["frac_essential_confident"])
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No genes with confident experiments", ha="center", va="center")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["n_confident_experiments"], df["frac_essential_confident"], alpha=0.3, s=5)
    ax.set_xlabel("Number of confident experiments")
    ax.set_ylabel("Fraction essential (confident only)")
    ax.set_title(f"n_confident vs frac_essential_confident ({subset})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scatter_frac_essential_all_vs_confident(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    df = genes[genes["n_confident_experiments"] > 0].copy()
    df = df.dropna(subset=["frac_essential_confident", "frac_essential_all"])
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No genes with confident experiments", ha="center", va="center")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["frac_essential_all"], df["frac_essential_confident"], alpha=0.3, s=5)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="y=x")
    ax.set_xlabel("Fraction essential (all experiments)")
    ax.set_ylabel("Fraction essential (confident only)")
    ax.set_title(f"frac_essential_all vs frac_essential_confident ({subset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_organism_class_summary(genes: pd.DataFrame, subset: str, output_path: Path) -> None:
    """Two-panel figure: gene counts per organism + normalized class percentages.

    Left panel: horizontal bar chart of total gene count per organism.
    Right panel: 100% stacked horizontal bar chart of essentiality class fractions.
    Organisms are sorted by total gene count (descending top-to-bottom).
    """
    counts = (
        genes.groupby(["orgId", "essentiality_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ESSENTIALITY_CLASSES, fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=True)

    pct = counts[ESSENTIALITY_CLASSES].div(counts["total"], axis=0) * 100

    class_colors = dict(zip(ESSENTIALITY_CLASSES, sns.color_palette("husl", 4)))
    organisms = counts.index.tolist()
    y = np.arange(len(organisms))

    fig, (ax_count, ax_pct) = plt.subplots(1, 2, figsize=(16, max(8, len(organisms) * 0.35)), sharey=True)

    ax_count.barh(y, counts["total"].values, color="steelblue", edgecolor="black", linewidth=0.3)
    ax_count.set_yticks(y)
    ax_count.set_yticklabels(organisms, fontsize=8)
    ax_count.set_xlabel("Total genes")
    ax_count.set_title("Gene count per organism")
    for i, v in enumerate(counts["total"].values):
        ax_count.text(v + counts["total"].max() * 0.01, i, str(v), va="center", fontsize=7)

    left = np.zeros(len(organisms))
    for cls in ESSENTIALITY_CLASSES:
        widths = pct[cls].values
        ax_pct.barh(y, widths, left=left, color=class_colors[cls], label=cls, edgecolor="white", linewidth=0.3)
        left += widths
    ax_pct.set_xlim(0, 100)
    ax_pct.set_xlabel("Percentage of genes")
    ax_pct.set_title("Essentiality class distribution (%)")
    ax_pct.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig.suptitle(f"Per-organism gene counts and class distribution ({subset})", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_processed_vs_mvp_comparison(
    genes_proc: pd.DataFrame,
    genes_mvp: pd.DataFrame,
    exps_proc: pd.DataFrame,
    exps_mvp: pd.DataFrame,
    fitness_proc: pd.DataFrame,
    fitness_mvp: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ec_proc = genes_proc["essentiality_class"].value_counts()
    ec_mvp = genes_mvp["essentiality_class"].value_counts()
    x = np.arange(len(ESSENTIALITY_CLASSES))
    w = 0.35
    for i, c in enumerate(ESSENTIALITY_CLASSES):
        axes[0, 0].bar(x[i] - w/2, ec_proc.get(c, 0), w, label="processed" if i == 0 else "", color="steelblue")
        axes[0, 0].bar(x[i] + w/2, ec_mvp.get(c, 0), w, label="mvp" if i == 0 else "", color="coral")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(ESSENTIALITY_CLASSES, rotation=20, ha="right")
    axes[0, 0].set_ylabel("Gene count")
    axes[0, 0].set_title("Essentiality class distribution")
    axes[0, 0].legend()

    fit_proc = fitness_proc["fit"].dropna()
    fit_mvp = fitness_mvp["fit"].dropna()
    if len(fit_proc) > 500_000:
        fit_proc = fit_proc.sample(500_000, random_state=42)
    if len(fit_mvp) > 500_000:
        fit_mvp = fit_mvp.sample(500_000, random_state=42)
    axes[0, 1].hist(fit_proc, bins=60, alpha=0.5, label="processed", color="steelblue", density=True)
    axes[0, 1].hist(fit_mvp, bins=60, alpha=0.5, label="mvp", color="coral", density=True)
    axes[0, 1].axvline(x=ESSENTIALITY_FIT_THRESHOLD, color="red", linestyle="--")
    axes[0, 1].set_xlabel("Fitness (log₂)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Fitness distribution")
    axes[0, 1].legend()

    cor_proc = exps_proc["cor12"].dropna()
    cor_mvp = exps_mvp["cor12"].dropna()
    axes[1, 0].hist(cor_proc, bins=40, alpha=0.5, label="processed", color="steelblue", density=True)
    axes[1, 0].hist(cor_mvp, bins=40, alpha=0.5, label="mvp", color="coral", density=True)
    axes[1, 0].axvline(x=0.2, color="red", linestyle="--", label="MVP threshold")
    axes[1, 0].set_xlabel("Replicate correlation (cor12)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("cor12 distribution")
    axes[1, 0].legend()

    n_exp_proc = exps_proc.groupby("orgId").size()
    n_exp_mvp = exps_mvp.groupby("orgId").size()
    orgs = sorted(set(n_exp_proc.index) | set(n_exp_mvp.index))
    y = np.arange(len(orgs))
    axes[1, 1].barh(y - 0.2, [n_exp_proc.get(o, 0) for o in orgs], 0.4, label="processed", color="steelblue")
    axes[1, 1].barh(y + 0.2, [n_exp_mvp.get(o, 0) for o in orgs], 0.4, label="mvp", color="coral")
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels(orgs, fontsize=6)
    axes[1, 1].set_xlabel("Number of experiments")
    axes[1, 1].set_title("Experiments per organism")
    axes[1, 1].legend()

    fig.suptitle("Processed vs MVP comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create data visualizations for GeneEssentiality")
    parser.add_argument(
        "--subset",
        choices=["processed", "mvp", "all"],
        default="all",
        help="Which subset(s) to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "figures",
        help="Output directory for figures",
    )
    parser.add_argument("--top-conditions", type=int, default=30, help="Top N conditions for heatmaps")
    parser.add_argument("--top-media", type=int, default=40, help="Top N media for heatmaps")
    args = parser.parse_args()

    subsets = ["processed", "mvp"] if args.subset == "all" else [args.subset]

    for subset in subsets:
        out_dir = args.output_dir / subset
        out_dir.mkdir(parents=True, exist_ok=True)

        genes = load_genes(subset)
        experiments = load_experiments(subset)
        fitness = load_fitness(subset)

        plot_essentiality_distribution(genes, subset, out_dir / "01_essentiality_class_distribution.png")
        plot_heatmap_condition_organism(
            experiments, subset, out_dir / "02_heatmap_condition_organism.png", args.top_conditions
        )
        plot_heatmap_media_organism(
            experiments, subset, out_dir / "03_heatmap_media_organism.png", args.top_media
        )
        plot_histogram_genes_vs_experiments(genes, subset, out_dir / "04_histogram_genes_vs_experiments.png")
        plot_histogram_genes_vs_fitness_all(genes, subset, out_dir / "05a_histogram_genes_vs_fitness_all.png")
        plot_histogram_genes_vs_fitness_confident(
            genes, subset, out_dir / "05b_histogram_genes_vs_fitness_confident.png"
        )
        plot_histogram_fitness_scores(fitness, subset, out_dir / "06_histogram_fitness_scores.png")
        plot_histogram_t_statistic(fitness, subset, out_dir / "07_histogram_t_statistic.png")
        plot_histogram_cor12(experiments, subset, out_dir / "08_histogram_cor12.png")
        plot_experiments_per_organism(experiments, subset, out_dir / "09_experiments_per_organism.png")
        plot_essentiality_class_by_organism(genes, subset, out_dir / "10_essentiality_class_by_organism.png")
        plot_genes_per_organism(genes, subset, out_dir / "11_genes_per_organism.png")
        plot_heatmap_media_condition(
            experiments,
            subset,
            out_dir / "12_heatmap_media_condition.png",
            top_media=args.top_media,
            top_conditions=args.top_conditions,
        )
        plot_top_conditions_by_count(
            experiments, subset, out_dir / "13_top_conditions_by_count.png", top_n=20
        )
        plot_scatter_n_confident_vs_frac_essential(
            genes, subset, out_dir / "14_scatter_n_confident_vs_frac_essential.png"
        )
        plot_scatter_frac_essential_all_vs_confident(
            genes, subset, out_dir / "15_scatter_frac_essential_all_vs_confident.png"
        )
        plot_organism_class_summary(
            genes, subset, out_dir / "16_organism_class_summary.png"
        )
        print(f"Wrote {len(list(out_dir.glob('*.png')))} figures to {out_dir}")

    if args.subset == "all":
        comp_dir = args.output_dir / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        genes_proc = load_genes("processed")
        genes_mvp = load_genes("mvp")
        exps_proc = load_experiments("processed")
        exps_mvp = load_experiments("mvp")
        fitness_proc = load_fitness("processed")
        fitness_mvp = load_fitness("mvp")
        plot_processed_vs_mvp_comparison(
            genes_proc, genes_mvp, exps_proc, exps_mvp, fitness_proc, fitness_mvp,
            comp_dir / "17_processed_vs_mvp_comparison.png",
        )
        print(f"Wrote comparison figure to {comp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
