#!/usr/bin/env python3
"""
Check train/val/test organism splits from config/model.yaml.

Validates that all MVP organisms are assigned exactly once, and prints
per-split gene counts and class distributions.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml

from src.data_io import load_genes

ESSENTIALITY_CLASSES = ["always_essential", "conditional", "non_essential", "no_data"]


def load_split_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("split", {})


def check_splits(config_path: Path | None = None) -> int:
    if config_path is None:
        config_path = project_root / "config" / "model.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    split = load_split_config(config_path)
    train = set(split.get("train_organisms", []))
    val = set(split.get("val_organisms", []))
    test = set(split.get("test_organisms", []))

    genes = load_genes("mvp")
    all_orgs = set(genes["orgId"].unique())

    # Validation
    print("=== Split validation ===\n")

    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        print(f"ERROR: Organisms in multiple splits: {overlap}")
    else:
        print("OK: No organism overlap between splits.")

    assigned = train | val | test
    missing = all_orgs - assigned
    extra = assigned - all_orgs

    if missing:
        print(f"ERROR: Organisms in data but not in any split: {missing}")
    else:
        print("OK: All organisms assigned to a split.")

    if extra:
        print(f"WARNING: Organisms in config but not in MVP data: {extra}")

    print()

    # Per-split stats
    print("=== Per-split statistics ===\n")

    def stats_for_orgs(org_ids: set) -> dict:
        if not org_ids:
            return {"genes": 0, "always_essential": 0, "conditional": 0, "non_essential": 0, "no_data": 0}
        mask = genes["orgId"].isin(org_ids)
        sub = genes[mask]
        return {
            "genes": len(sub),
            "always_essential": (sub["essentiality_class"] == "always_essential").sum(),
            "conditional": (sub["essentiality_class"] == "conditional").sum(),
            "non_essential": (sub["essentiality_class"] == "non_essential").sum(),
            "no_data": (sub["essentiality_class"] == "no_data").sum(),
        }

    def print_split(name: str, org_ids: set) -> None:
        s = stats_for_orgs(org_ids)
        total = s["genes"]
        if total == 0:
            print(f"{name}: (empty)")
            return
        ae_pct = 100 * s["always_essential"] / total
        cond_pct = 100 * s["conditional"] / total
        ne_pct = 100 * s["non_essential"] / total
        nd_pct = 100 * s["no_data"] / total
        labeled = total - s["no_data"]
        print(f"{name}:")
        print(f"  Organisms: {len(org_ids)} ({sorted(org_ids)})")
        print(f"  Total genes: {total} ({100 * total / len(genes):.1f}% of MVP)")
        print(f"  Labeled (y≠-1): {labeled}")
        print(f"  always_essential: {s['always_essential']:>5d} ({ae_pct:5.1f}%)")
        print(f"  conditional:      {s['conditional']:>5d} ({cond_pct:5.1f}%)")
        print(f"  non_essential:    {s['non_essential']:>5d} ({ne_pct:5.1f}%)")
        print(f"  no_data:          {s['no_data']:>5d} ({nd_pct:5.1f}%)")
        print()

    print_split("Train", train)
    print_split("Val", val)
    print_split("Test", test)

    # Sanity checks
    print("=== Sanity checks ===\n")
    val_s = stats_for_orgs(val)
    test_s = stats_for_orgs(test)
    if val_s["always_essential"] < 50:
        print(f"WARNING: Val has only {val_s['always_essential']} always_essential genes (AUPRC may be noisy).")
    else:
        print(f"OK: Val has {val_s['always_essential']} always_essential genes.")
    if test_s["always_essential"] < 50:
        print(f"WARNING: Test has only {test_s['always_essential']} always_essential genes (AUPRC may be noisy).")
    else:
        print(f"OK: Test has {test_s['always_essential']} always_essential genes.")

    return 0 if not overlap and not missing else 1


if __name__ == "__main__":
    sys.exit(check_splits())
