#!/usr/bin/env python3
"""Run the GeneEssentiality data pipeline.

Usage examples:
    python scripts/run_pipeline.py                           # all steps
    python scripts/run_pipeline.py --step label_embeddings   # single step
    python scripts/run_pipeline.py --from create_fastas      # from step to end
    python scripts/run_pipeline.py --to filter_mvp           # from start to step
    python scripts/run_pipeline.py --from create_fastas --to label_embeddings
    python scripts/run_pipeline.py --config config/custom.yaml
    python scripts/run_pipeline.py --list                    # show step names
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.pipeline import STEP_ORDER, DataPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger for console output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GeneEssentiality data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available steps (in order): {', '.join(STEP_ORDER)}",
    )

    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--step",
        type=str,
        default=None,
        metavar="NAME",
        help="Run a single step by name",
    )
    step_group.add_argument(
        "--list",
        action="store_true",
        help="List available pipeline steps and exit",
    )

    parser.add_argument(
        "--from",
        dest="from_step",
        type=str,
        default=None,
        metavar="NAME",
        help="First step to run (inclusive). Ignored if --step is set.",
    )
    parser.add_argument(
        "--to",
        dest="to_step",
        type=str,
        default=None,
        metavar="NAME",
        help="Last step to run (inclusive). Ignored if --step is set.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to pipeline YAML config (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if args.list:
        print("Pipeline steps (in order):")
        for i, name in enumerate(STEP_ORDER, 1):
            print(f"  {i}. {name}")
        return 0

    config = load_config(args.config)
    pipeline = DataPipeline(config)

    pipeline.run(
        start=args.from_step,
        end=args.to_step,
        single=args.step,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
