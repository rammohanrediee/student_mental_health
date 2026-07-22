"""Command-line entry point for the student mental-health analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mental_health_pipeline import train_and_evaluate


DEFAULT_DATASET = "Student Mental Health Analysis During Online Learning.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the student mental-health classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(DEFAULT_DATASET),
        help=f"Path to the source CSV (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for metrics, plots, and the fitted model.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_and_evaluate(
        args.data, args.output_dir, random_state=args.random_state
    )
    summary = {
        "rows": results["dataset"]["rows"],
        "accuracy": results["test_metrics"]["accuracy"],
        "macro_f1": results["test_metrics"]["macro_f1"],
        "baseline_accuracy": results["baseline"]["accuracy"],
        "cv_macro_f1_mean": results["cross_validation"]["macro_f1_mean"],
        "artifacts": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

