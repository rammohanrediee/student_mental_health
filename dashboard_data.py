"""Validated artifact loading for the Streamlit project showcase."""

from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_FILES = {
    "metrics": "metrics.json",
    "quality": "data_quality.json",
    "eda": "eda_summary.json",
}
REQUIRED_METRIC_SECTIONS = {
    "dataset",
    "test_metrics",
    "baseline",
    "cross_validation",
    "feature_importance",
    "limitations",
}


def load_dashboard_data(artifact_dir: str | Path) -> dict[str, dict]:
    """Load dashboard JSON artifacts and fail clearly when they are incomplete."""
    directory = Path(artifact_dir)
    loaded = {}
    for key, filename in ARTIFACT_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"required dashboard artifact not found: {filename}")
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))

    missing_sections = sorted(REQUIRED_METRIC_SECTIONS - loaded["metrics"].keys())
    if missing_sections:
        raise ValueError(
            "metrics.json is missing sections: " + ", ".join(missing_sections)
        )
    return loaded
