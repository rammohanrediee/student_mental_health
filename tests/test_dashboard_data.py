import json
from pathlib import Path

import pytest

from dashboard_data import load_dashboard_data


def _write_dashboard_artifacts(directory: Path) -> None:
    metrics = {
        "dataset": {"rows": 1000, "features": 8, "target_classes": ["A", "B"]},
        "test_metrics": {"accuracy": 0.55, "macro_f1": 0.52},
        "baseline": {"accuracy": 0.50, "macro_f1": 0.33},
        "cross_validation": {"macro_f1_mean": 0.49, "macro_f1_std": 0.03},
        "feature_importance": [{"feature": "sleep", "importance_mean": 0.04}],
        "limitations": ["Observational data cannot establish causality."],
    }
    quality = {"missing_values_total": 0, "duplicate_rows": 0}
    eda = {"target_distribution": {"A": {"count": 500, "proportion": 0.5}}}
    for name, payload in {
        "metrics.json": metrics,
        "data_quality.json": quality,
        "eda_summary.json": eda,
    }.items():
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def test_load_dashboard_data_returns_validated_artifacts(tmp_path):
    _write_dashboard_artifacts(tmp_path)

    dashboard = load_dashboard_data(tmp_path)

    assert dashboard["metrics"]["dataset"]["rows"] == 1000
    assert dashboard["quality"]["missing_values_total"] == 0
    assert dashboard["eda"]["target_distribution"]["A"]["count"] == 500


def test_load_dashboard_data_reports_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="metrics.json"):
        load_dashboard_data(tmp_path)


def test_load_dashboard_data_rejects_incomplete_metrics(tmp_path):
    _write_dashboard_artifacts(tmp_path)
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing sections"):
        load_dashboard_data(tmp_path)
