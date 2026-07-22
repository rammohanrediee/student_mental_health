from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mental_health_pipeline import (
    REQUIRED_COLUMNS,
    build_data_quality_report,
    build_eda_summary,
    build_pipeline,
    load_dataset,
    normalize_columns,
    train_and_evaluate,
)


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    rows = 180
    target = np.resize(["Declined", "Improved", "Same"], rows)
    target_effect = pd.Series(target).map({"Declined": 2.0, "Improved": -2.0, "Same": 0.0})
    frame = pd.DataFrame(
        {
            "Name": [f"Student {index}" for index in range(rows)],
            "Gender": np.resize(["Female", "Male", "Other"], rows),
            "Age": 18 + np.arange(rows) % 8,
            "Education Level": np.resize(["BTech", "MSc", "Class 12"], rows),
            "Screen Time (hrs/day)": 6.0 + target_effect + (np.arange(rows) % 5) * 0.1,
            "Sleep Duration (hrs)": 7.0 - target_effect * 0.5,
            "Physical Activity (hrs/week)": 4.0 - target_effect * 0.4,
            "Stress Level": pd.Series(target).map(
                {"Declined": "High", "Improved": "Low", "Same": "Medium"}
            ),
            "Anxious Before Exams": pd.Series(target).map(
                {"Declined": "Yes", "Improved": "No", "Same": "No"}
            ),
            "Academic Performance Change": target,
        }
    )
    path = tmp_path / "student_mental_health.csv"
    frame.to_csv(path, index=False)
    return path


def test_normalize_columns_produces_stable_snake_case_names():
    frame = pd.DataFrame(columns=["Screen Time (hrs/day)", " Anxious Before Exams "])

    normalized = normalize_columns(frame)

    assert normalized.columns.tolist() == [
        "screen_time_hrs_day",
        "anxious_before_exams",
    ]


def test_load_dataset_validates_required_schema(tmp_path):
    bad_data = pd.DataFrame({"Age": [20], "Stress Level": ["Low"]})
    path = tmp_path / "bad.csv"
    bad_data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_load_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        load_dataset(tmp_path / "missing.csv")


@pytest.mark.parametrize("problem", ["empty", "missing", "duplicate"])
def test_load_dataset_rejects_invalid_rows(tmp_path, dataset_path, problem):
    source = pd.read_csv(dataset_path)
    if problem == "empty":
        source = source.iloc[0:0]
    elif problem == "missing":
        source.loc[0, "Age"] = None
    else:
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    path = tmp_path / f"{problem}.csv"
    source.to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_dataset(path)


def test_load_dataset_preserves_all_three_target_classes(dataset_path):
    frame = load_dataset(dataset_path)

    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    assert set(frame["academic_performance_change"].unique()) == {
        "Declined",
        "Improved",
        "Same",
    }


def test_data_quality_report_profiles_numeric_and_categorical_columns(dataset_path):
    frame = load_dataset(dataset_path)

    report = build_data_quality_report(frame)

    assert report["rows"] == 180
    assert report["columns"] == 10
    assert report["missing_values_total"] == 0
    assert report["duplicate_rows"] == 0
    assert report["numeric_summary"]["age"]["min"] == 18.0
    assert report["numeric_summary"]["age"]["max"] == 25.0
    assert report["categorical_cardinality"]["gender"] == 3
    assert report["categorical_cardinality"]["academic_performance_change"] == 3


def test_eda_summary_reports_target_balance_and_numeric_patterns(dataset_path):
    frame = load_dataset(dataset_path)

    summary = build_eda_summary(frame)

    assert summary["target_distribution"]["Declined"] == {
        "count": 60,
        "proportion": 0.3333,
    }
    assert set(summary["numeric_means_by_target"]) == {
        "Declined",
        "Improved",
        "Same",
    }
    assert summary["numeric_means_by_target"]["Declined"]["screen_time_hrs_day"] > (
        summary["numeric_means_by_target"]["Improved"]["screen_time_hrs_day"]
    )
    assert summary["numeric_correlations"]["age"]["age"] == 1.0


def test_pipeline_handles_unseen_categories_without_failure(dataset_path):
    frame = load_dataset(dataset_path)
    X = frame.drop(columns=["academic_performance_change", "name"])
    y = frame["academic_performance_change"]
    pipeline = build_pipeline(X)
    pipeline.fit(X.iloc[:150], y.iloc[:150])
    unseen = X.iloc[[150]].copy()
    unseen.loc[:, "education_level"] = "New Program"

    prediction = pipeline.predict(unseen)

    assert prediction.shape == (1,)
    assert prediction[0] in {"Declined", "Improved", "Same"}


def test_training_generates_metrics_model_and_plots(tmp_path, dataset_path):
    result = train_and_evaluate(dataset_path, tmp_path / "artifacts", random_state=42)

    output_dir = tmp_path / "artifacts"
    assert result["dataset"]["rows"] == 180
    assert result["dataset"]["target_classes"] == ["Declined", "Improved", "Same"]
    assert result["test_metrics"]["macro_f1"] > result["baseline"]["macro_f1"]
    assert 0 <= result["test_metrics"]["macro_f1"] <= 1
    assert len(result["cross_validation"]["macro_f1_scores"]) == 5
    assert result["data_leakage_controls"]["preprocessing_fit_on_training_only"] is True
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "data_quality.json").is_file()
    assert (output_dir / "eda_summary.json").is_file()
    assert (output_dir / "target_distribution.png").is_file()
    assert (output_dir / "numeric_features_by_target.png").is_file()
    assert (output_dir / "model.joblib").is_file()
    assert (output_dir / "confusion_matrix.png").is_file()
    assert (output_dir / "feature_importance.png").is_file()
