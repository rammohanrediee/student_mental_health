"""Leakage-safe training pipeline for the student mental-health dataset."""

from __future__ import annotations

import json
import re
from math import ceil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "academic_performance_change"
IDENTIFIER_COLUMNS = ["name"]
REQUIRED_COLUMNS = [
    "name",
    "gender",
    "age",
    "education_level",
    "screen_time_hrs_day",
    "sleep_duration_hrs",
    "physical_activity_hrs_week",
    "stress_level",
    "anxious_before_exams",
    TARGET,
]


def normalize_column_name(name: str) -> str:
    """Convert a human-readable column name to stable snake_case."""
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    result = frame.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result


def load_dataset(path: str | Path, *, enforce_quality: bool = True) -> pd.DataFrame:
    """Load, normalize, and validate the source CSV."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    frame = normalize_columns(pd.read_csv(dataset_path))
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("dataset is empty")
    if enforce_quality and frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("dataset contains missing values in required columns")
    if enforce_quality and frame.duplicated().any():
        raise ValueError("dataset contains duplicate rows")
    return frame


def build_data_quality_report(frame: pd.DataFrame) -> dict:
    """Summarize schema health and feature ranges for auditability."""
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in frame.columns if column not in numeric_columns
    ]
    numeric_summary = {}
    for column in numeric_columns:
        series = frame[column]
        numeric_summary[column] = {
            "min": _rounded(series.min()),
            "max": _rounded(series.max()),
            "mean": _rounded(series.mean()),
            "std": _rounded(series.std()),
        }

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values_total": int(frame.isna().sum().sum()),
        "missing_values_by_column": {
            str(column): int(count)
            for column, count in frame.isna().sum().items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "numeric_summary": numeric_summary,
        "categorical_cardinality": {
            str(column): int(frame[column].nunique(dropna=False))
            for column in categorical_columns
        },
    }


def build_eda_summary(frame: pd.DataFrame) -> dict:
    """Summarize target balance and feature patterns for reproducible EDA."""
    target_counts = frame[TARGET].value_counts().sort_index()
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in IDENTIFIER_COLUMNS
    ]
    numeric_by_target = frame.groupby(TARGET, observed=True)[numeric_columns].mean()

    return {
        "target_distribution": {
            str(label): {
                "count": int(count),
                "proportion": _rounded(count / len(frame)),
            }
            for label, count in target_counts.items()
        },
        "numeric_means_by_target": {
            str(label): {
                str(column): _rounded(value)
                for column, value in values.items()
            }
            for label, values in numeric_by_target.iterrows()
        },
        "numeric_correlations": {
            str(column): {
                str(other): _rounded(value)
                for other, value in values.items()
            }
            for column, values in frame[numeric_columns].corr().iterrows()
        },
    }


def _save_eda_plots(frame: pd.DataFrame, destination: Path) -> None:
    """Persist compact plots for target balance and numeric feature patterns."""
    target_order = sorted(frame[TARGET].unique().tolist())
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.countplot(data=frame, x=TARGET, order=target_order, color="#16879b", ax=axis)
    axis.set_title("Academic Performance Change Distribution")
    axis.set_xlabel("Academic performance change")
    axis.set_ylabel("Students")
    figure.tight_layout()
    figure.savefig(destination / "target_distribution.png", dpi=180)
    plt.close(figure)

    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    melted = frame.melt(
        id_vars=TARGET,
        value_vars=numeric_columns,
        var_name="feature",
        value_name="value",
    )
    grid = sns.catplot(
        data=melted,
        x=TARGET,
        y="value",
        col="feature",
        col_wrap=2,
        kind="box",
        order=target_order,
        sharey=False,
        color="#16879b",
        height=3.4,
    )
    grid.set_axis_labels("Academic performance change", "Value")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Numeric Features by Academic Performance Change", y=1.02)
    grid.figure.savefig(destination / "numeric_features_by_target.png", dpi=180)
    plt.close(grid.figure)


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    """Build preprocessing and classification as one leakage-safe estimator."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": _rounded(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _rounded(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": _rounded(f1_score(y_true, y_pred, average="macro")),
        "weighted_precision": _rounded(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": _rounded(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def _save_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, labels: list[str], destination: Path
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title("Academic Performance Change - Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _save_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    destination: Path,
    random_state: int,
) -> list[dict[str, float | str]]:
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=20,
        random_state=random_state,
        n_jobs=1,
    )
    ranked = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    plot_data = ranked.sort_values("importance_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(
        plot_data["feature"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std"],
        color="#16879b",
        alpha=0.85,
    )
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_title("Permutation Importance on Held-Out Test Data")
    axis.set_xlabel("Change in macro F1 after feature permutation")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)

    return [
        {
            "feature": str(row.feature),
            "importance_mean": _rounded(row.importance_mean),
            "importance_std": _rounded(row.importance_std),
        }
        for row in ranked.itertuples(index=False)
    ]


def train_and_evaluate(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 42,
) -> dict:
    """Train the model, evaluate it, and persist reproducible artifacts."""
    frame = load_dataset(dataset_path, enforce_quality=False)
    data_quality = build_data_quality_report(frame)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data_quality.json").write_text(
        json.dumps(data_quality, indent=2), encoding="utf-8"
    )
    if data_quality["missing_values_total"]:
        raise ValueError(
            "dataset contains missing values; inspect artifacts/data_quality.json"
        )
    if data_quality["duplicate_rows"]:
        raise ValueError(
            "dataset contains duplicate rows; inspect artifacts/data_quality.json"
        )

    eda_summary = build_eda_summary(frame)
    X = frame.drop(columns=[TARGET, *IDENTIFIER_COLUMNS])
    y = frame[TARGET]
    labels = sorted(y.unique().tolist())
    minimum_class_count = int(y.value_counts().min())
    if minimum_class_count < 3:
        raise ValueError("training requires at least 3 rows per target class")
    test_rows = max(ceil(len(frame) * 0.2), len(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_rows,
        random_state=random_state,
        stratify=y,
    )

    cv_folds = min(5, int(y_train.value_counts().min()))
    model = build_pipeline(X_train)
    cross_validation_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    _save_eda_plots(frame, destination)
    _save_confusion_matrix(
        y_test, predictions, labels, destination / "confusion_matrix.png"
    )
    feature_importance = _save_feature_importance(
        model,
        X_test,
        y_test,
        destination / "feature_importance.png",
        random_state,
    )
    joblib.dump(model, destination / "model.joblib")

    results = {
        "data_quality": data_quality,
        "eda": eda_summary,
        "dataset": {
            "rows": int(len(frame)),
            "features": int(X.shape[1]),
            "target_classes": labels,
            "class_distribution": {
                str(label): int(count)
                for label, count in y.value_counts().sort_index().items()
            },
        },
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "stratified": True,
            "random_state": random_state,
        },
        "test_metrics": _classification_metrics(y_test, predictions),
        "baseline": _classification_metrics(y_test, baseline_predictions),
        "cross_validation": {
            "metric": "macro_f1",
            "folds": cv_folds,
            "macro_f1_scores": [_rounded(score) for score in cross_validation_scores],
            "macro_f1_mean": _rounded(cross_validation_scores.mean()),
            "macro_f1_std": _rounded(cross_validation_scores.std()),
        },
        "feature_importance": feature_importance,
        "data_leakage_controls": {
            "split_before_fit": True,
            "preprocessing_fit_on_training_only": True,
            "stratified_split": True,
            "cross_validation_refits_full_pipeline": True,
        },
        "limitations": [
            "Observational data cannot establish causal effects.",
            "Self-reported measures may contain response and recall bias.",
            "Performance should be validated on data from a different institution or time period.",
        ],
    }
    (destination / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (destination / "eda_summary.json").write_text(
        json.dumps(eda_summary, indent=2), encoding="utf-8"
    )
    return results
