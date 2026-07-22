"""Leakage-safe training pipeline for the student mental-health dataset."""

from __future__ import annotations

import json
import re
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


def load_dataset(path: str | Path) -> pd.DataFrame:
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
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("dataset contains missing values in required columns")
    if frame.duplicated().any():
        raise ValueError("dataset contains duplicate rows")
    return frame


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
    frame = load_dataset(dataset_path)
    X = frame.drop(columns=[TARGET, *IDENTIFIER_COLUMNS])
    y = frame[TARGET]
    labels = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline(X_train)
    cross_validation_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro",
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
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
            "folds": 5,
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
    return results
