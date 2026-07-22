"""Interactive portfolio dashboard for the student mental-health analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_data import load_dashboard_data


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="Student Mental Health Analysis",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: #f5f8fa; border: 1px solid #e2e8f0;
        border-radius: 0.75rem; padding: 1rem;}
    .insight {background: #eef7f8; border-left: 4px solid #16879b;
        border-radius: 0.35rem; padding: 0.9rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dashboard_data() -> dict[str, dict]:
    return load_dashboard_data(ARTIFACTS)


def show_image(filename: str, caption: str) -> None:
    path = ARTIFACTS / filename
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Run `python run_analysis.py` to generate {filename}.")


try:
    data = get_dashboard_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.code("python run_analysis.py")
    st.stop()

metrics = data["metrics"]
quality = data["quality"]
eda = data["eda"]

st.title("Student Mental Health & Academic Performance")
st.caption(
    "A reproducible, leakage-safe multiclass analysis of reported academic "
    "performance changes during online learning."
)

overview_tab, eda_tab, model_tab, quality_tab, limits_tab = st.tabs(
    ["Overview", "Exploration", "Model Results", "Data Quality", "Responsible Use"]
)

with overview_tab:
    dataset = metrics["dataset"]
    columns = st.columns(4)
    columns[0].metric("Students", f'{dataset["rows"]:,}')
    columns[1].metric("Model features", dataset["features"])
    columns[2].metric("Outcome classes", len(dataset["target_classes"]))
    columns[3].metric("Missing values", quality["missing_values_total"])
    st.markdown(
        '<div class="insight"><strong>Key finding:</strong> the model improves '
        "macro F1 over the majority baseline, but overall predictive performance "
        "remains weak. The dashboard reports this negative result transparently.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Outcome balance")
    distribution = pd.DataFrame.from_dict(
        eda["target_distribution"], orient="index"
    ).rename_axis("Outcome")
    st.bar_chart(distribution["count"], color="#16879b")

with eda_tab:
    st.subheader("Exploratory analysis")
    left, right = st.columns(2)
    with left:
        show_image("target_distribution.png", "Reported academic-performance outcomes")
    with right:
        show_image(
            "numeric_features_by_target.png",
            "Numeric feature distributions grouped by outcome",
        )
    st.subheader("Average numeric values by outcome")
    means = pd.DataFrame.from_dict(
        eda["numeric_means_by_target"], orient="index"
    ).rename_axis("Outcome")
    st.dataframe(means.style.format("{:.2f}"), use_container_width=True)

with model_tab:
    test_metrics = metrics["test_metrics"]
    baseline = metrics["baseline"]
    cv = metrics["cross_validation"]
    columns = st.columns(4)
    columns[0].metric("Test accuracy", f'{test_metrics["accuracy"]:.1%}')
    columns[1].metric(
        "Macro F1",
        f'{test_metrics["macro_f1"]:.1%}',
        f'{test_metrics["macro_f1"] - baseline["macro_f1"]:+.1%} vs baseline',
    )
    columns[2].metric("Baseline accuracy", f'{baseline["accuracy"]:.1%}')
    columns[3].metric(
        "Cross-validation F1",
        f'{cv["macro_f1_mean"]:.1%}',
        f'± {cv["macro_f1_std"]:.1%}',
    )
    left, right = st.columns(2)
    with left:
        show_image("confusion_matrix.png", "Held-out confusion matrix")
    with right:
        show_image("feature_importance.png", "Held-out permutation importance")
    importance = pd.DataFrame(metrics["feature_importance"])
    st.subheader("Feature-importance estimates")
    st.dataframe(importance, hide_index=True, use_container_width=True)

with quality_tab:
    columns = st.columns(3)
    columns[0].metric("Rows audited", f'{quality["rows"]:,}')
    columns[1].metric("Duplicate rows", quality["duplicate_rows"])
    columns[2].metric("Missing values", quality["missing_values_total"])
    st.subheader("Numeric feature profile")
    numeric_profile = pd.DataFrame.from_dict(
        quality["numeric_summary"], orient="index"
    ).rename_axis("Feature")
    st.dataframe(numeric_profile, use_container_width=True)
    with st.expander("Schema and categorical cardinality"):
        st.json(
            {
                "dtypes": quality["dtypes"],
                "categorical_cardinality": quality["categorical_cardinality"],
            }
        )

with limits_tab:
    st.subheader("Interpretation boundaries")
    for limitation in metrics["limitations"]:
        st.warning(limitation, icon="⚠️")
    st.markdown(
        "This portfolio project is an educational analysis—not a diagnostic, "
        "clinical, admissions, or student-monitoring system. Associations do not "
        "establish causation, and external validation is required before reuse."
    )
