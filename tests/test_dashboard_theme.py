from pathlib import Path


def test_dashboard_css_uses_theme_aware_colors():
    source = (Path(__file__).parents[1] / "streamlit_app.py").read_text(
        encoding="utf-8"
    )

    assert "var(--secondary-background-color)" in source
    assert "var(--text-color)" in source
    assert "var(--primary-color)" in source
    assert "#f5f8fa" not in source
    assert "#eef7f8" not in source
