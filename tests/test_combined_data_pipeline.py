from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import load_data, load_combined_data


def test_load_data_resolves_repo_root():
    df = load_data()
    assert not df.empty
    assert "TARGET_CALORIES" in df.columns
    assert "BMI" in df.columns


def test_load_combined_data_uses_all_sources():
    df = load_combined_data()
    required = {
        "AGE",
        "BMI",
        "HEALTH_RISK_SCORE",
        "RECIPE_RATING_MEAN",
        "NUTRIENT_DENSITY_SCORE",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing combined-data features: {sorted(missing)}"
    assert len(df) > 0
