"""
preprocess.py
-------------
Loads the project data sources and builds a model-ready feature matrix.

This repo includes three datasets:
- synthetic NHANES-like health records in "data/synthetic_nhanes.csv"
- USDA tables in "data/USDA/*.csv"
- Food.com recipe + interaction data in "data/FOOD/*.csv"
- nutrient reference tables in "data/NHANSE/*.csv"

The combined loader intentionally blends all three sources into a richer
health + nutrition context feature set without breaking the original
synthetic baseline workflow.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEATURE_COLS = [
    "AGE", "BMI", "HBA1C", "GLUCOSE_MGDL", "CHOLESTEROL_MGDL",
    "ACTIVITY_SCORE", "GENDER_M", "DIABETIC_FLAG", "HYPERTENSION_FLAG",
    "HEALTH_RISK_SCORE", "METABOLIC_RISK_SCORE",
    "FOOD_REF_CALORIES", "FOOD_REF_PROTEIN_G", "FOOD_REF_CARBS_G", "FOOD_REF_FAT_G",
    "NUTRIENT_DENSITY_SCORE", "RECIPE_RATING_MEAN", "RECIPE_RATING_COUNT",
    "RECIPE_CALORIE_LEVEL", "MEAL_QUALITY_SCORE", "USDA_CONTEXT_SCORE",
]
TARGET_COLS = ["TARGET_CALORIES", "TARGET_PROTEIN_G", "TARGET_CARBS_G", "TARGET_FAT_G"]


def _resolve_path(path=None):
    if path is None:
        return DATA_DIR / "synthetic_nhanes.csv"

    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def load_data(path=None):
    resolved = _resolve_path(path)
    if not resolved.exists():
        for candidate in [DATA_DIR / "synthetic_nhanes.csv", ROOT / "data" / "synthetic_nhanes.csv"]:
            if candidate.exists():
                return pd.read_csv(candidate)
        raise FileNotFoundError(f"Unable to locate dataset at {resolved}")
    return pd.read_csv(resolved)


_NHANSE_CACHE = None
_RECIPE_CACHE = None
_USDA_CACHE = None


def _load_nhanse_reference():
    global _NHANSE_CACHE
    if _NHANSE_CACHE is not None:
        return _NHANSE_CACHE

    nutrient_frames = []
    for name in ["train.csv", "test.csv"]:
        p = DATA_DIR / "NHANSE" / name
        if p.exists():
            nutrient_frames.append(pd.read_csv(p))

    if not nutrient_frames:
        _NHANSE_CACHE = pd.Series({
            "Energy_kcal": 200.0,
            "Protein_g": 15.0,
            "Carb_g": 25.0,
            "Fat_g": 10.0,
            "Fiber_g": 3.0,
            "Sugar_g": 8.0,
        })
        return _NHANSE_CACHE

    combined = pd.concat(nutrient_frames, ignore_index=True)
    summary = combined[["Energy_kcal", "Protein_g", "Carb_g", "Fat_g", "Fiber_g", "Sugar_g"]].mean()
    summary["NUTRIENT_DENSITY_SCORE"] = (
        (summary["Protein_g"] / (summary["Energy_kcal"] + 1.0)) * 100
        + (summary["Fiber_g"] * 2.0)
        - (summary["Sugar_g"] / (summary["Energy_kcal"] + 1.0)) * 25
    )
    _NHANSE_CACHE = summary
    return _NHANSE_CACHE


def _load_recipe_context():
    global _RECIPE_CACHE
    if _RECIPE_CACHE is not None:
        return _RECIPE_CACHE

    food_dir = DATA_DIR / "FOOD"
    interaction_frames = []
    for name in ["interactions_train.csv", "interactions_validation.csv", "interactions_test.csv"]:
        p = food_dir / name
        if p.exists():
            interaction_frames.append(pd.read_csv(p, low_memory=False))

    if not interaction_frames:
        _RECIPE_CACHE = {
            "RECIPE_RATING_MEAN": 4.1,
            "RECIPE_RATING_COUNT": 40.0,
            "RECIPE_CALORIE_LEVEL": 0.5,
            "MEAL_QUALITY_SCORE": 3.5,
        }
        return _RECIPE_CACHE

    interactions = pd.concat(interaction_frames, ignore_index=True)
    rating_summary = interactions.groupby("recipe_id")["rating"].agg(["mean", "count"]).reset_index()
    rating_summary["mean"] = rating_summary["mean"].fillna(4.0)
    rating_summary["count"] = rating_summary["count"].fillna(1)

    recipe_meta = pd.DataFrame()
    pp_path = food_dir / "PP_recipes.csv"
    if pp_path.exists():
        recipe_meta = pd.read_csv(pp_path, usecols=["id", "calorie_level"], low_memory=False)
        recipe_meta = recipe_meta.rename(columns={"id": "recipe_id"})

    if not recipe_meta.empty:
        rating_summary = rating_summary.merge(recipe_meta, on="recipe_id", how="left")
    else:
        rating_summary["calorie_level"] = 0.5

    rating_summary["calorie_level"] = rating_summary["calorie_level"].fillna(0.5)
    recipe_rating_mean = float(rating_summary["mean"].mean())
    recipe_rating_count = float(rating_summary["count"].mean())
    recipe_calorie_level = float(rating_summary["calorie_level"].mean())
    meal_quality = 0.55 + (recipe_rating_mean - 3.5) * 0.9 + np.log1p(recipe_rating_count) * 0.25

    _RECIPE_CACHE = {
        "RECIPE_RATING_MEAN": recipe_rating_mean,
        "RECIPE_RATING_COUNT": recipe_rating_count,
        "RECIPE_CALORIE_LEVEL": recipe_calorie_level,
        "MEAL_QUALITY_SCORE": float(meal_quality),
    }
    return _RECIPE_CACHE


def _load_usda_context():
    global _USDA_CACHE
    if _USDA_CACHE is not None:
        return _USDA_CACHE

    usda_dir = DATA_DIR / "USDA"
    csv_paths = sorted(usda_dir.glob("*.csv")) if usda_dir.exists() else []
    if not csv_paths:
        _USDA_CACHE = {"USDA_CONTEXT_SCORE": 0.0}
        return _USDA_CACHE

    row_count = 0
    for path in csv_paths:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="latin1")
            row_count += len(df)
        except Exception:
            continue

    if row_count == 0:
        _USDA_CACHE = {"USDA_CONTEXT_SCORE": 0.0}
        return _USDA_CACHE

    _USDA_CACHE = {"USDA_CONTEXT_SCORE": float(np.clip(row_count / 50000.0, 0.0, 1.0))}
    return _USDA_CACHE


def engineer_features(df):
    df = df.copy()
    df["GENDER_M"] = (df["GENDER"] == "M").astype(int)
    df["ACTIVITY_SCORE"] = df["ACTIVITY_LEVEL"].map({"low": 1, "moderate": 2, "high": 3})

    df["HEALTH_RISK_SCORE"] = (
        df["HBA1C"] * 0.55
        + (df["GLUCOSE_MGDL"] - 90) / 40 * 0.35
        + (df["CHOLESTEROL_MGDL"] - 180) / 80 * 0.20
        + df["DIABETIC_FLAG"] * 1.8
        + df["HYPERTENSION_FLAG"] * 1.6
        + (df["BMI"] - 22) / 10 * 0.25
        + (df["AGE"] - 40) / 25 * 0.10
    ).clip(0, 12)

    df["METABOLIC_RISK_SCORE"] = (
        df["HBA1C"] * 0.65
        + (df["GLUCOSE_MGDL"] >= 140).astype(int) * 2.0
        + (df["CHOLESTEROL_MGDL"] >= 240).astype(int) * 1.8
        + df["DIABETIC_FLAG"] * 2.2
        + df["HYPERTENSION_FLAG"] * 1.8
    ).clip(0, 12)

    nutrient_summary = _load_nhanse_reference()
    recipe_summary = _load_recipe_context()
    usda_summary = _load_usda_context()

    df["FOOD_REF_CALORIES"] = float(nutrient_summary.get("Energy_kcal", 200.0))
    df["FOOD_REF_PROTEIN_G"] = float(nutrient_summary.get("Protein_g", 15.0))
    df["FOOD_REF_CARBS_G"] = float(nutrient_summary.get("Carb_g", 25.0))
    df["FOOD_REF_FAT_G"] = float(nutrient_summary.get("Fat_g", 10.0))
    df["NUTRIENT_DENSITY_SCORE"] = float(nutrient_summary.get("NUTRIENT_DENSITY_SCORE", 30.0))
    df["RECIPE_RATING_MEAN"] = float(recipe_summary.get("RECIPE_RATING_MEAN", 4.1))
    df["RECIPE_RATING_COUNT"] = float(recipe_summary.get("RECIPE_RATING_COUNT", 40.0))
    df["RECIPE_CALORIE_LEVEL"] = float(recipe_summary.get("RECIPE_CALORIE_LEVEL", 0.5))
    df["MEAL_QUALITY_SCORE"] = float(recipe_summary.get("MEAL_QUALITY_SCORE", 3.5))
    df["USDA_CONTEXT_SCORE"] = float(usda_summary.get("USDA_CONTEXT_SCORE", 0.0))

    def bmi_cat(b):
        if b < 18.5:
            return "Underweight"
        if b < 25:
            return "Normal"
        if b < 30:
            return "Overweight"
        return "Obese"

    df["BMI_CATEGORY"] = df["BMI"].apply(bmi_cat)

    return df


def load_combined_data(path=None):
    df = load_data(path)
    return engineer_features(df)


def get_feature_target_matrices(df, target_col="TARGET_CALORIES"):
    X = df[FEATURE_COLS]
    y = df[target_col]
    return X, y


if __name__ == "__main__":
    df = load_combined_data()
    print(df[FEATURE_COLS + ["BMI_CATEGORY"] + TARGET_COLS].head())
    print("\nMissing values:\n", df[FEATURE_COLS].isna().sum())
