"""
train_ensemble.py
------------------
Trains a STACKED ENSEMBLE to predict personalized calorie targets using the
combined health + USDA + recipe dataset context.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import FEATURE_COLS, get_feature_target_matrices, load_combined_data


def build_stacked_model():
    base_learners = [
        ("hist_gb", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42)),
        ("grad_b", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42)),
    ]
    meta_learner = Ridge(alpha=1.0)
    model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner, cv=5, n_jobs=None)
    return model, base_learners


def train_and_evaluate(target_col="TARGET_CALORIES"):
    df = load_combined_data()
    X, y = get_feature_target_matrices(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    stacked_model, base_learners = build_stacked_model()
    stacked_model.fit(X_train, y_train)
    stacked_pred = stacked_model.predict(X_test)

    results = {}
    for name, learner in base_learners:
        learner.fit(X_train, y_train)
        pred = learner.predict(X_test)
        results[name] = {
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        }

    results["stacked_ensemble"] = {
        "RMSE": mean_squared_error(y_test, stacked_pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, stacked_pred),
        "R2": r2_score(y_test, stacked_pred),
    }

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)
    joblib.dump(stacked_model, output_dir / f"stacked_model_{target_col}.joblib")
    return results, (X_train, X_test, y_train, y_test, stacked_pred, stacked_model)


if __name__ == "__main__":
    results, _ = train_and_evaluate("TARGET_CALORIES")
    print("\n=== Model Comparison: Calorie Prediction ===")
    df_results = pd.DataFrame(results).T
    print(df_results.round(3))
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)
    df_results.to_csv(output_dir / "model_comparison_calories.csv")
