"""
explain_and_evaluate.py
------------------------
Produces the Results-section artifacts:
  1. Feature importance chart (permutation importance; substitute for SHAP,
     which needs `pip install shap` - not available offline in this sandbox)
  2. Predicted vs Actual scatter plot
  3. Model comparison bar chart (individual learners vs stacked ensemble)

NOTE: Swap `permutation_importance` for `shap.TreeExplainer` once shap is
installed - it computes a conceptually similar (but more theoretically
grounded) feature-attribution score. Usage is documented at the bottom.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from train_ensemble import train_and_evaluate
from preprocess import FEATURE_COLS

OUT = "../results"


def run():
    results, (X_train, X_test, y_train, y_test, stacked_pred, model) = train_and_evaluate("TARGET_CALORIES")

    # 1) Feature importance (permutation-based, SHAP substitute)
    perm = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(7, 5))
    plt.barh(importance_df["feature"], importance_df["importance_mean"],
              xerr=importance_df["importance_std"], color="#B71E42")
    plt.xlabel("Permutation Importance (increase in RMSE when shuffled)")
    plt.title("Feature Importance — Calorie Prediction\n(SHAP substitute; offline sklearn equivalent)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
    plt.close()

    # 2) Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, stacked_pred, alpha=0.4, color="#028090")
    lims = [min(y_test.min(), stacked_pred.min()), max(y_test.max(), stacked_pred.max())]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlabel("Actual Calorie Target")
    plt.ylabel("Predicted Calorie Target")
    plt.title("Stacked Ensemble: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT}/predicted_vs_actual.png", dpi=150)
    plt.close()

    # 3) Model comparison bar chart
    comp_df = pd.DataFrame(results).T
    plt.figure(figsize=(7, 5))
    x = np.arange(len(comp_df))
    plt.bar(x, comp_df["RMSE"], color=["#00A896"] * 3 + ["#B71E42"])
    plt.xticks(x, comp_df.index, rotation=20)
    plt.ylabel("RMSE (lower is better)")
    plt.title("Base Learners vs Stacked Ensemble — RMSE Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUT}/model_comparison.png", dpi=150)
    plt.close()

    importance_df.to_csv(f"{OUT}/feature_importance.csv", index=False)
    print("Saved plots + CSVs to", OUT)
    print("\nFeature importance ranking:\n", importance_df.sort_values("importance_mean", ascending=False))
    return results, importance_df


if __name__ == "__main__":
    run()

# ---------------------------------------------------------------
# REAL SHAP USAGE (once `pip install shap` succeeds with internet):
#
#   import shap
#   explainer = shap.TreeExplainer(base_learner)   # e.g. the RF or GB model
#   shap_values = explainer.shap_values(X_test)
#   shap.summary_plot(shap_values, X_test)
#   shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
# ---------------------------------------------------------------
