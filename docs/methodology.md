# Methodology

## Overview
SafeNutri-RAG predicts a person's calorie/macronutrient needs, applies
biomarker-based safety constraints, and recommends real, verified meals
through a hybrid content + collaborative + retrieval-grounded approach.

## Pipeline Stages

1. **Data Integration** — NHANES, USDA FoodData Central, and Food.com are
   combined through a merge-and-link pipeline (see `docs/dataset_sources.md`),
   not a single shared key.
2. **Feature Engineering** — BMI, TDEE (Mifflin-St Jeor equation), activity
   score, and biomarker risk flags (diabetic-range HbA1c ≥ 6.5%,
   hypertension-range cholesterol ≥ 240 mg/dL) are derived from raw fields.
3. **Predictive Modelling** — A stacked ensemble (LightGBM + XGBoost +
   Random Forest, combined via a Ridge meta-learner) predicts personalized
   calorie/macro targets. See `src/train_ensemble.py`.
4. **Constraint Enforcement** — Runs independently and in parallel with the
   prediction layer. Biomarker flags hard-filter or adjust the predicted
   targets (e.g., capping carbohydrate targets for diabetic-range HbA1c).
   See `src/constraint_layer.py`.
5. **Grounded Recommendation (RAG)** — Meal generation retrieves verified
   nutrient/recipe data before any natural-language plan is generated,
   preventing hallucinated nutrition claims. *(Planned — stretch goal; not
   yet implemented in this repo.)*
6. **Explainability & Evaluation** — Feature attribution (SHAP in
   production; `permutation_importance` substitute in this offline build)
   is applied to every prediction. Evaluation uses RMSE/MAE/R² for
   prediction accuracy and Precision@K/NDCG for recommendation quality
   *(recommendation-quality metrics planned, not yet implemented)*.
   See `src/explain_and_evaluate.py`.

## Why Prediction and Constraints Are Parallel, Not Sequential
If biomarkers were only a model input feature, an unsafe recommendation
could still emerge if other features statistically outweighed that signal.
Running the constraint layer independently means it can directly override
a prediction. This was verified with test cases in
`src/constraint_layer.py`: a diabetic profile (HbA1c 7.2%) had its
carbohydrate target reduced from 220g to 165g, while a healthy profile
(HbA1c 5.1%) passed through unadjusted.

## Development Environment Note
This repo's development sandbox has no internet access. Two substitutions
were made so the pipeline could be built and executed end-to-end with real
(not fabricated) results:

| In this repo | Production replacement |
|---|---|
| Synthetic NHANES-like data (`generate_synthetic_data.py`) | Real NHANES `.XPT` files |
| `HistGradientBoostingRegressor` / `GradientBoostingRegressor` (sklearn) | `LGBMRegressor` / `XGBRegressor` |
| `sklearn.inspection.permutation_importance` | `shap.TreeExplainer` |

Each swap is a documented one-line change — see comments at the top of the
relevant files in `src/`.
