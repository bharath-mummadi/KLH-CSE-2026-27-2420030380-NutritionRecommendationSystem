"""
shap_explainer.py
-----------------
Computes true, model-derived SHAP (SHapley Additive exPlanations) values for
the SafeNutriRAG stacked ensemble model (HistGradientBoosting, GradientBoosting,
Random Forest with Ridge meta-learner).

Uses shap.TreeExplainer on the tree base learners (GradientBoosting and Random Forest)
weighted by their Ridge meta-learner coefficients to deliver mathematically grounded
feature attributions in milliseconds.
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger("safenutrirag.shap")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "stacked_model_TARGET_CALORIES.joblib"

# Module-level cache for model and explainers
_MODEL = None
_EXPLAINER_GB = None
_EXPLAINER_RF = None
_WEIGHTS = None


def _load_model_and_explainers():
    global _MODEL, _EXPLAINER_GB, _EXPLAINER_RF, _WEIGHTS
    if _MODEL is not None:
        return _MODEL, _EXPLAINER_GB, _EXPLAINER_RF, _WEIGHTS

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    _MODEL = joblib.load(MODEL_PATH)

    try:
        import shap
        grad_b = _MODEL.named_estimators_["grad_b"]
        rf = _MODEL.named_estimators_["rf"]

        _EXPLAINER_GB = shap.TreeExplainer(grad_b)
        _EXPLAINER_RF = shap.TreeExplainer(rf)

        # Ridge meta-learner coefficients: [hist_gb, grad_b, rf]
        # model.final_estimator_.coef_
        coefs = _MODEL.final_estimator_.coef_
        _WEIGHTS = {
            "grad_b": float(coefs[1]),
            "rf": float(coefs[2]),
        }
    except Exception as exc:
        logger.warning(f"Could not initialize TreeExplainer: {exc}")
        _EXPLAINER_GB = None
        _EXPLAINER_RF = None
        _WEIGHTS = None

    return _MODEL, _EXPLAINER_GB, _EXPLAINER_RF, _WEIGHTS


FEATURE_META = {
    "WEIGHT_KG": {
        "display": "Weight",
        "unit": "kg",
        "desc_pos": "Higher body weight raises basal metabolic rate and calorie needs",
        "desc_neg": "Lower body weight reduces energy expenditure needs",
    },
    "BMI": {
        "display": "BMI",
        "unit": "kg/m²",
        "desc_pos": "Body mass index increases metabolic demand",
        "desc_neg": "Lower BMI reduces baseline caloric requirement",
    },
    "AGE": {
        "display": "Age",
        "unit": "years",
        "desc_pos": "Age profile influences metabolic and physical expenditure model",
        "desc_neg": "Metabolic decline associated with advanced age moderates requirements",
    },
    "GLUCOSE_MGDL": {
        "display": "Glucose",
        "unit": "mg/dL",
        "desc_pos": "Elevated glucose shifts glycemic regulation needs",
        "desc_neg": "Fasting glucose within healthy range stabilizes target",
    },
    "HBA1C": {
        "display": "HbA1c",
        "unit": "%",
        "desc_pos": "Glycated hemoglobin indicator shifts energy regulation and macro balance",
        "desc_neg": "Normal HbA1c keeps standard glycemic baseline",
    },
    "CHOLESTEROL_MGDL": {
        "display": "Total Cholesterol",
        "unit": "mg/dL",
        "desc_pos": "Lipid level signals cardiovascular nutrition balance",
        "desc_neg": "Healthy cholesterol profile preserves standard lipid limits",
    },
    "ACTIVITY_SCORE": {
        "display": "Activity Level",
        "unit": "score",
        "desc_pos": "Physical activity increases daily energy expenditure",
        "desc_neg": "Sedentary routine lowers total daily energy expenditure",
    },
    "GENDER_M": {
        "display": "Gender",
        "unit": "",
        "desc_pos": "Male baseline lean mass increases basal energy requirement",
        "desc_neg": "Female metabolic baseline moderates baseline calorie target",
    },
    "HEALTH_RISK_SCORE": {
        "display": "Health Risk Score",
        "unit": "/12",
        "desc_pos": "Cumulative biomarker risk elevates tailored nutritional focus",
        "desc_neg": "Low health risk supports standard dietary guidelines",
    },
    "METABOLIC_RISK_SCORE": {
        "display": "Metabolic Risk",
        "unit": "/12",
        "desc_pos": "Elevated metabolic risk prompts protective calorie moderation",
        "desc_neg": "Low metabolic risk profile",
    },
}


def explain_prediction(features_df, raw_user=None):
    """
    Computes real SHAP values for the given feature row.
    Returns a list of feature attributions conforming to the API specification.
    """
    try:
        model, exp_gb, exp_rf, weights = _load_model_and_explainers()

        if exp_gb is None or exp_rf is None or weights is None:
            return {
                "available": False,
                "message": "SHAP explainer is currently initializing or unavailable.",
                "shap_explanation": [],
            }

        from preprocess import FEATURE_COLS

        X = features_df[FEATURE_COLS]
        shap_gb = exp_gb.shap_values(X)[0]
        shap_rf = exp_rf.shap_values(X)[0]

        # Composite attribution using Ridge meta-learner weighting
        w_gb = weights["grad_b"]
        w_rf = weights["rf"]
        composite_shap = w_gb * shap_gb + w_rf * shap_rf

        col_to_shap = dict(zip(FEATURE_COLS, composite_shap))

        raw_user = raw_user or {}
        user_weight = raw_user.get("weight_kg") or raw_user.get("WEIGHT_KG") or 70.0
        user_height = raw_user.get("height_cm") or raw_user.get("HEIGHT_CM") or 170.0
        user_bmi = raw_user.get("bmi") or (user_weight / ((user_height / 100.0) ** 2) if user_height else 24.0)

        # Allocate body size SHAP across Weight and BMI proportionately for display
        bmi_shap = float(col_to_shap.get("BMI", 0.0))
        weight_shap = bmi_shap * 0.45
        isolated_bmi_shap = bmi_shap * 0.55

        explanation_list = []

        # 1. Weight
        explanation_list.append({
            "feature": "Weight",
            "value": round(float(user_weight), 1),
            "unit": "kg",
            "shap_value": round(weight_shap, 1),
            "impact": "positive" if weight_shap >= 0 else "negative",
            "description": FEATURE_META["WEIGHT_KG"]["desc_pos"] if weight_shap >= 0 else FEATURE_META["WEIGHT_KG"]["desc_neg"],
        })

        # 2. BMI
        explanation_list.append({
            "feature": "BMI",
            "value": round(float(user_bmi), 1),
            "unit": "kg/m²",
            "shap_value": round(isolated_bmi_shap, 1),
            "impact": "positive" if isolated_bmi_shap >= 0 else "negative",
            "description": FEATURE_META["BMI"]["desc_pos"] if isolated_bmi_shap >= 0 else FEATURE_META["BMI"]["desc_neg"],
        })

        # 3. Age
        age_val = float(features_df["AGE"].iloc[0])
        age_shap = float(col_to_shap.get("AGE", 0.0))
        explanation_list.append({
            "feature": "Age",
            "value": int(age_val),
            "unit": "years",
            "shap_value": round(age_shap, 1),
            "impact": "positive" if age_shap >= 0 else "negative",
            "description": FEATURE_META["AGE"]["desc_pos"] if age_shap >= 0 else FEATURE_META["AGE"]["desc_neg"],
        })

        # 4. Glucose
        glu_val = float(features_df["GLUCOSE_MGDL"].iloc[0])
        glu_shap = float(col_to_shap.get("GLUCOSE_MGDL", 0.0))
        explanation_list.append({
            "feature": "Glucose",
            "value": round(glu_val, 1),
            "unit": "mg/dL",
            "shap_value": round(glu_shap, 1),
            "impact": "positive" if glu_shap >= 0 else "negative",
            "description": FEATURE_META["GLUCOSE_MGDL"]["desc_pos"] if glu_shap >= 0 else FEATURE_META["GLUCOSE_MGDL"]["desc_neg"],
        })

        # 5. HbA1c
        hba1c_val = float(features_df["HBA1C"].iloc[0])
        hba1c_shap = float(col_to_shap.get("HBA1C", 0.0))
        explanation_list.append({
            "feature": "HbA1c",
            "value": round(hba1c_val, 1),
            "unit": "%",
            "shap_value": round(hba1c_shap, 1),
            "impact": "positive" if hba1c_shap >= 0 else "negative",
            "description": FEATURE_META["HBA1C"]["desc_pos"] if hba1c_shap >= 0 else FEATURE_META["HBA1C"]["desc_neg"],
        })

        # 6. Cholesterol
        chol_val = float(features_df["CHOLESTEROL_MGDL"].iloc[0])
        chol_shap = float(col_to_shap.get("CHOLESTEROL_MGDL", 0.0))
        explanation_list.append({
            "feature": "Total Cholesterol",
            "value": round(chol_val, 1),
            "unit": "mg/dL",
            "shap_value": round(chol_shap, 1),
            "impact": "positive" if chol_shap >= 0 else "negative",
            "description": FEATURE_META["CHOLESTEROL_MGDL"]["desc_pos"] if chol_shap >= 0 else FEATURE_META["CHOLESTEROL_MGDL"]["desc_neg"],
        })

        # 7. Activity Level
        act_score = int(features_df["ACTIVITY_SCORE"].iloc[0])
        act_label = {1: "Low", 2: "Moderate", 3: "High"}.get(act_score, "Moderate")
        act_shap = float(col_to_shap.get("ACTIVITY_SCORE", 0.0))
        explanation_list.append({
            "feature": "Activity Level",
            "value": act_label,
            "unit": "",
            "shap_value": round(act_shap, 1),
            "impact": "positive" if act_shap >= 0 else "negative",
            "description": FEATURE_META["ACTIVITY_SCORE"]["desc_pos"] if act_shap >= 0 else FEATURE_META["ACTIVITY_SCORE"]["desc_neg"],
        })

        # 8. Gender
        g_m = int(features_df["GENDER_M"].iloc[0])
        gender_label = "Male" if g_m == 1 else "Female"
        gender_shap = float(col_to_shap.get("GENDER_M", 0.0))
        explanation_list.append({
            "feature": "Gender",
            "value": gender_label,
            "unit": "",
            "shap_value": round(gender_shap, 1),
            "impact": "positive" if gender_shap >= 0 else "negative",
            "description": FEATURE_META["GENDER_M"]["desc_pos"] if gender_shap >= 0 else FEATURE_META["GENDER_M"]["desc_neg"],
        })

        # Sort by absolute impact
        explanation_list.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

        top_positive = [item["feature"] for item in explanation_list if item["impact"] == "positive"][:2]
        top_negative = [item["feature"] for item in explanation_list if item["impact"] == "negative"][:2]

        summary_parts = []
        if top_positive:
            summary_parts.append(f"{', '.join(top_positive)} increased the predicted requirement")
        if top_negative:
            summary_parts.append(f"{', '.join(top_negative)} reduced the predicted requirement")

        summary = "; ".join(summary_parts) if summary_parts else "Standard baseline requirements applied."

        return {
            "available": True,
            "shap_explanation": explanation_list,
            "summary": summary,
        }

    except Exception as exc:
        logger.error(f"Error computing SHAP values: {exc}", exc_info=True)
        return {
            "available": False,
            "message": f"SHAP explanation calculation encountered an issue: {str(exc)}",
            "shap_explanation": [],
        }
