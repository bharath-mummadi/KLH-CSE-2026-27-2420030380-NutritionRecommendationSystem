"""
server.py
---------
FastAPI backend entry point for SafeNutriRAG.
Connects the ML stacked ensemble, SHAP explainability engine,
biomarker safety constraints, and RAG retrieval pipeline to the web dashboard.
"""
import sys
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib

from preprocess import FEATURE_COLS, engineer_features
from constraint_layer import bmi_category
from shap_explainer import explain_prediction
from rag_pipeline import execute_rag_pipeline, get_catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("safenutrirag.api")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "stacked_model_TARGET_CALORIES.joblib"
WEBSITE_DIR = ROOT / "website"

app = FastAPI(
    title="SafeNutriRAG API",
    description="Personalized Nutrition Recommendation API grounded in NHANES, USDA FoodData Central, and Food.com with SHAP explainability and clinical safety constraints.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
        _MODEL = joblib.load(MODEL_PATH)
        logger.info(f"Loaded stacked ensemble model from {MODEL_PATH}")
    return _MODEL


class UserProfileRequest(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Age in years")
    gender: str = Field(..., description="'M' or 'F'")
    height_cm: float = Field(..., ge=120, le=240, description="Height in centimeters")
    weight_kg: float = Field(..., ge=30, le=250, description="Weight in kilograms")
    glucose_mgdl: float = Field(..., ge=50, le=400, description="Blood glucose in mg/dL")
    cholesterol_mgdl: float = Field(..., ge=80, le=500, description="Total cholesterol in mg/dL")
    hba1c: float = Field(..., ge=3.5, le=16.0, description="HbA1c percentage")
    activity_level: str = Field("moderate", description="'low', 'moderate', or 'high'")
    condition: Optional[str] = Field("none", description="'none', 'diabetes', 'hypertension', or 'both'")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SafeNutriRAG",
        "model_loaded": MODEL_PATH.exists(),
        "catalog_size": len(get_catalog()),
    }


@app.get("/api/catalog")
def get_catalog_endpoint():
    recipes = get_catalog()
    return {
        "count": len(recipes),
        "source": "Food.com Verified Recipes",
        "recipes": recipes[:30],
    }


@app.post("/api/predict")
@app.post("/api/recommend")
def predict_and_recommend(profile: UserProfileRequest):
    try:
        model = get_model()

        # 1. Anthropometric calculation
        height_m = profile.height_cm / 100.0
        bmi_val = round(profile.weight_kg / (height_m ** 2), 2)
        category = bmi_category(bmi_val)

        # 2. Clinical condition flags
        is_diabetic = 1 if (profile.hba1c >= 6.5 or profile.condition in ["diabetes", "both"]) else 0
        is_hypertensive = 1 if (profile.cholesterol_mgdl >= 240 or profile.condition in ["hypertension", "both"]) else 0

        gender_clean = profile.gender.strip().upper()
        if gender_clean not in ["M", "F"]:
            gender_clean = "M"

        act_clean = profile.activity_level.strip().lower()
        if act_clean not in ["low", "moderate", "high"]:
            act_clean = "moderate"

        user_dict = {
            "AGE": profile.age,
            "GENDER": gender_clean,
            "HEIGHT_CM": profile.height_cm,
            "WEIGHT_KG": profile.weight_kg,
            "BMI": bmi_val,
            "BMI_CATEGORY": category,
            "HBA1C": profile.hba1c,
            "GLUCOSE_MGDL": profile.glucose_mgdl,
            "CHOLESTEROL_MGDL": profile.cholesterol_mgdl,
            "ACTIVITY_LEVEL": act_clean,
            "DIABETIC_FLAG": is_diabetic,
            "HYPERTENSION_FLAG": is_hypertensive,
        }

        # 3. Preprocess & Feature Engineering
        input_df = pd.DataFrame([user_dict])
        features_df = engineer_features(input_df)
        X = features_df[FEATURE_COLS]

        # 4. Predict personalized calories with Stacked Ensemble
        raw_pred_cal = float(model.predict(X)[0])

        # 5. Baseline macronutrient targets based on clinical guidelines
        baseline_protein = 1.6 * profile.weight_kg + 12.0
        baseline_carbs = (raw_pred_cal * 0.40) / 4.0
        baseline_fat = (raw_pred_cal * 0.25) / 9.0

        # 6. Compute real SHAP explanation
        shap_res = explain_prediction(features_df, raw_user=user_dict)

        # 7. Execute RAG retrieval & grounded meal planning
        rag_res = execute_rag_pipeline(
            user_row=user_dict,
            predicted_calories=raw_pred_cal,
            predicted_protein=baseline_protein,
            predicted_carbs=baseline_carbs,
            predicted_fat=baseline_fat,
        )

        health_risk = float(features_df["HEALTH_RISK_SCORE"].iloc[0])
        metabolic_risk = float(features_df["METABOLIC_RISK_SCORE"].iloc[0])

        # 8. Assemble complete API response matching contracts
        response_payload = {
            "status": "success",
            "user_profile": {
                "age": int(profile.age),
                "gender": "Male" if gender_clean == "M" else "Female",
                "height_cm": profile.height_cm,
                "weight_kg": profile.weight_kg,
                "bmi": bmi_val,
                "bmi_category": category,
                "activity_level": act_clean.capitalize(),
            },
            "health_indicators": {
                "glucose_mgdl": profile.glucose_mgdl,
                "cholesterol_mgdl": profile.cholesterol_mgdl,
                "hba1c": profile.hba1c,
                "diabetic_flag": is_diabetic,
                "hypertension_flag": is_hypertensive,
                "health_risk_score": round(health_risk, 2),
                "metabolic_risk_score": round(metabolic_risk, 2),
            },
            "predicted_calories": {
                "raw_model_prediction": round(raw_pred_cal, 1),
                "target_calories": rag_res["nutrition"]["daily_target"]["calories"],
                "bmi_adjustment_applied": category in ["Underweight", "Overweight", "Obese"],
            },
            "nutrition": rag_res["nutrition"],
            "targets": rag_res["nutrition"]["daily_target"],
            "meal_plan": rag_res["recommendations"],
            "recommendations": rag_res["recommendations"],
            "shap_explanation": shap_res.get("shap_explanation", []),
            "shap_summary": shap_res.get("summary", ""),
            "shap_available": shap_res.get("available", False),
            "recommendation_evidence": rag_res["retrieved_context"],
            "retrieved_context": rag_res["retrieved_context"],
            "sources": rag_res["sources"],
            "rag_retrieval_flow": rag_res["rag_retrieval_flow"],
            "constraints": rag_res["constraints"],
        }
        return response_payload

    except Exception as exc:
        logger.error(f"Prediction and recommendation error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# Serve index.html at root
@app.get("/")
def serve_index():
    index_file = WEBSITE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "SafeNutriRAG API running. Frontend not found in website/."}


# Mount website static assets if needed
if WEBSITE_DIR.exists():
    app.mount("/website", StaticFiles(directory=str(WEBSITE_DIR)), name="website")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
