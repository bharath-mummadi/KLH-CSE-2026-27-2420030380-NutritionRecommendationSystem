# KLH-CSE-2026-27-2420030380-NutritionRecommendationSystem

Abstract
Generic diet plans ignore individual biology — age, BMI, biomarkers like glucose and HbA1c, and medical conditions. Most existing AI nutrition systems either stop at predicting calorie/macro needs, or use LLMs to generate meal plans that aren't grounded in real nutrition data, risking incorrect claims. This project builds a personalized nutrition recommendation system combining a stacked ensemble model (LightGBM + XGBoost + Random Forest) trained on NHANES health data, biomarker-based safety constraints, RAG-grounded meal recommendations using USDA and Food.com data, and SHAP-based explainability — producing recommendations that are personalized, medically safe, verifiable, and explainable.

Problem Statement

Generic plans ignore individual biology and medical conditions.
Predictions rarely turn into real, verified meal plans.
LLM-based generators can hallucinate nutrition facts.
No system explains why a recommendation was made.

Objectives

Predict personalized calorie/macro needs from NHANES data.
Build a stronger ensemble prediction model.
Enforce medical safety using biomarker constraints.
Ground meal generation in verified nutrition data.
Recommend real meals via hybrid (content + collaborative) filtering.
Explain every recommendation using SHAP.

Datasets
NHANES 2017–18 (health/biomarkers) · USDA FoodData Central (nutrients) · Food.com (recipes/ratings)

Innovation
Biomarker hard constraints · stacked ensemble · RAG-grounded recommendations · hybrid recommendation · SHAP explainability · new evaluation metrics (Precision@K, NDCG, hallucination-rate)

Status
Core ML pipeline (ensemble + constraints + explainability) is fully achievable; the RAG/LLM grounding layer is a stretch goal.
