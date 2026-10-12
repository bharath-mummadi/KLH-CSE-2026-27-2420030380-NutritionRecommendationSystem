# SafeNutri-RAG: ML-Based Personalized Nutrition Recommendation System

Personalized nutrition recommendation system that predicts a person's
calorie/macronutrient needs from real health data, enforces medical safety
via biomarker-based constraints, and (in its full design) grounds meal
recommendations in verified nutrition data instead of free-form AI
generation.

## Repository Structure
```
data/       Sample/synthetic dataset (stand-in for real NHANES data)
docs/       Methodology, dataset sources, and architecture diagram
reports/    Formal Word reports (Review-1, Review-2, Mini-Hackathon-1)
results/    Generated plots, metrics, and the trained model file
src/        Python source code for the core pipeline
```

## Problem Statement
Generic diet plans ignore individual biology — age, BMI, biomarkers like
glucose and HbA1c, and medical conditions. Existing systems predict
calorie/macro needs but rarely turn these into real, verified meal plans,
and none provide an explanation for their recommendations.

## Objectives
- Predict personalized calorie/macronutrient needs using NHANES health data.
- Build a stronger ensemble model (LightGBM + XGBoost + Random Forest).
- Enforce medical safety using biomarker-based hard constraints.
- Ground meal generation in verified nutrition data (RAG).
- Recommend meals via hybrid content + collaborative filtering.
- Explain every recommendation using SHAP.

## Datasets
| Dataset | Role | Link |
|---|---|---|
| NHANES 2017–2018 | Health/biomarker data | wwwn.cdc.gov/nchs/nhanes/continuousnhanes |
| USDA FoodData Central | Verified nutrient values | fdc.nal.usda.gov/download-datasets.html |
| Food.com Recipes & Interactions | Recipes & ratings | kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions |

Full details: [`docs/dataset_sources.md`](docs/dataset_sources.md)

## System Architecture
![System Architecture](docs/system_architecture.png)

Full methodology: [`docs/methodology.md`](docs/methodology.md)

## What's Implemented vs. Planned

| Component | Status |
|---|---|
| Data pipeline (synthetic stand-in) | ✅ Implemented, executed |
| Feature engineering | ✅ Implemented, executed |
| Stacked ensemble | ✅ Implemented, executed |
| Biomarker constraint layer | ✅ Implemented, tested |
| Feature importance / explainability | ✅ Implemented (SHAP substitute) |
| RAG-grounded recommendation | ⏳ Planned (stretch goal) |
| Hybrid content + collaborative recommendation | ⏳ Planned |
| Streamlit demo | ⏳ Planned |
| Real NHANES / LightGBM / XGBoost / SHAP | ⏳ Swap-in once internet access is available |

## Interactive Website
`website/index.html` — open directly in any browser (no server needed).
Includes a live demo that runs the actual feature-engineering and
constraint-layer logic from `src/` in JavaScript, plus real charts from the
executed results.

## How to Run
```bash
pip install -r requirements.txt
cd src
python generate_synthetic_data.py
python explain_and_evaluate.py   # runs training + evaluation + saves plots to results/
```

## Results (on synthetic data — see docs/methodology.md for caveats)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| HistGradientBoosting (LightGBM-analog) | 213.24 | 170.88 | 0.739 |
| GradientBoosting (XGBoost-analog) | 209.03 | 165.91 | 0.749 |
| Random Forest | 206.63 | 164.11 | 0.755 |
| **Stacked Ensemble (proposed)** | **206.51** | **164.57** | **0.755** |

See `results/` for full plots (predicted-vs-actual, model comparison,
feature importance) and `reports/` for the full write-ups.

## Reports
- `reports/Nutrition_RecSys_Project_Report.docx` — Review-1 (Problem Statement → Project Plan)
- `reports/Nutrition_RecSys_Review2_Report.docx` — Review-2 (Methodology, System Design, Implementation, Results)
- `reports/MiniHackathon1_SafeNutriRAG.docx` — Mini-Hackathon-1 submission (incl. Q&A prep)

## Demo Video
*(Not yet recorded — add a link here once available.)*

## Status
Core ML pipeline is complete and executed. RAG-based grounded
recommendation and the Streamlit demo are stretch goals, planned next.
