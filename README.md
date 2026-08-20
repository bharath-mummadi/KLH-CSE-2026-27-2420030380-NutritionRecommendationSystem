# KLH-CSE-2026-27-2420030380-NutritionRecommendationSystem

ML-Based Personalized Nutrition Recommendation System

Course: Applied Machine Learning for Text Analysis (24ALT3101)

Team Members

M. Bharath — 2420030380
K. Abhinav — 2420030263
Problem Statement
Generic diet plans ignore individual biology — age, BMI, biomarkers (glucose, HbA1c), and medical conditions.
Existing academic models predict calories/macros but rarely turn predictions into real, verified meal plans.
LLM-based meal generators can produce nutrition claims not grounded in real food data.
Recommendation quality (relevance, ranking) is almost never formally evaluated in prior work.
Objectives
To develop a predictive model that estimates personalized calorie and macronutrient needs using NHANES health data.
To design a stronger ensemble model by combining LightGBM, XGBoost, and Random Forest.
To develop a safety mechanism that enforces biomarker-based constraints (HbA1c, glucose) on recommendations.
To design an explainability module using SHAP to justify every recommendation.
Literature Survey
Title / Paper	Model	Outcome	Limitation	Dataset (link)
Personalized Diet Recommendation System Using ML (Kumari et al., IJERT 2024)	ML classification (fitness/obesity status)	Classifies user health status from age, meals, exercise, weight goals	No long-term adherence evaluation; limited dataset validation	Custom/undisclosed — ijert.org
Diet Recommendation System using ML (IJCRT 2024)	Content-based filtering + contextual ML	Personalized recs supporting the 70/30 eating rule	Content-filtering only; no collaborative signal or biomarker constraints	Custom — ijcrt.org/papers/IJCRT2404947
NHANES + LLM Meal-Plan Generation (chosen baseline)	Regression + LLM (NLP parsing)	Numeric calorie/macro prediction + NL-driven meal plan	LLM generation ungrounded; no explainability; no ranking metrics	NHANES 2017–18 — wwwn.cdc.gov/nchs/nhanes
ML Classifiers for Nutrition Recommendation (LightGBM/RF/XGB/SVC/MLP)	Multi-model classification comparison	LightGBM best: F1 0.972, AUC 0.997	No explainability; single custom dataset; no ranking metrics	Custom (age, BMI, conditions) — not public
Ensemble Learning for Nutrition Recommendation (RF+XGB+MLP)	Simple-averaged ensemble	87.6% accuracy, beats individual models	Averaging, not stacked; no Precision@K/NDCG	Curated (final_dataset.csv) — not public
AI-Powered Personalized Dietary Recommendation — Golagana et al. (2023)	ML pipeline (health params + nutrient reqs)	Individualized food suggestions from user profile	Relies on predefined static datasets; no dynamic lifestyle adaptation	Not specified — rjwave.org/ijedr
Deep Learning + IoT for Real-Time Diet Recs — Alshamrani & Srinivasan (2024)	Deep learning + IoT sensor fusion	Real-time recs via continuous glucose/calorie monitoring	Hardware-dependent; high cost/privacy concerns; not reproducible	Sensor-based, proprietary — rjwave.org/ijedr
Fairness & Bias in AI Nutrition Systems — Nakamura et al. (2024)	Analytical/review framework	Identifies fairness/bias gaps from regional & cultural food differences	No standardized benchmark or evaluation protocol proposed	N/A (review) — rjwave.org/ijedr
ML-Powered Dietary Guidance for Diabetes Patients (IJRASET 2024)	Decision Tree / Random Forest / Neural Net comparison	Personalized meal plans using glucose levels & history	Diabetes-specific only; limited generalization	Patient data (custom) — doi.org/10.22214/ijraset.2024.62139
MOPI-HFRS: Multi-Objective Health-Aware Food Rec. + LLM Interpretation (2024)	Multi-objective optimization + LLM explanation	Balances health-awareness & preference; LLM explains reasoning	Computationally heavy; not evaluated for medical-constraint safety	Recipe/interaction data — arxiv.org/pdf/2412.08847
Research Gap
What Exists	The Gap
Medical conditions used only as a classifier feature	No system enforces biomarkers as a hard safety constraint
LLM meal-plan generation is free-form	Not grounded in verified nutrient data — risk of incorrect claims
Evaluation limited to accuracy/F1/AUC	No recommendation-quality metrics (Precision@K, NDCG) reported
Ensembles pick a best model or simple average	No system uses a proper stacked meta-learner
No explainability in any reviewed system	Users get a recommendation with no reasoning behind it
Innovation, Creativity & Novelty
Biomarker-aware constraints — HbA1c, glucose & cholesterol used as hard safety filters, not soft features.
Stacked ensemble core — LightGBM + XGBoost + RF combined via a meta-learner.
RAG-grounded recommendations — meal plans built only from verified USDA/Food.com data, avoiding hallucinated claims.
Hybrid 3-dataset design — NHANES + USDA + Food.com enables content + collaborative recommendations.
Explainable output (SHAP) — feature-level explanation for every recommendation.
Rigorous new metrics — Precision@K, NDCG, nutrient-accuracy, hallucination-rate.
Feasibility Analysis
Easy: NHANES direct download, pandas merge on SEQN, LightGBM/XGBoost/RF training, SHAP explainability, Streamlit demo app.
Medium: NHANES data cleaning (survey codes, missing values), domain-correct constraint thresholds, defining Precision@K/NDCG relevance criteria.
Hard (Stretch): RAG setup (embeddings + FAISS retrieval), reliably grounding LLM output, local LLM compute/time constraints.
Verdict: Core ML pipeline is fully achievable within timeline; RAG/LLM layer is a stretch goal.
Datasets Used
Dataset	Purpose	Link
NHANES 2017–2018	Demographics, body measurements, biomarkers, medical conditions	wwwn.cdc.gov/nchs/nhanes/continuousnhanes
USDA FoodData Central	Verified nutrient values per food item	fdc.nal.usda.gov/download-datasets.html
Food.com Recipes & Interactions	Recipes, ingredients, user ratings	kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
Tech Stack
ML: Python, scikit-learn, LightGBM, XGBoost, SHAP
NLP/RAG: sentence-transformers, FAISS, LLM (via Ollama or API)
Data: pandas, NumPy
Demo: Streamlit
