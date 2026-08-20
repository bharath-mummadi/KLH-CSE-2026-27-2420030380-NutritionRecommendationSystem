# KLH-CSE-2026-27-2420030380-NutritionRecommendationSystem

ML-Based Personalized Nutrition Recommendation System

Course: Applied Machine Learning for Text Analysis (24ALT3101)


Team Members: M. Bharath (2420030380), K. Abhinav (2420030263)

Problem Statement

Generic diet plans ignore individual biology, including age, BMI, biomarkers such as glucose and HbA1c, and existing medical conditions. Existing academic models are able to predict calorie and macronutrient needs, but they rarely translate these predictions into real, verified meal plans. Many LLM-based meal generators produce nutrition claims that are not grounded in real food data, creating a risk of inaccurate recommendations. Furthermore, recommendation quality in terms of relevance and ranking is almost never formally evaluated in prior work, leaving a gap in how these systems are assessed.

Objectives

This project aims to develop a predictive model that estimates personalized calorie and macronutrient needs using NHANES health data. It also aims to design a stronger ensemble prediction model by combining LightGBM, XGBoost, and Random Forest to improve accuracy over single-model approaches. In addition, the project seeks to develop a safety mechanism that enforces biomarker-based constraints, such as HbA1c and glucose levels, on the recommendations generated. Finally, it aims to design an explainability module using SHAP so that every recommendation made by the system can be justified.

Literature Survey

A range of prior work was reviewed to establish a strong baseline for comparison. Kumari et al. (IJERT 2024) proposed an ML classification approach for personalized diet recommendation that classifies user health status based on age, meals, exercise, and weight goals, though it lacks long-term adherence evaluation. A separate system published in IJCRT (2024) used content-based filtering combined with contextual ML to support the 70/30 eating rule, but relied solely on content filtering without any collaborative signal or biomarker constraints. The chosen baseline for this project combines regression with an LLM for natural-language preference parsing, using NHANES 2017–18 data, but its LLM-generated meal plans are not grounded in verified data and it offers no explainability or ranking metrics. Other reviewed systems include a multi-model classification comparison (LightGBM, Random Forest, XGBoost, SVC, and MLP) that achieved an F1 score of 0.972 and AUC of 0.997 but lacked explainability and used a single custom dataset; a simple-averaged ensemble of Random Forest, XGBoost, and MLP that reached 87.6% accuracy but did not use proper stacking or report ranking metrics; and Golagana et al.'s (2023) AI-powered dietary recommendation system, which relies on predefined static datasets without dynamic lifestyle adaptation. Alshamrani and Srinivasan (2024) explored deep learning combined with IoT sensors for real-time recommendations, though their approach is hardware-dependent and raises cost and privacy concerns. Nakamura et al. (2024) contributed an analytical review identifying fairness and bias gaps arising from regional and cultural food differences, but proposed no standardized benchmark. A diabetes-specific system published in IJRASET (2024) compared Decision Tree, Random Forest, and Neural Network models for personalized meal plans based on glucose levels, though it remains limited to diabetic patients. Lastly, the MOPI-HFRS framework (2024) combines multi-objective optimization with LLM-based interpretation to balance health-awareness and user preference, but is computationally heavy and has not been evaluated for medical-constraint safety.

Research Gap

Across the reviewed literature, medical conditions are typically used only as a classifier feature rather than as an enforced safety constraint, meaning no existing system uses biomarkers to actively restrict unsafe recommendations. LLM-based meal-plan generation tends to be free-form and is not grounded in verified nutrient data, creating a risk of incorrect nutritional claims. Evaluation across prior work is largely limited to accuracy, F1, and AUC, with no system reporting recommendation-quality metrics such as Precision@K or NDCG. Where ensembles are used, they typically rely on selecting a single best model or a simple average rather than a properly stacked meta-learner. Perhaps most notably, none of the reviewed systems offer any form of explainability, leaving users without any reasoning behind the recommendations they receive.

Innovation, Creativity & Novelty

This project addresses the identified gaps directly. Biomarker-aware constraints are used as hard safety filters rather than soft classifier features, ensuring that conditions such as elevated HbA1c or glucose actively shape recommendations. A stacked ensemble core combining LightGBM, XGBoost, and Random Forest through a meta-learner is used in place of simple averaging or single-model selection. Recommendations are grounded using a retrieval-augmented generation (RAG) approach, ensuring meal plans are built only from verified USDA and Food.com data rather than being freely generated and potentially hallucinated. The system also adopts a hybrid three-dataset design, combining NHANES, USDA, and Food.com data to enable both content-based and collaborative recommendations. Explainability is built in through SHAP, providing a feature-level explanation for every recommendation, and the system is evaluated using rigorous new metrics, including Precision@K, NDCG, nutrient-accuracy, and hallucination-rate, none of which are reported in the reviewed literature.

Feasibility Analysis

The core components of this project are considered low-risk and highly achievable, including direct NHANES data download, merging datasets on the SEQN identifier, training LightGBM, XGBoost, and Random Forest models, applying SHAP for explainability, and building a Streamlit demo application. Moderate effort is expected for tasks such as cleaning NHANES data to handle survey-specific codes and missing values, determining domain-appropriate constraint thresholds, and defining relevance criteria for Precision@K and NDCG evaluation. The most challenging component is the RAG setup, which involves building embeddings and FAISS-based retrieval, reliably grounding LLM output, and managing compute and time constraints when running a local LLM. Overall, the core machine learning pipeline is fully achievable within the project timeline, while the RAG and LLM layer is treated as a stretch goal to be added if time permits.

Datasets Used

The project draws on three datasets. NHANES 2017–2018 provides demographic, body-measurement, biomarker, and medical-condition data, and is available directly from the CDC at wwwn.cdc.gov/nchs/nhanes/continuousnhanes. USDA FoodData Central provides verified nutrient values for individual food items and can be downloaded from fdc.nal.usda.gov/download-datasets.html. Food.com Recipes and Interactions provides recipes, ingredient lists, and user ratings, and is available on Kaggle at kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions.

Tech Stack

The machine learning components are built using Python with scikit-learn, LightGBM, XGBoost, and SHAP. The natural language and retrieval components rely on sentence-transformers and FAISS, alongside an LLM accessed either locally via Ollama or through an API. Data handling is done using pandas and NumPy, and the final system is demonstrated through a Streamlit web application.
Data: pandas, NumPy
Demo: Streamlit
Demo: Streamlit
