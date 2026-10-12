"""
generate_synthetic_data.py
--------------------------
Generates a synthetic dataset with the same FIELD STRUCTURE as the real
NHANES 2017-2018 files (DEMO_J, BMX_J, GHB_J, GLU_J, TCHOL_J, PAQ_J,
DR1TOT_J) so the rest of the pipeline can be built and tested end-to-end
without internet access.

IMPORTANT: This is a stand-in for real data. Replace `load_data()` in
preprocess.py with the actual NHANES .XPT loader (see README) once you
have internet access to download the real files from:
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000  # synthetic participants

def generate():
    seqn = np.arange(1, N + 1)
    age = np.random.randint(18, 80, N)
    gender = np.random.choice(["M", "F"], N)

    height_cm = np.random.normal(165, 10, N).clip(140, 200)
    weight_kg = np.random.normal(72, 15, N).clip(40, 150)
    bmi = weight_kg / (height_cm / 100) ** 2

    # biomarkers correlated loosely with BMI/age for realism
    hba1c = (4.8 + 0.02 * (bmi - 22) + 0.01 * (age - 40) / 10 + np.random.normal(0, 0.4, N)).clip(4.0, 12.0)
    glucose = (85 + 1.5 * (hba1c - 5.0) * 10 + np.random.normal(0, 8, N)).clip(60, 300)
    cholesterol = (180 + 0.8 * (bmi - 22) + np.random.normal(0, 20, N)).clip(120, 320)

    activity_level = np.random.choice(["low", "moderate", "high"], N, p=[0.4, 0.4, 0.2])
    activity_factor = pd.Series(activity_level).map({"low": 1.2, "moderate": 1.5, "high": 1.75}).values

    # medical condition flags derived from biomarkers (as MCQ_J would encode)
    diabetic_flag = (hba1c >= 6.5).astype(int)
    hypertension_flag = (cholesterol >= 240).astype(int)

    # BMR (Mifflin-St Jeor) + TDEE as the "ground truth" calorie target
    is_male = (gender == "M").astype(int)
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 * is_male - 161 * (1 - is_male))
    tdee = bmr * activity_factor + np.random.normal(0, 50, N)

    # macro targets (grams) - realistic, health-aware ranges for adults.
    # Protein is kept higher for body-weight support and medical safety; fat is reduced
    # to avoid excessive intake while preserving essential fatty acids and satiety.
    protein_g = (1.6 * weight_kg + 12 + np.random.normal(0, 6, N))
    carbs_g = (tdee * 0.40) / 4 + np.random.normal(0, 8, N)
    fat_g = (tdee * 0.25) / 9 + np.random.normal(0, 4, N)

    df = pd.DataFrame({
        "SEQN": seqn,
        "AGE": age,
        "GENDER": gender,
        "HEIGHT_CM": height_cm.round(1),
        "WEIGHT_KG": weight_kg.round(1),
        "BMI": bmi.round(2),
        "HBA1C": hba1c.round(2),
        "GLUCOSE_MGDL": glucose.round(1),
        "CHOLESTEROL_MGDL": cholesterol.round(1),
        "ACTIVITY_LEVEL": activity_level,
        "DIABETIC_FLAG": diabetic_flag,
        "HYPERTENSION_FLAG": hypertension_flag,
        "TARGET_CALORIES": tdee.round(0),
        "TARGET_PROTEIN_G": protein_g.clip(30, None).round(1),
        "TARGET_CARBS_G": carbs_g.clip(50, None).round(1),
        "TARGET_FAT_G": fat_g.clip(20, None).round(1),
    })
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("../data/synthetic_nhanes.csv", index=False)
    print("Generated:", df.shape)
    print(df.head())
