"""
constraint_layer.py
--------------------
Applies BMI/weight and biomarker HARD constraints to a predicted
nutrition target. Medical rules override the model instead of sitting
as a soft input feature.
"""

BMI_CALORIE_FACTORS = {
    "Underweight": 1.15,
    "Normal": 1.0,
    "Overweight": 0.85,
    "Obese": 0.75,
}


def bmi_category(bmi):
    if bmi is None:
        return None
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def apply_constraints(predicted_calories, predicted_carbs_g, row):
    """
    row: dict/Series with DIABETIC_FLAG, HYPERTENSION_FLAG, HBA1C,
    CHOLESTEROL_MGDL, and optionally BMI / BMI_CATEGORY.
    Returns adjusted calories, carbs, and constraint messages.
    """
    messages = []
    adj_calories = predicted_calories
    adj_carbs = predicted_carbs_g

    category = row.get("BMI_CATEGORY") or bmi_category(row.get("BMI"))
    factor = BMI_CALORIE_FACTORS.get(category, 1.0)
    if factor != 1.0:
        new_cal = adj_calories * factor
        direction = "raised" if factor > 1 else "reduced"
        messages.append(
            f"Calorie target {direction} from {adj_calories:.0f} to {new_cal:.0f} kcal "
            f"for BMI category {category}."
        )
        adj_calories = new_cal

    if row.get("DIABETIC_FLAG", 0) == 1 or row.get("HBA1C", 0) >= 6.5:
        cap = adj_carbs * 0.75
        if adj_carbs > cap:
            messages.append(
                f"Carb target reduced from {adj_carbs:.0f}g to {cap:.0f}g due to "
                f"elevated HbA1c ({row.get('HBA1C'):.1f}%)."
            )
            adj_carbs = cap

    if row.get("HYPERTENSION_FLAG", 0) == 1 or row.get("CHOLESTEROL_MGDL", 0) >= 240:
        messages.append(
            f"Flagged for reduced saturated-fat / sodium recommendations due to "
            f"elevated cholesterol ({row.get('CHOLESTEROL_MGDL'):.0f} mg/dL)."
        )

    if not messages:
        messages.append("No biomarker constraints triggered; standard targets apply.")

    return adj_calories, adj_carbs, messages


if __name__ == "__main__":
    sample_diabetic = {
        "HBA1C": 7.2, "DIABETIC_FLAG": 1, "CHOLESTEROL_MGDL": 210,
        "HYPERTENSION_FLAG": 0, "BMI": 31.2, "BMI_CATEGORY": "Obese",
    }
    sample_healthy = {
        "HBA1C": 5.1, "DIABETIC_FLAG": 0, "CHOLESTEROL_MGDL": 180,
        "HYPERTENSION_FLAG": 0, "BMI": 22.0, "BMI_CATEGORY": "Normal",
    }

    for label, row in [("Diabetic + obese profile", sample_diabetic), ("Healthy profile", sample_healthy)]:
        cal, carbs, msgs = apply_constraints(2000, 220, row)
        print(f"\n{label}: adj_calories={cal:.0f}, adj_carbs={carbs:.0f}")
        for m in msgs:
            print(" -", m)
