import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["catalog_size"] > 0


def test_predict_and_recommend_pipeline(client):
    payload = {
        "age": 34,
        "gender": "M",
        "height_cm": 170.0,
        "weight_kg": 78.0,
        "glucose_mgdl": 96.0,
        "cholesterol_mgdl": 190.0,
        "hba1c": 5.4,
        "activity_level": "moderate",
        "condition": "none",
    }
    res = client.post("/api/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()

    # User Profile
    assert data["user_profile"]["age"] == 34
    assert data["user_profile"]["gender"] == "Male"
    assert round(data["user_profile"]["bmi"], 1) == 27.0
    assert data["user_profile"]["bmi_category"] == "Overweight"

    # Health Indicators
    assert data["health_indicators"]["glucose_mgdl"] == 96.0
    assert data["health_indicators"]["hba1c"] == 5.4
    assert data["health_indicators"]["cholesterol_mgdl"] == 190.0

    # Daily Targets
    targets = data["targets"]
    assert "calories" in targets and targets["calories"] > 1500
    assert "protein" in targets and "protein_g" in targets
    assert "carbohydrates" in targets and "carbs_g" in targets
    assert "fat" in targets and "fat_g" in targets
    assert targets["protein"] > 50
    assert targets["carbohydrates"] > 50
    assert targets["fat"] > 20

    # Macro Alignment (Prevent random values / huge drift)
    actual_totals = data["nutrition"]["actual_totals"]
    assert abs(actual_totals["calories"] - targets["calories"]) / targets["calories"] < 0.15
    assert abs(actual_totals["carbohydrates"] - targets["carbohydrates"]) / targets["carbohydrates"] < 0.15
    assert abs(actual_totals["fat"] - targets["fat"]) / targets["fat"] < 0.15
    assert abs(actual_totals["protein"] - targets["protein"]) / targets["protein"] < 0.15

    # Meal Plan
    meals = data["meal_plan"]
    assert len(meals) == 4  # Breakfast, Lunch, Dinner, Snack
    slots = [m["slot"] for m in meals]
    assert slots == ["breakfast", "lunch", "dinner", "snack"]

    for m in meals:
        assert m["calories"] > 0
        assert m["protein"] >= 0
        assert m["protein_g"] >= 0
        assert m["carbohydrates"] >= 0
        assert m["carbs_g"] >= 0
        assert m["fat"] >= 0
        assert m["fat_g"] >= 0
        assert len(m["why_this_recipe"]) > 0
        assert m["recipe"]["name"]

    # SHAP Explanation
    assert data["shap_available"] is True
    shap_list = data["shap_explanation"]
    assert len(shap_list) >= 6
    for item in shap_list:
        assert "feature" in item
        assert "value" in item
        assert "shap_value" in item
        assert item["impact"] in ["positive", "negative"]

    # RAG Retrieval & Sources
    assert len(data["sources"]) == 2
    assert any("Food.com" in s["name"] for s in data["sources"])
    assert any("USDA" in s["name"] for s in data["sources"])
    assert len(data["retrieved_context"]) >= 4
    for r in data["retrieved_context"]:
        assert r["name"]
        assert "calories" in r
        assert "protein_g" in r
        assert "carbs_g" in r
        assert "fat_g" in r
        assert r["source"]


def test_diabetic_profile_hard_constraint(client):
    payload = {
        "age": 52,
        "gender": "F",
        "height_cm": 160.0,
        "weight_kg": 85.0,
        "glucose_mgdl": 165.0,
        "cholesterol_mgdl": 220.0,
        "hba1c": 7.8,
        "activity_level": "low",
        "condition": "diabetes",
    }
    res = client.post("/api/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["health_indicators"]["diabetic_flag"] == 1
    # Check that constraint was triggered
    assert any("Carb" in m or "elevated HbA1c" in m or "BMI" in m for m in data["constraints"])
