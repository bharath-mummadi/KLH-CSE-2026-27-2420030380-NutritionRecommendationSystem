"""
recommend_meals.py
------------------
Builds a one-day diet plan from Food.com recipes, filtered by the
calorie/macro targets and the hard health constraints.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constraint_layer import apply_constraints, bmi_category

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "foodcom_catalog.json"

SLOT_SHARE = {
    "breakfast": 0.25,
    "lunch": 0.32,
    "dinner": 0.33,
    "snack": 0.10,
}


def load_catalog(path=CATALOG_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["recipes"]


def allowed_for_profile(recipe, diabetic, high_chol, slot):
    cat_lower = (recipe.get("category") or "").lower()
    name_lower = (recipe.get("name") or "").lower()
    dessertish = cat_lower in {
        "dessert", "pie", "bar cookie", "drop cookies", "candy",
        "cheesecake", "frozen desserts",
    } or "cake" in name_lower or "pie" in name_lower

    if slot in ("breakfast", "lunch", "dinner") and dessertish:
        return False

    if diabetic:
        if recipe["carbs_g"] > 45 and slot != "snack":
            return False
        if recipe["sugar_g"] > 18:
            return False
        if recipe["carbs_g"] > 25 and slot == "snack":
            return False
        if dessertish:
            return False
    if high_chol:
        if recipe["sat_fat_g"] > 8:
            return False
        if recipe["sodium_mg"] > 700:
            return False
        if recipe["fat_g"] > 28 and slot != "dinner":
            return False
    return True


def score_recipe(recipe, calorie_target, protein_share, diabetic, high_chol, slot=None):
    cal_err = abs(recipe["calories"] - calorie_target) / max(calorie_target, 1)
    protein_bonus = min(recipe["protein_g"] / max(protein_share, 1), 1.5)
    rating = (recipe.get("rating") or 3.5) / 5.0
    health = 0.0
    if diabetic:
        health += max(0, 1 - recipe["carbs_g"] / 50)
        health += max(0, 1 - recipe["sugar_g"] / 20)
    if high_chol:
        health += max(0, 1 - recipe["sat_fat_g"] / 10)
        health += max(0, 1 - recipe["sodium_mg"] / 800)

    affinity = 0.0
    name_and_cat = (recipe.get("name", "") + " " + recipe.get("category", "")).lower()
    if slot == "breakfast":
        if any(w in name_and_cat for w in ["egg", "quiche", "frittata", "omelet", "muffin", "pancake", "waffle", "oatmeal", "scone", "hash brown", "smoothie", "breakfast"]):
            affinity += 0.8
    elif slot == "lunch":
        if any(w in name_and_cat for w in ["soup", "salad", "chowder", "stew", "sandwich", "wrap", "bowl", "chili"]):
            affinity += 0.5
    elif slot == "dinner":
        if any(w in name_and_cat for w in ["chicken", "beef", "salmon", "fish", "pasta", "lasagna", "casserole", "roast", "pork", "steak", "dinner"]):
            affinity += 0.5
    elif slot == "snack":
        if any(w in name_and_cat for w in ["dip", "zucchini", "bites", "snack", "smoothie", "bar", "fruit"]):
            affinity += 0.5

    return -(cal_err * 2.4) + protein_bonus * 0.6 + rating * 0.5 + health * 0.45 + affinity


def pick_slot(recipes, slot, calorie_target, protein_share, diabetic, high_chol, used_ids):
    candidates = [
        r for r in recipes
        if r["slot"] == slot and r["id"] not in used_ids
        and allowed_for_profile(r, diabetic, high_chol, slot)
    ]
    if not candidates:
        candidates = [
            r for r in recipes
            if r["id"] not in used_ids and allowed_for_profile(r, diabetic, high_chol, slot)
        ]
    if not candidates:
        candidates = [r for r in recipes if r["id"] not in used_ids]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda r: score_recipe(r, calorie_target, protein_share, diabetic, high_chol, slot=slot),
        reverse=True,
    )
    return ranked[0]


def select_balanced_day_plan(recipes, target_calories, target_protein, target_carbs, target_fat, diabetic, high_chol):
    by_slot = {}
    for slot, share in SLOT_SHARE.items():
        t_cal = target_calories * share
        t_prot = target_protein * share
        t_carb = target_carbs * share
        t_fat = target_fat * share

        candidates = [
            r for r in recipes
            if r["slot"] == slot and allowed_for_profile(r, diabetic, high_chol, slot)
        ]
        if not candidates:
            candidates = [r for r in recipes if allowed_for_profile(r, diabetic, high_chol, slot)]
        if not candidates:
            candidates = recipes

        def slot_fit(r):
            c_err = abs(r["calories"] - t_cal) / max(t_cal, 1)
            p_err = abs(r["protein_g"] - t_prot) / max(t_prot, 1)
            cb_err = abs(r["carbs_g"] - t_carb) / max(t_carb, 1)
            f_err = abs(r["fat_g"] - t_fat) / max(t_fat, 1)
            rating = (r.get("rating") or 3.5) / 5.0

            affinity = 0.0
            name_and_cat = (r.get("name", "") + " " + r.get("category", "")).lower()
            if slot == "breakfast" and any(w in name_and_cat for w in ["egg", "quiche", "frittata", "omelet", "muffin", "pancake", "waffle", "oatmeal", "scone", "hash brown", "smoothie", "breakfast", "salad"]):
                affinity += 0.5
            elif slot == "dinner" and any(w in name_and_cat for w in ["chicken", "beef", "salmon", "fish", "pasta", "lasagna", "casserole", "roast", "pork", "steak", "dinner"]):
                affinity += 0.4

            return c_err * 2.0 + cb_err * 1.5 + f_err * 1.8 + p_err * 0.8 - rating * 0.3 - affinity

        by_slot[slot] = sorted(candidates, key=slot_fit)[:14]

    best_combo = None
    best_loss = float("inf")

    b_list = by_slot.get("breakfast", [])
    l_list = by_slot.get("lunch", [])
    d_list = by_slot.get("dinner", [])
    s_list = by_slot.get("snack", [])

    for b in b_list:
        for l in l_list:
            for d in d_list:
                for s in s_list:
                    if len({b["id"], l["id"], d["id"], s["id"]}) < 4:
                        continue
                    tot_cal = b["calories"] + l["calories"] + d["calories"] + s["calories"]
                    tot_prot = b["protein_g"] + l["protein_g"] + d["protein_g"] + s["protein_g"]
                    tot_carb = b["carbs_g"] + l["carbs_g"] + d["carbs_g"] + s["carbs_g"]
                    tot_fat = b["fat_g"] + l["fat_g"] + d["fat_g"] + s["fat_g"]

                    c_loss = abs(tot_cal - target_calories) / max(target_calories, 1)
                    p_loss = abs(tot_prot - target_protein) / max(target_protein, 1)
                    cb_loss = abs(tot_carb - target_carbs) / max(target_carbs, 1)
                    f_loss = abs(tot_fat - target_fat) / max(target_fat, 1)

                    loss = c_loss * 2.5 + cb_loss * 2.2 + f_loss * 2.2 + p_loss * 1.2
                    if loss < best_loss:
                        best_loss = loss
                        best_combo = [
                            {"slot": "breakfast", "recipe": b},
                            {"slot": "lunch", "recipe": l},
                            {"slot": "dinner", "recipe": d},
                            {"slot": "snack", "recipe": s},
                        ]

    if not best_combo:
        best_combo = [
            {"slot": "breakfast", "recipe": b_list[0] if b_list else recipes[0]},
            {"slot": "lunch", "recipe": l_list[0] if l_list else recipes[0]},
            {"slot": "dinner", "recipe": d_list[0] if d_list else recipes[0]},
            {"slot": "snack", "recipe": s_list[0] if s_list else recipes[0]},
        ]

    return best_combo


def build_day_plan(row, predicted_calories, predicted_protein_g, predicted_carbs_g, predicted_fat_g, recipes=None):
    recipes = recipes if recipes is not None else load_catalog()
    adj_calories, adj_carbs, messages = apply_constraints(predicted_calories, predicted_carbs_g, row)
    diabetic = row.get("DIABETIC_FLAG", 0) == 1 or row.get("HBA1C", 0) >= 6.5
    high_chol = row.get("HYPERTENSION_FLAG", 0) == 1 or row.get("CHOLESTEROL_MGDL", 0) >= 240
    category = row.get("BMI_CATEGORY") or bmi_category(row.get("BMI"))

    protein = predicted_protein_g
    fat = predicted_fat_g
    if adj_calories != predicted_calories and predicted_calories:
        scale = adj_calories / predicted_calories
        protein *= scale
        fat *= scale

    picks = select_balanced_day_plan(
        recipes=recipes,
        target_calories=adj_calories,
        target_protein=protein,
        target_carbs=adj_carbs,
        target_fat=fat,
        diabetic=diabetic,
        high_chol=high_chol,
    )

    meals = []
    for item in picks:
        slot = item["slot"]
        pick = item["recipe"]
        share = SLOT_SHARE.get(slot, 0.25)

        norm_r = dict(pick)
        norm_r["kcal"] = norm_r["calories"]
        norm_r["protein"] = norm_r["protein_g"]
        norm_r["carbohydrates"] = norm_r["carbs_g"]
        norm_r["carbs"] = norm_r["carbs_g"]
        norm_r["fat"] = norm_r["fat_g"]

        meals.append({
            "slot": slot,
            "slot_display": slot.capitalize(),
            "target_kcal": round(adj_calories * share),
            "target_protein_g": round(protein * share, 1),
            "target_carbs_g": round(adj_carbs * share, 1),
            "target_fat_g": round(fat * share, 1),
            "calories": norm_r["calories"],
            "kcal": norm_r["calories"],
            "protein": norm_r["protein_g"],
            "protein_g": norm_r["protein_g"],
            "carbohydrates": norm_r["carbs_g"],
            "carbs": norm_r["carbs_g"],
            "carbs_g": norm_r["carbs_g"],
            "fat": norm_r["fat_g"],
            "fat_g": norm_r["fat_g"],
            "recipe": norm_r,
        })

    total_cal = round(sum(m["calories"] for m in meals), 1)
    total_prot = round(sum(m["protein"] for m in meals), 1)
    total_carb = round(sum(m["carbohydrates"] for m in meals), 1)
    total_fat_val = round(sum(m["fat"] for m in meals), 1)
    total_sod = round(sum(m["recipe"]["sodium_mg"] for m in meals), 1)

    totals = {
        "calories": total_cal,
        "kcal": total_cal,
        "protein": total_prot,
        "protein_g": total_prot,
        "carbohydrates": total_carb,
        "carbs": total_carb,
        "carbs_g": total_carb,
        "fat": total_fat_val,
        "fat_g": total_fat_val,
        "sodium_mg": total_sod,
    }

    targets = {
        "calories": round(adj_calories),
        "kcal": round(adj_calories),
        "protein": round(protein),
        "protein_g": round(protein),
        "carbohydrates": round(adj_carbs),
        "carbs": round(adj_carbs),
        "carbs_g": round(adj_carbs),
        "fat": round(fat),
        "fat_g": round(fat),
    }

    return {
        "bmi_category": category,
        "targets": targets,
        "constraints": messages,
        "meals": meals,
        "totals": totals,
        "source": "Food.com",
    }


if __name__ == "__main__":
    demo = {
        "HBA1C": 7.0, "DIABETIC_FLAG": 1, "CHOLESTEROL_MGDL": 250,
        "HYPERTENSION_FLAG": 1, "BMI": 29.4, "BMI_CATEGORY": "Overweight",
    }
    plan = build_day_plan(demo, 2200, 80, 240, 70)
    print("Targets:", plan["targets"])
    for msg in plan["constraints"]:
        print(" -", msg)
    for meal in plan["meals"]:
        r = meal["recipe"]
        print(f"{meal['slot']:10} {r['name'][:40]:40} {r['calories']} kcal  carbs={r['carbs_g']}g")
    print("Day totals:", plan["totals"])
