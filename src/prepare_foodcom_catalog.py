"""
Build a compact Food.com recipe catalog for meal planning.

Reads data/foodcom_raw_recipes.csv (Food.com recipes, Kaggle/HuggingFace
mirror of the public Food.com dump) and writes:
  - data/foodcom_catalog.json
  - website/foodcom_catalog.json
"""
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = ROOT / "data" / "foodcom_raw_recipes.csv"
OUT_DATA = ROOT / "data" / "foodcom_catalog.json"
OUT_WEB = ROOT / "website" / "foodcom_catalog.json"

QUOTED = re.compile(r'"([^"]*)"')

BREAKFAST_HINTS = (
    "breakfast", "brunch", "pancake", "waffle", "oatmeal", "omelet", "omelette",
    "eggs", "muffin", "scone", "granola", "cereal", "french toast", "smoothie",
    "yogurt",
)
LUNCH_HINTS = (
    "lunch", "salad", "sandwich", "soup", "chowder", "wrap", "chili", "stew",
    "bowl",
)
DINNER_HINTS = (
    "dinner", "chicken", "beef", "pork", "fish", "salmon", "turkey", "meat",
    "tofu", "pasta", "casserole", "curry", "roast", "stir-fry", "stir fry",
    "main dish", "poultry", "shrimp", "lamb",
)
SNACK_HINTS = (
    "snack", "appetizer", "dip", "fruit", "beverage", "shake", "smoothie",
)


def parse_r_char(value):
    if not value or value == "NA":
        return []
    value = value.strip()
    found = QUOTED.findall(value)
    if found:
        return found
    if value.startswith("http"):
        return [value.strip('"')]
    return []


def to_float(value, default=None):
    try:
        if value in (None, "", "NA"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def first_image(raw):
    urls = parse_r_char(raw)
    for url in urls:
        if url.startswith("http"):
            return url
    if raw and raw.startswith("http"):
        return raw.strip().strip('"')
    return ""


def blob(row):
    parts = [
        row.get("Name") or "",
        row.get("RecipeCategory") or "",
        row.get("Keywords") or "",
        row.get("Description") or "",
    ]
    return " ".join(parts).lower()


def classify_slot(row):
    text = blob(row)
    cat = (row.get("RecipeCategory") or "").lower()
    calories = to_float(row.get("Calories"), 0) or 0

    if any(h in text for h in BREAKFAST_HINTS) or cat in ("breakfast", "scones"):
        return "breakfast"
    if any(h in text for h in LUNCH_HINTS) or cat in ("lunch/snacks", "chowders"):
        return "lunch"
    if any(h in text for h in DINNER_HINTS):
        return "dinner"
    if any(h in text for h in SNACK_HINTS) or (calories and calories < 220 and cat not in ("dessert", "pie", "candy")):
        return "snack"
    if cat in ("dessert", "pie", "bar cookie", "drop cookies", "candy", "cheesecake", "frozen desserts"):
        return "snack"
    if cat in ("vegetable", "beans", "potato"):
        return "lunch"
    return None


def quality_score(row):
    rating = to_float(row.get("AggregatedRating"), 0) or 0
    reviews = to_float(row.get("ReviewCount"), 0) or 0
    return rating * math.log1p(reviews)


def compact(row, slot):
    ingredients = parse_r_char(row.get("RecipeIngredientParts") or "")[:12]
    steps = parse_r_char(row.get("RecipeInstructions") or "")[:8]
    recipe_id = str(int(float(row["RecipeId"]))) if row.get("RecipeId") else ""
    name = (row.get("Name") or "").strip()
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return {
        "id": recipe_id,
        "name": name,
        "slot": slot,
        "category": (row.get("RecipeCategory") or "").strip(),
        "minutes": None,
        "calories": round(to_float(row.get("Calories"), 0) or 0, 1),
        "protein_g": round(to_float(row.get("ProteinContent"), 0) or 0, 1),
        "carbs_g": round(to_float(row.get("CarbohydrateContent"), 0) or 0, 1),
        "fat_g": round(to_float(row.get("FatContent"), 0) or 0, 1),
        "sat_fat_g": round(to_float(row.get("SaturatedFatContent"), 0) or 0, 1),
        "sodium_mg": round(to_float(row.get("SodiumContent"), 0) or 0, 1),
        "sugar_g": round(to_float(row.get("SugarContent"), 0) or 0, 1),
        "fiber_g": round(to_float(row.get("FiberContent"), 0) or 0, 1),
        "rating": to_float(row.get("AggregatedRating")),
        "reviews": int(to_float(row.get("ReviewCount"), 0) or 0),
        "ingredients": ingredients,
        "steps": steps,
        "image": first_image(row.get("Images") or ""),
        "url": f"https://www.food.com/recipe/{slug}-{recipe_id}" if recipe_id else "",
        "source": "Food.com",
    }


def load_valid():
    if not SRC_CSV.exists():
        raise SystemExit(f"Missing {SRC_CSV}. Download Food.com recipes.csv first.")
    by_slot = {"breakfast": [], "lunch": [], "dinner": [], "snack": []}
    seen = set()
    with SRC_CSV.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("Name") or "").strip()
            calories = to_float(row.get("Calories"))
            protein = to_float(row.get("ProteinContent"))
            carbs = to_float(row.get("CarbohydrateContent"))
            fat = to_float(row.get("FatContent"))
            if not name or calories is None or protein is None or carbs is None or fat is None:
                continue
            if calories < 80 or calories > 850:
                continue
            if protein < 0 or carbs < 0 or fat < 0:
                continue
            slot = classify_slot(row)
            if not slot:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            item = compact(row, slot)
            item["_score"] = quality_score(row)
            by_slot[slot].append(item)
    return by_slot


def select_catalog(by_slot, caps=None):
    caps = caps or {"breakfast": 40, "lunch": 40, "dinner": 55, "snack": 30}
    catalog = []
    for slot, cap in caps.items():
        ranked = sorted(by_slot[slot], key=lambda r: r["_score"], reverse=True)
        for item in ranked[:cap]:
            item.pop("_score", None)
            catalog.append(item)
    return catalog


if __name__ == "__main__":
    by_slot = load_valid()
    print({k: len(v) for k, v in by_slot.items()})
    catalog = select_catalog(by_slot)
    payload = {
        "source": "Food.com Recipes (public Kaggle dump: shuyangli94 / irkaal food.com recipes)",
        "count": len(catalog),
        "recipes": catalog,
    }
    text = json.dumps(payload, indent=2)
    OUT_DATA.write_text(text, encoding="utf-8")
    OUT_WEB.write_text(text, encoding="utf-8")
    print(f"Wrote {len(catalog)} recipes -> {OUT_DATA.name} and website/{OUT_WEB.name}")
