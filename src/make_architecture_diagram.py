import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)
ax.axis("off")

MAROON = "#B71E42"
TEAL = "#028090"
NAVY = "#1B2A4A"
BEIGE = "#F4F1EC"

def box(x, y, w, h, text, fc=BEIGE, ec=MAROON, tc="black", fs=10.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, wrap=True, zorder=3, fontweight="bold")
    return (x, y, w, h)

def arrow(b1, b2, text=None):
    x1 = b1[0] + b1[2] / 2
    y1 = b1[1]
    x2 = b2[0] + b2[2] / 2
    y2 = b2[1] + b2[3]
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.4, color=NAVY, zorder=1)
    ax.add_patch(a)

# Row 1: Data sources
d1 = box(0.3, 6.7, 2.4, 0.9, "NHANES\n(Health/Biomarkers)", fc="#FFFFFF", ec=TEAL)
d2 = box(3.05, 6.7, 2.4, 0.9, "USDA FoodData\n(Nutrients)", fc="#FFFFFF", ec=TEAL)
d3 = box(5.8, 6.7, 2.4, 0.9, "Food.com\n(Recipes/Ratings)", fc="#FFFFFF", ec=TEAL)

# Row 2: Preprocessing
pre = box(2.0, 5.3, 5.0, 0.9, "Data Cleaning, Merge (SEQN) &\nIngredient-to-USDA Linking")
arrow(d1, pre); arrow(d2, pre); arrow(d3, pre)

# Row 3: Feature engineering
feat = box(2.0, 4.0, 5.0, 0.8, "Feature Engineering\n(BMI, TDEE, Biomarker Risk Flags)")
arrow(pre, feat)

# Row 4: Ensemble
ens = box(0.7, 2.7, 3.6, 0.9, "Stacked Ensemble\n(LightGBM + XGBoost + RF)", fc="#FDEDF0")
arrow(feat, ens)

# Row 4b: Constraint layer (parallel)
con = box(5.3, 2.7, 3.6, 0.9, "Biomarker Constraint Layer\n(Hard Safety Filters)", fc="#FDEDF0")
arrow(feat, con)

# Row 5: Recommendation engine
rec = box(1.5, 1.4, 5.0, 0.9, "Hybrid Recommendation Engine\n(RAG Retrieval + Content + Collaborative)")
arrow(ens, rec); arrow(con, rec)

# Row 6: Output
out1 = box(0.3, 0.1, 3.0, 0.8, "Explainability\n(SHAP)", fc="#FFFFFF", ec=NAVY)
out2 = box(3.6, 0.1, 3.0, 0.8, "Streamlit Demo /\nEvaluation Report", fc="#FFFFFF", ec=NAVY)
arrow(rec, out1); arrow(rec, out2)

plt.title("System Architecture — SafeNutri-RAG", fontsize=15, fontweight="bold", color=MAROON, pad=15)
plt.tight_layout()
plt.savefig("../results/system_architecture.png", dpi=160)
print("saved")
