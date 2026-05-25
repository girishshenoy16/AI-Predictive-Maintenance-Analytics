import pandas as pd
import shap
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = Path("data/processed/fd001_features.csv")
MODEL_PATH = Path("models/RF_Classifier_failure_classifier.pkl")
OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

# --------------------------------------------------
# Load Data & Model
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

TARGET_CLASS = "failure_within_30"

DROP_COLS = [
    "engine_id",
    "cycle",
    "source",
    TARGET_CLASS,
    "RUL"
]

X = df.drop(columns=DROP_COLS)

model = joblib.load(MODEL_PATH)

print("Model expects:", model.n_features_in_)
print("X shape:", X.shape[1])

# --------------------------------------------------
# SHAP Explainer
# --------------------------------------------------
SAMPLE_SIZE = min(2000, len(X))
X_sample = X.sample(n=SAMPLE_SIZE, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_to_plot = shap_values[:, :, 1]
else:
    shap_values_to_plot = shap_values

print("SHAP shape:", shap_values_to_plot.shape)
print("X Sample shape:", X_sample.shape)

plt.figure()
shap.summary_plot(
    shap_values_to_plot,
    X_sample,
    show=False
)


# --------------------------------------------------
# Global Feature Importance (Beeswarm)
# --------------------------------------------------
plt.figure()
shap.summary_plot(
    shap_values_to_plot,
    X_sample,
    show=False
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "shap_summary_plot.png", dpi=300)
plt.close()

# --------------------------------------------------
# Bar Importance Plot
# --------------------------------------------------
plt.figure()
shap.summary_plot(
    shap_values_to_plot,   # ✅ Use normalized values
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "shap_bar_plot.png", dpi=300)
plt.close()

print("SHAP explainability visuals saved successfully.")
print(f"Saved to: {OUTPUT_PATH}")
