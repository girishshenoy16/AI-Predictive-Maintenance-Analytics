import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_PATH = Path("data/processed/fd001_features.csv")
MODEL_PATH = Path("models")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(exist_ok=True)

DECISION_OFFSET = 30  # cycles before end-of-life

HIGH_PERCENTILE = 0.20
MEDIUM_PERCENTILE = 0.50

# --------------------------------------------------
# Load data & models
# --------------------------------------------------
print("Loading feature dataset...")
df = pd.read_csv(DATA_PATH)

print("Loading trained models...")
clf = joblib.load(MODEL_PATH / "RF_Classifier_failure_classifier.pkl")
reg = joblib.load(MODEL_PATH / "RF_Regressor_rul_regressor.pkl")

# --------------------------------------------------
# Prepare features
# --------------------------------------------------
TARGET_CLASS = "failure_within_30"
TARGET_REG = "RUL"

DROP_COLS = [
    "engine_id",
    "cycle",
    "source",
    TARGET_CLASS,
    TARGET_REG
]

X = df.drop(columns=DROP_COLS)

# --------------------------------------------------
# Generate predictions
# --------------------------------------------------
print("Generating predictions...")

df["failure_probability"] = clf.predict_proba(X)[:, 1]
df["predicted_RUL"] = reg.predict(X)

# Prevent negative RUL (safety)
df["predicted_RUL"] = df["predicted_RUL"].clip(lower=0)

# --------------------------------------------------
# Simulate real-time operational snapshot
# --------------------------------------------------
print(f"Applying decision offset of {DECISION_OFFSET} cycles...")

df_sorted = df.sort_values(["engine_id", "cycle"])

def safe_snapshot(group):
    if len(group) >= DECISION_OFFSET:
        return group.iloc[-DECISION_OFFSET]
    else:
        return group.iloc[-1]  # fallback if too short

snapshot_df = (
    df_sorted.groupby("engine_id", group_keys=False)
    .apply(safe_snapshot)
    .reset_index(drop=True)
)

print("Snapshot engines count:", snapshot_df["engine_id"].nunique())

# --------------------------------------------------
# Rank-based risk segmentation
# --------------------------------------------------
print("Applying rank-based risk segmentation...")

snapshot_df = snapshot_df.sort_values("predicted_RUL").reset_index(drop=True)

n = len(snapshot_df)

high_cutoff = max(int(HIGH_PERCENTILE * n), 1)
medium_cutoff = max(int(MEDIUM_PERCENTILE * n), high_cutoff + 1)

snapshot_df["risk_level"] = "LOW"

snapshot_df.loc[:high_cutoff - 1, "risk_level"] = "HIGH"
snapshot_df.loc[high_cutoff:medium_cutoff - 1, "risk_level"] = "MEDIUM"

# --------------------------------------------------
# Round values for clean UI (after segmentation)
# --------------------------------------------------
snapshot_df["failure_probability"] = snapshot_df["failure_probability"].round(2)
snapshot_df["predicted_RUL"] = snapshot_df["predicted_RUL"].round(2)

df["failure_probability"] = df["failure_probability"].round(2)
df["predicted_RUL"] = df["predicted_RUL"].round(2)

# --------------------------------------------------
# Save full prediction dataset
# --------------------------------------------------
df.to_csv(
    REPORT_PATH / "fd001_predictions.csv",
    index=False
)

# --------------------------------------------------
# Save machine-level summary for dashboard
# --------------------------------------------------
machine_risk = snapshot_df[
    ["engine_id", "failure_probability", "predicted_RUL", "risk_level"]
].sort_values("engine_id")

machine_risk.to_csv(
    REPORT_PATH / "machine_risk_summary.csv",
    index=False
)

# --------------------------------------------------
# Console summary
# --------------------------------------------------
print("✅ Predictions completed successfully.")
print("📁 Saved:")
print("   - reports/fd001_predictions.csv")
print("   - reports/machine_risk_summary.csv")

print("\n📊 Risk Distribution:")
print(machine_risk["risk_level"].value_counts())

original_count = df["engine_id"].nunique()
snapshot_count = snapshot_df["engine_id"].nunique()

if original_count != snapshot_count:
    print("WARNING: Engine count mismatch detected!")
