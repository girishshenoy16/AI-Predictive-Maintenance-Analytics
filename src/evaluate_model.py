import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/fd001_features.csv")
MODEL_PATH = Path("models")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(exist_ok=True)

# -----------------------------
# Load Data & Models
# -----------------------------
df = pd.read_csv(DATA_PATH)

clf = joblib.load(MODEL_PATH / "RF_Classifier_failure_classifier.pkl")
reg = joblib.load(MODEL_PATH / "RF_Regressor_rul_regressor.pkl")

# -----------------------------
# Feature / Target Split
# -----------------------------
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
y_class = df[TARGET_CLASS]
y_reg = df[TARGET_REG]

# -----------------------------
# Engine-level Train/Test Split
# -----------------------------
engine_ids = df["engine_id"].unique()
np.random.seed(42)

test_engines = np.random.choice(
    engine_ids,
    size=int(0.2 * len(engine_ids)),
    replace=False
)

test_mask = df["engine_id"].isin(test_engines)

X_test = X[test_mask]
y_class_test = y_class[test_mask]
y_reg_test = y_reg[test_mask]

# -----------------------------
# Classification Evaluation
# -----------------------------
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

roc_auc = roc_auc_score(y_class_test, y_prob)

conf_matrix = confusion_matrix(y_class_test, y_pred)

class_report = classification_report(
    y_class_test,
    y_pred,
    output_dict=True
)

# Round classification report values
class_report_df = pd.DataFrame(class_report).transpose().round(2)

class_report_df.to_csv(
    REPORT_PATH / "classification_report.csv"
)

# Save confusion matrix (no rounding needed)
pd.DataFrame(
    conf_matrix,
    index=["Actual_No_Failure", "Actual_Failure"],
    columns=["Pred_No_Failure", "Pred_Failure"]
).to_csv(REPORT_PATH / "confusion_matrix.csv")

# -----------------------------
# Regression Evaluation (RUL)
# -----------------------------
y_reg_pred = reg.predict(X_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))  # TRUE RMSE

rul_metrics = pd.DataFrame({
    "metric": ["MAE", "RMSE"],
    "value": [
        round(mae, 2),
        round(rmse, 2)
    ]
})

rul_metrics.to_csv(
    REPORT_PATH / "rul_regression_metrics.csv",
    index=False
)

# -----------------------------
# Combined Evaluation Summary
# -----------------------------
evaluation_summary = pd.DataFrame({
    "metric": ["ROC_AUC", "MAE", "RMSE"],
    "value": [
        round(roc_auc, 2),
        round(mae, 2),
        round(rmse, 2)
    ]
})

evaluation_summary.to_csv(
    REPORT_PATH / "model_evaluation_summary.csv",
    index=False
)

# -----------------------------
# Console Summary
# -----------------------------
print("\n=== MODEL EVALUATION SUMMARY ===")
print(f"ROC-AUC (Failure Prediction): {roc_auc:.2f}")
print(f"RUL MAE (cycles): {mae:.2f}")
print(f"RUL RMSE (cycles): {rmse:.2f}")

print("\nReports saved to /reports folder")