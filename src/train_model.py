import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

DATA_PATH = Path("data/processed/fd001_features.csv")
MODEL_PATH = Path("models")
REPORT_PATH = Path("reports")

MODEL_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

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

engine_ids = df["engine_id"].unique()

train_ids, test_ids = train_test_split(
    engine_ids, test_size=0.2, random_state=42
)

train_mask = df["engine_id"].isin(train_ids)
test_mask = df["engine_id"].isin(test_ids)

X_train, X_test = X[train_mask], X[test_mask]
y_class_train, y_class_test = y_class[train_mask], y_class[test_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]


# --------------------------------------------------
# CLASSIFICATION MODEL
# --------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_class_train)

y_pred_prob = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

roc_auc = roc_auc_score(y_class_test, y_pred_prob)

print("ROC-AUC:", round(roc_auc, 2))
print("\nClassification Report:")
print(classification_report(y_class_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_class_test, y_pred))


# Save classifier feature importance (rounded)
feature_importance_clf = pd.DataFrame({
    "feature": X.columns,
    "importance": clf.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance_clf["importance"] = feature_importance_clf["importance"].round(4)

feature_importance_clf.to_csv(
    REPORT_PATH / "RF_Classifier_feature_importance.csv",
    index=False
)

joblib.dump(clf, MODEL_PATH / "RF_Classifier_failure_classifier.pkl")


# --------------------------------------------------
# REGRESSION MODEL
# --------------------------------------------------
reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

reg.fit(X_train, y_reg_train)

y_reg_pred = reg.predict(X_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))

print("RUL MAE:", round(mae, 2))
print("RUL RMSE:", round(rmse, 2))


# Save regression feature importance (rounded)
feature_importance_reg = pd.DataFrame({
    "feature": X.columns,
    "importance": reg.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance_reg["importance"] = feature_importance_reg["importance"].round(4)

feature_importance_reg.to_csv(
    REPORT_PATH / "RF_Regressor_feature_importance.csv",
    index=False
)

joblib.dump(reg, MODEL_PATH / "RF_Regressor_rul_regressor.pkl")


# --------------------------------------------------
# SAVE EVALUATION SUMMARY
# --------------------------------------------------
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

print("\nModels saved successfully.")
print("Evaluation summary saved.")