# 🚀 AI Predictive Maintenance Platform

### Industry-Grade End-to-End AI System for Manufacturing

---

<p align="center">
  <b>Failure Prediction • Remaining Useful Life Estimation • Risk Segmentation • Explainable AI</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange">
  <img src="https://img.shields.io/badge/Explainability-SHAP-red">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-green">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success">
</p>

---

# 📌 Overview

This project delivers a **complete, industry-oriented AI Predictive Maintenance platform** designed for manufacturing environments.

It simulates real-world deployment of:

* 🔴 Failure Probability Prediction
* 🟠 Remaining Useful Life (RUL) Estimation
* 🟢 Risk-Based Machine Segmentation
* 🔍 SHAP-Based Explainable AI
* 📊 Executive Dashboard for Decision-Making

This is not an academic notebook — it is a full ML pipeline with production-style structure.

---

# 🏭 Business Problem

Manufacturing operations face:

* Unexpected machine breakdowns
* Expensive emergency repairs
* Downtime losses
* Inefficient maintenance planning

This system helps maintenance teams:

✔ Identify high-risk machines early
✔ Prioritize inspections intelligently
✔ Reduce unplanned downtime
✔ Optimize maintenance scheduling

---

# 🧠 System Architecture

```
Raw Sensor Data
      ↓
Data Cleaning & Label Engineering
      ↓
Feature Engineering (Rolling + Degradation Trends)
      ↓
Random Forest Models
      ↓
Failure Probability + RUL Prediction
      ↓
Risk Segmentation (HIGH / MEDIUM / LOW)
      ↓
SHAP Explainability
      ↓
Interactive Executive Dashboard
```

---

# 📊 Model Performance

## 🔹 Failure Prediction (Binary Classification)

| Metric              | Value    |
| ------------------- | -------- |
| ROC-AUC             | **0.99** |
| Accuracy            | **97%**  |
| Precision (Failure) | **0.89** |
| Recall (Failure)    | **0.84** |
| F1-Score            | **0.86** |

## 🔹 Remaining Useful Life (Regression)

| Metric | Value            |
| ------ | ---------------- |
| MAE    | **23.77 cycles** |
| RMSE   | **33.77 cycles** |

---

# 📈 Risk Segmentation Strategy

Machines are ranked by predicted RUL:

* 🔴 Top 20% → HIGH Risk
* 🟠 Next 30% → MEDIUM Risk
* 🟢 Remaining 50% → LOW Risk

This ensures realistic and proportional maintenance prioritization.

---

# 🔍 Explainable AI (SHAP)

The system integrates SHAP for:

### 🌍 Global Explainability

* Sensor importance ranking
* Feature contribution visualization

### 🔬 Per-Engine Explainability

* Top 5 risk drivers
* Waterfall contribution plots
* Directional impact interpretation

This ensures:

* Transparency
* Trust
* Engineering validation
* Responsible AI implementation

---

# 📊 Dashboard Capabilities

The Streamlit dashboard includes:

* Fleet Overview KPIs
* Risk Distribution Visualization
* Machine Drill-Down Analysis
* Failure Probability (%)
* Predicted RUL (cycles)
* AI Maintenance Recommendation
* Global SHAP Importance
* Per-Engine SHAP Analysis

---

# 📁 Project Structure

```
AI-Predictive-Maintenance/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── Executive_Performance_Report.md
│   ├── shap_summary_plot.png
│   ├── shap_bar_plot.png
│   ├── fd001_predictions.csv
│   ├── machine_risk_summary.csv
│   ├── model_evaluation_summary.csv
│   ├── rul_regression_metrics.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── RF_Regressor_feature_importance.csv
│   └── RF_Classifier_feature_importance.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── explain_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

Clone repository:

```
git clone https://github.com/girishshenoy16/AI-Predictive-Maintenance-Platform.git
cd AI-Predictive-Maintenance
```

Create virtual environment:

```
python -m venv venv
python.exe -m pip install --upgrade pip
.\venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run Full Pipeline

### 1️⃣ Data Preprocessing

```
python src/data_preprocessing.py
```

### 2️⃣ Feature Engineering

```
python src/feature_engineering.py
```

### 3️⃣ Train Model

```
python src/train_model.py
```

### 4️⃣ Evaluate Model

```
python src/evaluate_model.py
```

### 5️⃣ Generate Predictions

```
python src/predict.py
```

### 6️⃣ Generate SHAP Reports

```
python src/explain_model.py
```

### 7️⃣ Launch Dashboard

```
streamlit run .\dashboard\app.py
```

---

# 🧪 Dataset

NASA Turbofan Engine Degradation Dataset (FD001)

* 100 engines
* Multiple sensor readings
* Time-based degradation simulation

Used to replicate real-world predictive maintenance environment.

---

# 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* SHAP
* Plotly
* Streamlit
* Joblib

---

# 🎯 Skills Demonstrated

* End-to-End ML System Design
* Time-Series Feature Engineering
* Classification & Regression Modeling
* Risk-Based Segmentation
* Model Evaluation & Validation
* Explainable AI Implementation
* Production-Style Dashboard Development
* Business Impact Framing

---

# 🔮 Future Enhancements

* Model Drift Monitoring
* Real-Time API Deployment (FastAPI)
* Streaming Sensor Integration
* Cost-Based Optimization Layer
* Ensemble Model Comparison

---

# ⭐ Why This Project Stands Out

This project demonstrates:

* Production-style ML pipeline
* Business-aligned AI solution
* Explainable AI integration
* Deployable dashboard
* Realistic industry application

It reflects how predictive maintenance systems are built in real manufacturing organizations.

---

# 👨‍💻 Author
**Girish Shenoy**
Computer Science Student | Aspiring AI & Business Analyst