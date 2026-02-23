# 🏭 AI-Driven Predictive Maintenance & Failure Risk Analytics for Manufacturing Equipment  

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

This project delivers a complete, industry-oriented AI system for predictive maintenance and failure risk analytics in manufacturing environments.

It simulates real-world industrial deployment of:

- 🔴 Failure Probability Prediction (within 30 cycles)
- 🟠 Remaining Useful Life (RUL) Estimation
- 🟢 Risk-Based Machine Segmentation (High / Medium / Low)
- 🔍 SHAP-Based Explainable AI
- 📊 Executive Dashboard for Decision-Making

This is not an academic notebook — it is a structured, production-style ML pipeline aligned with real manufacturing use cases.

---

# 🏭 Business Problem

Manufacturing organizations face:

- Unexpected machine breakdowns  
- High emergency repair costs  
- Downtime losses  
- Inefficient maintenance scheduling  

Traditional reactive maintenance increases cost and risk.

This AI-driven system enables:

✔ Proactive failure detection  
✔ Intelligent maintenance prioritization  
✔ Reduced downtime  
✔ Better asset lifecycle planning  
✔ Data-driven operational decisions  

---

# 🧠 System Architecture

```
Raw Sensor Data
      ↓
Data Cleaning & Label Engineering
      ↓
Feature Engineering (Rolling + Degradation Trends)
      ↓
Random Forest Models (Classifier + Regressor)
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

The model demonstrates strong separability and reliable early-failure detection.

---

## 🔹 Remaining Useful Life (Regression)

| Metric | Value            |
| ------ | ---------------- |
| MAE    | **23.77 cycles** |
| RMSE   | **33.77 cycles** |

The RUL estimation provides operationally meaningful lifecycle predictions for maintenance planning.

---

# 📈 Risk Segmentation Strategy

Machines are ranked by predicted RUL:

* 🔴 Top 20% → HIGH Risk
* 🟠 Next 30% → MEDIUM Risk
* 🟢 Remaining 50% → LOW Risk

This rank-based segmentation avoids unstable probability thresholds and ensures proportional resource allocation.

This also ensures realistic and proportional maintenance prioritization.

---

# 🔍 Explainable AI (SHAP Integration)

The system integrates SHAP fo model transparency:

### 🌍 Global Explainability

* Sensor importance ranking
* Sensor contribution visualization
* Feature contribution visualization 
* Feature importance ranking
* System-wide degradation drivers

### 🔬 Per-Engine Explainability

* Top 5 failed drivers
* Waterfall contribution plots
* Directional impact interpretation (↑ increases risk / ↓ reduces risk)

Explainability ensures:

* Transparency
* Engineering validation
* Stakeholder Trust
* Responsible AI implementation

---

# 📊 Dashboard Capabilities

The Streamlit dashboard provides:

* Fleet Overview KPIs
* Risk Distribution Visualization
* Machine Drill-Down Analysis
* Failure Probability (%)
* Predicted RUL (cycles)
* AI-based Maintenance Recommendation
* Global SHAP Importance
* Per-Engine SHAP Analysis

The dashboard simulates executive-level operational monitoring.

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
git clone https://github.com/girishshenoy16/AI-Predictive-Maintenance-Analytics.git
cd AI-Predictive-Maintenance-Analytics
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
* Time-based degradation simulations or patterns

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
* Explainable AI Implementation
* Model Evaluation & Validation
* Production-Style Dashboard Development
* Business Impact Framing

---

# 🔮 Future Enhancements

* Model Drift Monitoring
* Real-Time API Deployment (FastAPI)
* Streaming Sensor Integration
* Cost-Based Maintenance Optimization 
* Ensemble Model Comparison

---

# ⭐ Why This Project Stands Out

This project reflects how predictive maintenance solutions are built in real manufacturing environments:

* Production-style ML pipeline
* Business-aligned AI design & solution
* Explainable AI integration
* Transferrable and explainable modeling
* Scalable system structure
* Deployment-ready dashboard
* Realistic industry application

It demonstrates practical AI system thinking beyond academic experimentation.

---