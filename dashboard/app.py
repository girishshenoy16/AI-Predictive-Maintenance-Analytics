import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    layout="wide"
)

# --------------------------------------------------
# Global Startup UI Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            max-width: 1200px;
            margin: auto;
        }

        .glass-card {
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 28px;
            border: 1px solid rgba(255,255,255,0.08);
            text-align: center;
        }

        .kpi-title {
            font-size: 14px;
            color: #9CA3AF;
        }

        .kpi-value {
            font-size: 36px;
            font-weight: 700;
            color: white;
        }

        .section-title {
            font-size: 24px;
            font-weight: 600;
            text-align: center;
            margin-top: 40px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
SUMMARY_PATH = Path("reports/machine_risk_summary.csv")
FEATURE_PATH = Path("data/processed/fd001_features.csv")
MODEL_PATH = Path("models/RF_Classifier_failure_classifier.pkl")


# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_PATH)


@st.cache_data
def load_features():
    return pd.read_csv(FEATURE_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


summary = load_summary()
feature_df = load_features()
model = load_model()

summary["failure_probability"] = summary["failure_probability"].round(2)
summary["predicted_RUL"] = summary["predicted_RUL"].round(2)

# --------------------------------------------------
# Prepare Features for SHAP
# --------------------------------------------------
DROP_COLS = [
    "engine_id",
    "cycle",
    "source",
    "failure_within_30",
    "RUL"
]

X_full = feature_df.drop(columns=DROP_COLS)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


explainer = get_explainer(model)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2 = st.tabs(
    [
        "📊 Prediction Dashboard",
        "🔍 Explainability"
    ]
)

# ==================================================
# TAB 1 — MAIN DASHBOARD
# ==================================================
with tab1:
    st.markdown(
        "<h1 style='text-align:center;'>"
        "🚀 AI Predictive Maintenance"
        "</h1>", unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align:center; color:#9CA3AF;'>"
        "Real-Time Fleet Intelligence Platform"
        "</h3>", unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Fleet Overview'
        '</div>', unsafe_allow_html=True
    )

    kpi_cols = st.columns(4)


    def kpi_card(title, value):
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True
        )


    with kpi_cols[0]:
        kpi_card("Total Machines", summary.shape[0])

    with kpi_cols[1]:
        kpi_card("High Risk", (summary["risk_level"] == "HIGH").sum())

    with kpi_cols[2]:
        kpi_card("Medium Risk", (summary["risk_level"] == "MEDIUM").sum())

    with kpi_cols[3]:
        kpi_card("Low Risk", (summary["risk_level"] == "LOW").sum())

    st.divider()

    # --------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Risk Distribution'
        '</div>', unsafe_allow_html=True
    )

    risk_order = [
        "HIGH",
        "MEDIUM",
        "LOW"
    ]

    risk_counts = summary["risk_level"].value_counts().reindex(risk_order).fillna(0)

    colors = {
        "HIGH": "#FF4D4D",
        "MEDIUM": "#FFA726",
        "LOW": "#00E676"
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=risk_order,
        y=risk_counts.values,
        marker_color=[colors[r] for r in risk_order],
        text=risk_counts.values,
        textposition="outside"
    ))

    fig.update_layout(
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(tickangle=0),
        showlegend=False,
        margin=dict(t=80, b=40, l=20, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    # --------------------------------------------------
    # MACHINE DRILL DOWN (REFINED + CENTERED)
    # --------------------------------------------------
    st.markdown(
        '<div class="section-title" style="text-alignment:center;">'
        'Machine Drill-Down'
        '</div>', unsafe_allow_html=True
    )

    selected_engine = st.selectbox(
        "Select Engine ID",
        sorted(summary["engine_id"].unique()),
        key="engine_select"
    )

    engine = summary[summary["engine_id"] == selected_engine].iloc[0]

    st.write("Currently Selected Engine ID:", selected_engine)

    risk_colors = {
        "HIGH": "#FF4D4D",
        "MEDIUM": "#FFA726",
        "LOW": "#00E676"
    }

    risk_color = risk_colors.get(engine["risk_level"], "#00E676")

    colA, colB, colC = st.columns(3)

    with colA:

        st.markdown(
            f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="kpi-title">Failure Probability</div>
                    <div class="kpi-value">{engine['failure_probability'] * 100:.0f}%</div>
                    <br>
                </div>
                """, unsafe_allow_html=True
        )

    with colB:

        st.markdown(
            f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="kpi-title">Predicted RUL (cycles)</div>
                    <div class="kpi-value">{engine['predicted_RUL']:.2f}</div>
                    <br>
                </div>
                """, unsafe_allow_html=True
        )

    with colC:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center;">
                <div class="kpi-title">Risk Level</div>
                <div style="font-size:34px;
                            font-weight:700;
                            color:{risk_color};">
                    {engine['risk_level']}
                </div>
                <br>
            </div>
            <br>
            """, unsafe_allow_html=True
        )

    max_rul = summary["predicted_RUL"].max()
    rul_ratio = engine["predicted_RUL"] / max_rul

    st.markdown(
        '<div style="text-align:center; font-size:14px; color:#9CA3AF;">'
        'Remaining Useful Life'
        '</div>', unsafe_allow_html=True
    )

    st.progress(rul_ratio)

    st.divider()

    if engine["risk_level"] == "HIGH":
        recommendation = "Immediate inspection required. Schedule maintenance in next cycle."
    elif engine["risk_level"] == "MEDIUM":
        recommendation = "Monitor closely and prepare preventive maintenance."
    else:
        recommendation = "Machine operating within healthy range. Continue routine monitoring."

    st.markdown(
        f"""
        <div style="
            text-align:center; 
            padding:18px; 
            -radius:16px;
            background: linear-gradient(90deg, {risk_color}33, {risk_color}11);
            border:1px solid {risk_color}; 
            font-size:16px;
            font-weight:500;
        ">
            <b>AI Recommendation:</b><br><br>
            {recommendation}
        </div>
        <br>
        """, unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------------------
    # CENTERED DOWNLOAD BUTTON
    # --------------------------------------------------

    center_col = st.columns([1, 2, 1])[1]

    with center_col:
        st.download_button(
            "⬇ Download Machine Risk Summary",
            summary.to_csv(index=False),
            file_name="machine_risk_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==================================================
# TAB 2 — EXPLAINABILITY
# ==================================================
with tab2:
    st.markdown('<div class="section-title">Model Explainability (SHAP)</div>', unsafe_allow_html=True)

    # Sample for performance
    SAMPLE_SIZE = min(1500, len(X_full))
    X_sample = X_full.sample(n=SAMPLE_SIZE, random_state=42)

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

    st.subheader("Global Feature Importance")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Per-Engine Explanation")

    shap_engine_id = st.selectbox(
        "Select Engine for SHAP Explanation",
        sorted(feature_df["engine_id"].unique()),
        key="shap_engine"
    )

    engine_row = feature_df[
        feature_df["engine_id"] == shap_engine_id
        ].sort_values("cycle").iloc[-1]

    engine_features = engine_row.drop(DROP_COLS).to_frame().T

    shap_vals = explainer.shap_values(engine_features)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    single_shap = shap_vals[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    elif isinstance(expected_value, np.ndarray):
        expected_value = expected_value.flatten()[1] if len(expected_value) > 1 else expected_value.item()

    explanation = shap.Explanation(
        values=single_shap,
        base_values=float(expected_value),
        data=engine_features.iloc[0].values,
        feature_names=engine_features.columns
    )

    # Top 5 drivers
    st.markdown("### 🔎 Top 5 Risk Drivers")

    abs_vals = np.abs(single_shap)
    top_idx = np.argsort(abs_vals)[-5:][::-1]

    for idx in top_idx:
        feature_name = engine_features.columns[idx]
        shap_val = single_shap[idx]
        direction = "↑ increases risk" if shap_val > 0 else "↓ reduces risk"
        st.markdown(f"- **{feature_name}** ({shap_val:.4f}) → {direction}")

    fig2, ax2 = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig2)
    plt.close(fig2)