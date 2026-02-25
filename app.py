import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="💳 AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: white;
}
.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
# 💳 AI-Powered Fraud Detection System
### Real-Time Risk Scoring & Transaction Monitoring
""")

st.markdown("---")

# ---------------- LOAD MODEL ----------------
model = joblib.load("fraud_model.pkl")

with open("threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ System Info")
st.sidebar.success("Model: Cost-Sensitive XGBoost")
st.sidebar.info(f"Optimized Threshold: {threshold:.2f}")
st.sidebar.markdown("Built for ML Engineer Portfolio 🚀")

# ---------------- MAIN INPUT SECTION ----------------
st.subheader("📥 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

input_data = {}

for i, col in enumerate(feature_columns):
    if i % 3 == 0:
        with col1:
            input_data[col] = st.number_input(col, value=0.0)
    elif i % 3 == 1:
        with col2:
            input_data[col] = st.number_input(col, value=0.0)
    else:
        with col3:
            input_data[col] = st.number_input(col, value=0.0)

st.markdown("---")

# ---------------- PREDICTION BUTTON ----------------
if st.button("🚀 Analyze Transaction", use_container_width=True):

    input_df = pd.DataFrame([input_data])
    probability = model.predict_proba(input_df)[0][1]

    # Risk Decision
    if probability >= threshold:
        decision = "🚨 FRAUD DETECTED"
        color = "red"
    else:
        decision = "✅ LEGITIMATE"
        color = "green"

    st.markdown("## 🔎 Prediction Result")

    # Risk Meter (Gauge)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Fraud Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<h2 style='color:{color};'>{decision}</h2>", unsafe_allow_html=True)

    # Probability Bar
    st.progress(float(probability))
