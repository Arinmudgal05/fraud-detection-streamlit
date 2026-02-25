import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="💳 AI Fraud Risk Engine",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
model = joblib.load("fraud_model.pkl")

with open("threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HEADER ----------------
st.markdown("""
# 💳 AI-Powered Fraud Risk Monitoring System
### Real-Time Transaction Risk Engine
""")

st.markdown("---")

# ================================
# SECTION 1 — TRANSACTION SIMULATOR
# ================================

st.subheader("📥 Transaction Simulator")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

with col2:
    time = st.number_input("Transaction Time (seconds)", min_value=0.0, value=0.0)

with col3:
    generate = st.button("🎲 Generate Random Transaction")

# Generate random PCA features
if generate:
    random_features = np.random.normal(0, 1, len(feature_columns))
    input_data = dict(zip(feature_columns, random_features))
    input_data["Amount"] = amount
    input_data["Time"] = time
else:
    input_data = dict(zip(feature_columns, np.zeros(len(feature_columns))))
    input_data["Amount"] = amount
    input_data["Time"] = time

input_df = pd.DataFrame([input_data])

st.markdown("---")

# ================================
# SECTION 2 — ANALYZE BUTTON
# ================================

analyze = st.button("🚀 Analyze Transaction", use_container_width=True)

if analyze:

    probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.session_state.history.append(probability)

    st.markdown("## 🔎 Risk Assessment Result")

    # ================================
    # SECTION 3 — RESULT PANEL
    # ================================

    colA, colB = st.columns([2, 1])

    with colA:
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

    with colB:
        if prediction == 1:
            st.error("🚨 HIGH RISK - FRAUD DETECTED")
        else:
            st.success("✅ LOW RISK - LEGITIMATE")

        st.metric("Confidence Score", f"{probability*100:.2f}%")

    # SHAP Explainability Toggle
    if st.checkbox("🔍 Show SHAP Explanation"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, input_df)
        st.pyplot(bbox_inches='tight')

st.markdown("---")

# ================================
# SECTION 4 — MONITORING DASHBOARD
# ================================

st.subheader("📊 Live Monitoring Dashboard")

if len(st.session_state.history) > 0:

    total_tx = len(st.session_state.history)
    fraud_count = len([p for p in st.session_state.history if p >= threshold])
    fraud_rate = (fraud_count / total_tx) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total_tx)
    col2.metric("Fraud Detected", fraud_count)
    col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}%")

    # Risk Distribution Chart
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=st.session_state.history,
        nbinsx=20
    ))
    fig2.update_layout(
        title="Fraud Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="Count"
    )

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("No transactions analyzed yet.")
