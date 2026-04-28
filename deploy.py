import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Dashboard", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    h1 { color: #4f9fff !important; text-align: center; font-weight: 800; }
    .metric-container { background: #1e2235; border-radius: 10px; padding: 15px; border: 1px solid #3a3f5c; }
    .result-approved { background: linear-gradient(135deg, #0d2e1a, #0a3d20); border: 2px solid #22c55e; border-radius: 16px; padding: 25px; text-align: center; }
    .result-rejected { background: linear-gradient(135deg, #2e0d0d, #3d0a0a); border: 2px solid #ef4444; border-radius: 16px; padding: 25px; text-align: center; }
    .sidebar-section { color: #4f9fff; font-weight: 700; text-transform: uppercase; font-size: 12px; margin-top: 20px; border-bottom: 1px solid #2d3045; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m, s = joblib.load("credit_risk_model.pkl"), joblib.load("scaler.pkl")
    return m, s


model, scaler = load_models()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def build_feature_vector(income, age, years_emp, children, gender, education, marital):
    actual_features = scaler.feature_names_in_
    row = {col: 0 for col in actual_features}

    # Mapping values
    income_log = np.log1p(income)
    inc_per_child = income / (children + 1)
    emp_age_ratio = years_emp / age if age > 0 else 0

    # Fill base features
    for k, v in {"Income": income_log, "Income_log": income_log, "Age": age,
                 "Years_of_Employment": years_emp, "Children": children,
                 "Children count": children, "Income_per_child": inc_per_child,
                 "Employment_Age_Ratio": emp_age_ratio}.items():
        if k in row: row[k] = v

    # One-Hot Encoding
    for col in row:
        if f"Gender_{gender}" in col or f"Education level_{education}" in col or f"Marital status_{marital}" in col:
            row[col] = 1

    return pd.DataFrame([row])[actual_features]


def get_explanation(prob, income, years_emp, age):
    if prob < 0.4:
        reasons = []
        if income > 100000: reasons.append("Strong income")
        if years_emp > 3: reasons.append("Stable employment")
        return "✅ Approved: " + " & ".join(reasons) if reasons else "✅ Approved based on low overall risk profile."
    else:
        return "🚨 Rejected: High probability of default detected based on history and financial ratio."


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Settings")
    income = st.slider("Annual Income", 10000, 500000, 120000)
    children = st.number_input("Children", 0, 10, 0)
    age = st.slider("Age", 18, 75, 35)
    gender = st.selectbox("Gender", ["M", "F"])
    marital = st.selectbox("Marital Status",
                           ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
    education = st.selectbox("Education", ["Higher education", "Secondary / secondary special", "Incomplete higher",
                                           "Lower secondary"])
    years_emp = st.slider("Years of Employment", 0, 40, 5)
    predict_btn = st.button("Predict Credit Risk", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────
st.title("🏦 Credit Risk Dashboard")

# Top Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Income", f"${income:,.0f}")
c2.metric("Age", f"{age} yrs")
c3.metric("Employment", f"{years_emp} yrs")
c4.metric("Emp/Age Ratio", f"{(years_emp / age if age > 0 else 0):.2f}")

st.divider()

if predict_btn:
    X = build_feature_vector(income, age, years_emp, children, gender, education, marital)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### Assessment Result")
        if prob < 0.4:
            st.markdown(f"""<div class="result-approved">
                <h2 style="color:#22c55e">✅ APPROVED</h2>
                <p style="font-size:20px">Low Risk ({prob * 100:.1f}%)</p>
                <p style="color:#b0bec5">{get_explanation(prob, income, years_emp, age)}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-rejected">
                <h2 style="color:#ef4444">❌ REJECTED</h2>
                <p style="font-size:20px">High Risk ({prob * 100:.1f}%)</p>
                <p style="color:#b0bec5">{get_explanation(prob, income, years_emp, age)}</p>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("### Risk Analysis")
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={'suffix': "%", 'font': {'color': '#e0e6f0'}},
            gauge={'bar': {'color': "#22c55e" if prob < 0.4 else "#ef4444"},
                   'bgcolor': "#1e2235", 'axis': {'range': [0, 100]}}
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Probability Bar
        st.markdown("**Probability Breakdown**")
        progress_color = "success" if prob < 0.4 else "danger"
        st.progress(float(prob))
        st.caption(f"Default Probability: {prob * 100:.2f}% | Safe Margin: {(1 - prob) * 100:.2f}%")

else:
    st.markdown("""
    <div style="text-align:center; padding:50px; border:2px dashed #3a3f5c; border-radius:20px">
        <h3 style="color:#8892b0">🔍 Ready for Analysis</h3>
        <p style="color:#a8b2d1">Please enter the customer details on the left and click <b>Predict Risk</b> to start.</p>
    </div>
    """, unsafe_allow_html=True)