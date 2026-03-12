"""
E-commerce Conversion Prediction — Streamlit App
Run: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-commerce Conversion Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────
# Load model artifacts
# ──────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, 'xgb_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_meta.json'), 'r') as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_artifacts()
    FEATURES = meta['features']
    THRESHOLD = meta['best_threshold']
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning(
        "Model not found. Please run `notebooks/02_modeling.ipynb` first to train and save the model.",
        icon="⚠️"
    )

# ──────────────────────────────────────────────────────────────────────
# Feature engineering helper (mirrors notebook 02)
# ──────────────────────────────────────────────────────────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    row = raw.copy()
    total_dur = (row['Administrative_Duration'] + row['Informational_Duration'] +
                 row['ProductRelated_Duration'] + 1e-6)
    row['session_engagement_score'] = (
        row['ProductRelated'] * 0.4 +
        (row['ProductRelated_Duration'] / (5000 + 1)) * 0.3 +
        (row['PageValues'] / (400 + 1)) * 0.3
    )
    row['product_time_ratio'] = row['ProductRelated_Duration'] / total_dur
    row['total_pages'] = row['Administrative'] + row['Informational'] + row['ProductRelated']
    row['avg_product_page_time'] = row['ProductRelated_Duration'] / (row['ProductRelated'] + 1)
    row['bounce_exit_score'] = row['BounceRates'] * row['ExitRates']
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    row['month_num'] = month_map.get(row.pop('Month', 'May'), 5)
    visitor_map = {'Returning_Visitor': 2, 'New_Visitor': 1, 'Other': 0}
    row['visitor_type_enc'] = visitor_map.get(row.pop('VisitorType', 'New_Visitor'), 1)
    row['Weekend'] = int(row['Weekend'])
    df_row = pd.DataFrame([row])
    # Keep only model features, in correct order
    return df_row[FEATURES]

# ──────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────
st.title("🛒 E-commerce Conversion Predictor")
st.markdown(
    "Enter session behavior metrics to predict **purchase conversion probability** "
    "and simulate the **ROI of a targeted discount campaign**."
)

if model_loaded:
    st.success(
        f"Model loaded — XGBoost | ROC-AUC: **{meta['xgb_auc']:.3f}** | "
        f"Decision threshold: **{THRESHOLD:.2f}**",
        icon="✅"
    )

st.divider()

# ── Sidebar: Session inputs ────────────────────────────────────────────
st.sidebar.header("Session Behavior Inputs")
st.sidebar.markdown("*Adjust sliders to match the browsing session profile.*")

with st.sidebar:
    st.subheader("Page Engagement")
    administrative = st.slider("Administrative Pages Visited", 0, 20, 2)
    administrative_duration = st.slider("Administrative Duration (sec)", 0, 1000, 80)
    informational = st.slider("Informational Pages Visited", 0, 10, 1)
    informational_duration = st.slider("Informational Duration (sec)", 0, 600, 30)
    product_related = st.slider("Product Pages Visited", 0, 50, 8)
    product_related_duration = st.slider("Product Duration (sec)", 0, 10000, 900)

    st.subheader("Engagement Quality")
    bounce_rates = st.slider("Bounce Rate", 0.0, 0.20, 0.02, step=0.005, format="%.3f")
    exit_rates = st.slider("Exit Rate", 0.0, 0.20, 0.04, step=0.005, format="%.3f")
    page_values = st.slider("Page Value", 0.0, 400.0, 25.0, step=1.0)
    special_day = st.slider("Special Day Proximity (0=no, 1=close)", 0.0, 1.0, 0.0, step=0.2)

    st.subheader("Session Context")
    month = st.selectbox("Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=4)
    visitor_type = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'])
    weekend = st.checkbox("Weekend Session", value=False)
    operating_system = st.selectbox("Operating System (encoded)", [1, 2, 3, 4, 5, 6, 7, 8], index=1)
    browser = st.selectbox("Browser (encoded)", list(range(1, 14)), index=1)
    region = st.selectbox("Region (encoded)", list(range(1, 10)), index=0)
    traffic_type = st.selectbox("Traffic Type (encoded)", list(range(1, 21)), index=1)

    st.divider()
    st.subheader("Campaign Simulation")
    avg_order_value = st.number_input("Avg Order Value ($)", value=85, min_value=10, max_value=500)
    discount_pct = st.slider("Discount Offered (%)", 5, 30, 10) / 100
    lift_rate = st.slider("Expected Conversion Lift (%)", 10, 50, 30) / 100
    campaign_cost = st.number_input("Campaign Cost per Session ($)", value=2.0,
                                    min_value=0.0, max_value=20.0, step=0.5)

# ── Main: Prediction ───────────────────────────────────────────────────
raw_inputs = {
    'Administrative': administrative,
    'Administrative_Duration': administrative_duration,
    'Informational': informational,
    'Informational_Duration': informational_duration,
    'ProductRelated': product_related,
    'ProductRelated_Duration': product_related_duration,
    'BounceRates': bounce_rates,
    'ExitRates': exit_rates,
    'PageValues': page_values,
    'SpecialDay': special_day,
    'Month': month,
    'OperatingSystems': operating_system,
    'Browser': browser,
    'Region': region,
    'TrafficType': traffic_type,
    'VisitorType': visitor_type,
    'Weekend': int(weekend),
}

col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

if model_loaded:
    X_input = engineer_features(raw_inputs)
    conv_prob = float(model.predict_proba(X_input)[0, 1])
    will_convert = conv_prob >= THRESHOLD

    # ── Column 1: Conversion probability gauge ─────────────────────
    with col1:
        st.subheader("Conversion Probability")
        color = "#2ecc71" if conv_prob >= 0.5 else ("#f39c12" if conv_prob >= 0.25 else "#e74c3c")

        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
        theta_end = np.pi * conv_prob
        ax.barh(1, theta_end, left=0, color=color, height=0.5, alpha=0.85)
        ax.barh(1, np.pi - theta_end, left=theta_end, color='#ecf0f1', height=0.5, alpha=0.5)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_ylim(0, 2)
        ax.axis('off')
        ax.text(np.pi / 2, 0.4, f"{conv_prob*100:.1f}%",
                ha='center', va='center', fontsize=26, fontweight='bold', color=color,
                transform=ax.transData)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        verdict = "WILL PURCHASE" if will_convert else "WILL NOT PURCHASE"
        verdict_color = "green" if will_convert else "red"
        st.markdown(
            f"<div style='text-align:center; font-size:18px; font-weight:bold; color:{verdict_color}'>"
            f"{verdict}</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Threshold: {THRESHOLD:.2f}")

    # ── Column 2: Feature contribution bars ────────────────────────
    with col2:
        st.subheader("Key Signal Indicators")

        indicators = {
            "Page Value": min(page_values / 100, 1.0),
            "Product Engagement": min(product_related / 30, 1.0),
            "Browse Time": min(product_related_duration / 3000, 1.0),
            "Low Bounce": max(0, 1 - bounce_rates * 10),
            "Session Depth": min((product_related + informational + administrative) / 20, 1.0),
        }

        for name, score in indicators.items():
            bar_color = "#2ecc71" if score > 0.6 else ("#f39c12" if score > 0.3 else "#e74c3c")
            st.markdown(f"**{name}**")
            st.progress(score)

        st.markdown("---")
        st.metric("Predicted Probability", f"{conv_prob*100:.1f}%")
        st.metric("Visitor Type", visitor_type.replace('_', ' '))
        st.metric("Month", month)

    # ── Column 3: Campaign ROI simulation ──────────────────────────
    with col3:
        st.subheader("Campaign ROI Simulation")
        st.markdown(
            f"*What if you offered a **{discount_pct*100:.0f}% discount** to "
            f"this session type and expected a **{lift_rate*100:.0f}% lift** in conversion?*"
        )

        # For this single session type, simulate across 1000 similar sessions
        N_SESSIONS = 1000
        baseline_conv_rate = conv_prob
        baseline_revenue = N_SESSIONS * baseline_conv_rate * avg_order_value

        # With campaign: non-converters get a lift
        non_conv_rate = 1 - baseline_conv_rate
        new_conv_rate = baseline_conv_rate + non_conv_rate * lift_rate
        incremental_conversions = N_SESSIONS * (new_conv_rate - baseline_conv_rate)
        inc_revenue = incremental_conversions * avg_order_value * (1 - discount_pct)
        disc_cost = N_SESSIONS * baseline_conv_rate * avg_order_value * discount_pct
        op_cost = N_SESSIONS * campaign_cost
        net_uplift = inc_revenue - disc_cost - op_cost
        roi = (net_uplift / (disc_cost + op_cost)) * 100 if (disc_cost + op_cost) > 0 else 0

        m1, m2 = st.columns(2)
        m1.metric("Baseline CVR", f"{baseline_conv_rate*100:.1f}%")
        m2.metric("Campaign CVR", f"{new_conv_rate*100:.1f}%",
                  delta=f"+{(new_conv_rate-baseline_conv_rate)*100:.1f}pp")

        m3, m4 = st.columns(2)
        m3.metric("Net Uplift (1K sessions)", f"${net_uplift:,.0f}",
                  delta="positive" if net_uplift > 0 else "negative")
        m4.metric("Campaign ROI", f"{roi:.0f}%",
                  delta=f"{'Profitable' if roi > 0 else 'Loss'}")

        with st.expander("Simulation breakdown"):
            st.markdown(f"""
| Metric | Value |
|---|---|
| Sessions simulated | {N_SESSIONS:,} |
| Baseline conversions | {N_SESSIONS * baseline_conv_rate:.0f} |
| Campaign conversions | {N_SESSIONS * new_conv_rate:.0f} |
| Incremental revenue | ${inc_revenue:,.0f} |
| Discount cost (existing) | $-{disc_cost:,.0f} |
| Operational cost | $-{op_cost:,.0f} |
| **Net uplift** | **${net_uplift:,.0f}** |
| **ROI** | **{roi:.0f}%** |
            """)

        if net_uplift > 0:
            st.success(f"Campaign is **profitable** at this lift rate. Break-even lift: "
                       f"{((disc_cost + op_cost) / (N_SESSIONS * non_conv_rate * avg_order_value * (1 - discount_pct)))*100:.1f}%")
        else:
            st.error("Campaign is **not profitable** at this lift rate. Increase discount lift or reduce costs.")

else:
    st.info("Train the model by running `notebooks/02_modeling.ipynb`, then restart the app.")

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "E-commerce Conversion Prediction | XGBoost Model | "
    "UCI Online Shoppers Dataset (12,330 sessions) | "
    "Built with Streamlit"
)
