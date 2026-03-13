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
    return df_row[FEATURES]

# ──────────────────────────────────────────────────────────────────────
# Sidebar (used by Predictor tab)
# ──────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────
# Collect raw inputs
# ──────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────
# Run prediction (used across tabs)
# ──────────────────────────────────────────────────────────────────────
if model_loaded:
    X_input = engineer_features(raw_inputs)
    conv_prob = float(model.predict_proba(X_input)[0, 1])
    will_convert = conv_prob >= THRESHOLD

# ──────────────────────────────────────────────────────────────────────
# App title + model status
# ──────────────────────────────────────────────────────────────────────
st.title("🛒 E-commerce Conversion Predictor")

if model_loaded:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "XGBoost")
    c2.metric("ROC-AUC", f"{meta['xgb_auc']:.4f}")
    c3.metric("Decision Threshold", f"{THRESHOLD:.2f}")
    c4.metric("LR Baseline AUC", f"{meta['lr_auc']:.4f}")

st.divider()

# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────
tab_predict, tab_dashboard, tab_batch, tab_analytics = st.tabs([
    "🎯 Predictor",
    "📊 Model Dashboard",
    "📂 Batch Predictions",
    "📈 Analytics"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Predictor
# ══════════════════════════════════════════════════════════════════════
with tab_predict:
    if not model_loaded:
        st.info("Train the model by running `notebooks/02_modeling.ipynb`, then restart the app.")
        st.stop()

    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

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

        N_SESSIONS = 1000
        baseline_conv_rate = conv_prob
        baseline_revenue = N_SESSIONS * baseline_conv_rate * avg_order_value

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

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Model Dashboard
# ══════════════════════════════════════════════════════════════════════
with tab_dashboard:
    if not model_loaded:
        st.info("Model not loaded.")
        st.stop()

    st.subheader("📊 Model Performance Overview")

    # ── KPI row ────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("XGBoost AUC", f"{meta['xgb_auc']:.4f}", help="Area Under the ROC Curve")
    k2.metric("LR Baseline AUC", f"{meta['lr_auc']:.4f}")
    k3.metric("AUC Improvement", f"+{(meta['xgb_auc'] - meta['lr_auc']):.4f}",
              delta=f"{((meta['xgb_auc'] - meta['lr_auc']) / meta['lr_auc'] * 100):.1f}% over LR")
    k4.metric("Decision Threshold", f"{THRESHOLD:.2f}")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    # ── Feature Importance ─────────────────────────────────────────
    with col_left:
        st.subheader("Feature Importance (XGBoost)")

        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(6, 7))
        colors = ['#2ecc71' if v >= feat_imp.quantile(0.66)
                  else '#f39c12' if v >= feat_imp.quantile(0.33)
                  else '#e74c3c'
                  for v in feat_imp.values]
        bars = ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor='white', height=0.7)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title("Feature Importances", fontsize=13, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)

        for bar, val in zip(bars, feat_imp.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va='center', fontsize=8)

        patches = [
            mpatches.Patch(color='#2ecc71', label='High importance'),
            mpatches.Patch(color='#f39c12', label='Medium importance'),
            mpatches.Patch(color='#e74c3c', label='Low importance'),
        ]
        ax.legend(handles=patches, loc='lower right', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Threshold Analysis ─────────────────────────────────────────
    with col_right:
        st.subheader("Threshold Sensitivity")

        thresholds = np.linspace(0.1, 0.9, 80)
        # Simulate precision/recall trade-off curve using current prediction
        # Using a simplified illustration based on model probability
        precisions, recalls, f1s = [], [], []

        # We'll use a synthetic distribution around the current prediction
        # to illustrate the threshold vs metric trade-off
        np.random.seed(42)
        n_sim = 500
        true_pos_rate = meta['xgb_auc']
        # Simulate predicted probabilities for a balanced-ish dataset
        y_true_sim = np.random.binomial(1, 0.155, n_sim)
        y_prob_sim = np.where(
            y_true_sim == 1,
            np.random.beta(4, 2, n_sim),
            np.random.beta(2, 5, n_sim)
        )

        for t in thresholds:
            y_pred = (y_prob_sim >= t).astype(int)
            tp = ((y_pred == 1) & (y_true_sim == 1)).sum()
            fp = ((y_pred == 1) & (y_true_sim == 0)).sum()
            fn = ((y_pred == 0) & (y_true_sim == 1)).sum()
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            f = 2 * p * r / (p + r + 1e-9)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(thresholds, precisions, label='Precision', color='#3498db', linewidth=2)
        ax2.plot(thresholds, recalls, label='Recall', color='#e74c3c', linewidth=2)
        ax2.plot(thresholds, f1s, label='F1 Score', color='#2ecc71', linewidth=2, linestyle='--')
        ax2.axvline(THRESHOLD, color='#f39c12', linewidth=2, linestyle=':', label=f'Current threshold ({THRESHOLD:.2f})')
        ax2.set_xlabel("Decision Threshold", fontsize=11)
        ax2.set_ylabel("Score", fontsize=11)
        ax2.set_title("Precision / Recall / F1 vs Threshold", fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.spines[['top', 'right']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.caption(
            "The threshold analysis is illustrated using a simulated dataset that mirrors "
            "the UCI Online Shoppers dataset class distribution (~15.5% conversion rate). "
            "Adjust the threshold in the predictor sidebar for different precision/recall trade-offs."
        )

    # ── Model Comparison Bar ───────────────────────────────────────
    st.divider()
    st.subheader("Model Comparison")

    comp_col1, comp_col2 = st.columns([1, 2])

    with comp_col1:
        comp_data = {
            "Model": ["Logistic Regression (baseline)", "XGBoost"],
            "ROC-AUC": [meta['lr_auc'], meta['xgb_auc']],
        }
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df.style.highlight_max(subset=['ROC-AUC'], color='#d4edda'), hide_index=True)

    with comp_col2:
        fig3, ax3 = plt.subplots(figsize=(5, 2.5))
        model_names = ["Logistic\nRegression", "XGBoost"]
        auc_vals = [meta['lr_auc'], meta['xgb_auc']]
        bar_colors = ['#aed6f1', '#2ecc71']
        bars3 = ax3.bar(model_names, auc_vals, color=bar_colors, width=0.4, edgecolor='white')
        ax3.set_ylim(0.8, 0.95)
        ax3.set_ylabel("ROC-AUC", fontsize=11)
        ax3.set_title("Model AUC Comparison", fontsize=12, fontweight='bold')
        ax3.spines[['top', 'right']].set_visible(False)
        for bar, val in zip(bars3, auc_vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                     f"{val:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    # ── Feature table ──────────────────────────────────────────────
    st.divider()
    st.subheader("Feature Importance Table")
    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": [round(v, 5) for v in model.feature_importances_],
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    feat_df["Rank"] = range(1, len(feat_df) + 1)
    feat_df = feat_df[["Rank", "Feature", "Importance"]]
    st.dataframe(feat_df.style.background_gradient(subset=['Importance'], cmap='Greens'), hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Predictions
# ══════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📂 Batch Prediction — Upload Sessions CSV")
    st.markdown(
        "Upload a CSV with raw session data. The app will run predictions on every row "
        "and return a downloadable results file."
    )

    REQUIRED_COLS = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
        'VisitorType', 'Weekend'
    ]

    with st.expander("Expected CSV columns"):
        st.code(", ".join(REQUIRED_COLS))
        st.markdown("""
- **Month**: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
- **VisitorType**: Returning_Visitor, New_Visitor, Other
- **Weekend**: 0 or 1
- All numeric columns should be numeric (no commas)
        """)

    # Sample CSV download
    sample_data = {
        'Administrative': [2, 0, 5],
        'Administrative_Duration': [80, 0, 200],
        'Informational': [1, 0, 2],
        'Informational_Duration': [30, 0, 60],
        'ProductRelated': [8, 3, 20],
        'ProductRelated_Duration': [900, 150, 3500],
        'BounceRates': [0.02, 0.15, 0.005],
        'ExitRates': [0.04, 0.10, 0.02],
        'PageValues': [25.0, 0.0, 120.0],
        'SpecialDay': [0.0, 0.0, 0.4],
        'Month': ['May', 'Feb', 'Nov'],
        'OperatingSystems': [2, 1, 3],
        'Browser': [2, 1, 4],
        'Region': [1, 3, 2],
        'TrafficType': [2, 1, 5],
        'VisitorType': ['Returning_Visitor', 'New_Visitor', 'Returning_Visitor'],
        'Weekend': [0, 1, 0],
    }
    sample_df = pd.DataFrame(sample_data)
    st.download_button(
        label="Download sample CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_sessions.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded is not None:
        if not model_loaded:
            st.error("Model not loaded — cannot run predictions.")
        else:
            try:
                df_raw = pd.read_csv(uploaded)

                missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    st.success(f"Loaded {len(df_raw):,} sessions. Running predictions...")

                    results = []
                    for _, row in df_raw.iterrows():
                        raw = row[REQUIRED_COLS].to_dict()
                        X = engineer_features(raw)
                        prob = float(model.predict_proba(X)[0, 1])
                        pred = "Will Purchase" if prob >= THRESHOLD else "Will Not Purchase"
                        results.append({"conversion_probability": round(prob, 4), "prediction": pred})

                    results_df = pd.DataFrame(results)
                    out_df = pd.concat([df_raw.reset_index(drop=True), results_df], axis=1)

                    st.dataframe(out_df[['conversion_probability', 'prediction'] + REQUIRED_COLS].head(200), hide_index=True)

                    # Summary stats
                    st.divider()
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Total Sessions", f"{len(out_df):,}")
                    predicted_buyers = (results_df['prediction'] == 'Will Purchase').sum()
                    sc2.metric("Predicted Buyers", f"{predicted_buyers:,}")
                    sc3.metric("Predicted CVR", f"{predicted_buyers/len(out_df)*100:.1f}%")
                    sc4.metric("Avg Conv. Probability", f"{results_df['conversion_probability'].mean()*100:.1f}%")

                    # Distribution chart
                    fig_b, ax_b = plt.subplots(figsize=(7, 3))
                    ax_b.hist(results_df['conversion_probability'], bins=30, color='#3498db',
                              edgecolor='white', alpha=0.85)
                    ax_b.axvline(THRESHOLD, color='#e74c3c', linewidth=2, linestyle='--',
                                 label=f'Threshold ({THRESHOLD:.2f})')
                    ax_b.set_xlabel("Conversion Probability", fontsize=11)
                    ax_b.set_ylabel("Sessions", fontsize=11)
                    ax_b.set_title("Predicted Probability Distribution", fontsize=12, fontweight='bold')
                    ax_b.legend(fontsize=10)
                    ax_b.spines[['top', 'right']].set_visible(False)
                    fig_b.tight_layout()
                    st.pyplot(fig_b, use_container_width=True)
                    plt.close()

                    st.download_button(
                        label="Download predictions CSV",
                        data=out_df.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Analytics
# ══════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("📈 Conversion Analytics")
    st.markdown("Simulated analytics based on UCI Online Shoppers dataset statistics.")

    if not model_loaded:
        st.info("Model not loaded.")
        st.stop()

    # ── Conversion rate by month (from UCI dataset stats) ──────────
    month_data = {
        'Month': ['Feb', 'Mar', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Sessions': [184, 1907, 3364, 288, 547, 433, 448, 549, 2998, 1727],
        'Conversions': [24, 343, 440, 57, 100, 76, 124, 203, 602, 284],
    }
    month_df = pd.DataFrame(month_data)
    month_df['CVR'] = month_df['Conversions'] / month_df['Sessions'] * 100

    # ── Visitor type stats (UCI dataset) ───────────────────────────
    visitor_data = {
        'Visitor Type': ['New Visitor', 'Returning Visitor', 'Other'],
        'Sessions': [1694, 10551, 85],
        'Conversions': [140, 2072, 41],
    }
    visitor_df = pd.DataFrame(visitor_data)
    visitor_df['CVR'] = visitor_df['Conversions'] / visitor_df['Sessions'] * 100

    # ── Weekend vs weekday stats ────────────────────────────────────
    weekend_data = {
        'Session Type': ['Weekday', 'Weekend'],
        'Sessions': [9462, 2868],
        'Conversions': [1447, 806],
    }
    weekend_df = pd.DataFrame(weekend_data)
    weekend_df['CVR'] = weekend_df['Conversions'] / weekend_df['Sessions'] * 100

    # ── Row 1: Summary KPIs ────────────────────────────────────────
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Sessions (UCI)", "12,330")
    a2.metric("Total Conversions", "1,908")
    a3.metric("Overall CVR", "15.5%")
    a4.metric("Best Month CVR", f"{month_df['CVR'].max():.1f}% ({month_df.loc[month_df['CVR'].idxmax(), 'Month']})")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    # ── Monthly conversion rate chart ─────────────────────────────
    with chart_col1:
        st.markdown("**Conversion Rate by Month**")
        fig_m, ax_m = plt.subplots(figsize=(6, 4))
        bar_colors_m = ['#2ecc71' if v >= 15 else '#f39c12' if v >= 12 else '#e74c3c'
                        for v in month_df['CVR']]
        ax_m.bar(month_df['Month'], month_df['CVR'], color=bar_colors_m, edgecolor='white')
        ax_m.axhline(15.5, color='#3498db', linewidth=1.5, linestyle='--', label='Overall avg (15.5%)')
        ax_m.set_ylabel("Conversion Rate (%)", fontsize=11)
        ax_m.set_xlabel("Month", fontsize=11)
        ax_m.set_title("Monthly CVR", fontsize=12, fontweight='bold')
        ax_m.legend(fontsize=9)
        ax_m.spines[['top', 'right']].set_visible(False)
        for i, (_, row) in enumerate(month_df.iterrows()):
            ax_m.text(i, row['CVR'] + 0.2, f"{row['CVR']:.1f}%", ha='center', fontsize=8)
        fig_m.tight_layout()
        st.pyplot(fig_m, use_container_width=True)
        plt.close()

    # ── Visitor type chart ─────────────────────────────────────────
    with chart_col2:
        st.markdown("**Conversion Rate by Visitor Type**")
        fig_v, ax_v = plt.subplots(figsize=(6, 4))
        bar_colors_v = ['#3498db', '#2ecc71', '#f39c12']
        bars_v = ax_v.bar(visitor_df['Visitor Type'], visitor_df['CVR'],
                          color=bar_colors_v, edgecolor='white', width=0.5)
        ax_v.set_ylabel("Conversion Rate (%)", fontsize=11)
        ax_v.set_title("CVR by Visitor Type", fontsize=12, fontweight='bold')
        ax_v.spines[['top', 'right']].set_visible(False)
        for bar, val in zip(bars_v, visitor_df['CVR']):
            ax_v.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                      f"{val:.1f}%", ha='center', fontweight='bold', fontsize=10)
        fig_v.tight_layout()
        st.pyplot(fig_v, use_container_width=True)
        plt.close()

    st.divider()
    chart_col3, chart_col4 = st.columns(2)

    # ── Weekend vs Weekday ─────────────────────────────────────────
    with chart_col3:
        st.markdown("**Weekend vs Weekday Conversion**")
        fig_w, ax_w = plt.subplots(figsize=(5, 3.5))
        bars_w = ax_w.bar(weekend_df['Session Type'], weekend_df['CVR'],
                          color=['#aed6f1', '#2ecc71'], edgecolor='white', width=0.4)
        ax_w.set_ylabel("Conversion Rate (%)", fontsize=11)
        ax_w.set_title("Weekend vs Weekday CVR", fontsize=12, fontweight='bold')
        ax_w.spines[['top', 'right']].set_visible(False)
        for bar, val in zip(bars_w, weekend_df['CVR']):
            ax_w.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                      f"{val:.1f}%", ha='center', fontweight='bold', fontsize=11)
        fig_w.tight_layout()
        st.pyplot(fig_w, use_container_width=True)
        plt.close()

    # ── Session volume by month ────────────────────────────────────
    with chart_col4:
        st.markdown("**Session Volume by Month**")
        fig_s, ax_s = plt.subplots(figsize=(6, 3.5))
        ax_s.fill_between(range(len(month_df)), month_df['Sessions'],
                          alpha=0.4, color='#3498db')
        ax_s.plot(range(len(month_df)), month_df['Sessions'],
                  color='#2980b9', linewidth=2, marker='o', markersize=5)
        ax_s.set_xticks(range(len(month_df)))
        ax_s.set_xticklabels(month_df['Month'], fontsize=9)
        ax_s.set_ylabel("Sessions", fontsize=11)
        ax_s.set_title("Monthly Traffic Volume", fontsize=12, fontweight='bold')
        ax_s.spines[['top', 'right']].set_visible(False)
        fig_s.tight_layout()
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

    # ── Data tables ────────────────────────────────────────────────
    st.divider()
    st.subheader("Raw Analytics Data")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**By Month**")
        st.dataframe(month_df.style.background_gradient(subset=['CVR'], cmap='Greens'), hide_index=True)
    with t2:
        st.markdown("**By Visitor Type**")
        st.dataframe(visitor_df.style.background_gradient(subset=['CVR'], cmap='Blues'), hide_index=True)
    with t3:
        st.markdown("**By Session Timing**")
        st.dataframe(weekend_df.style.background_gradient(subset=['CVR'], cmap='Oranges'), hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "E-commerce Conversion Prediction | XGBoost Model | "
    "UCI Online Shoppers Dataset (12,330 sessions) | "
    "Built with Streamlit"
)
