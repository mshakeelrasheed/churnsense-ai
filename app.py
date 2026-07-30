import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. PAGE ENGINE & THEME CONFIG
# ==========================================
st.set_page_config(
    page_title="ChurnSense AI — Intelligence Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection for dark-mode modern SaaS design frameworks
st.markdown("""
    <style>
    /* Styling high-contrast metric values */
    div[data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-size: 2.3rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        color: #8A99AD;
    }

    /* Premium animated header styles */
    .premium-header-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        font-family: 'Inter', sans-serif;
        margin-bottom: 15px;
    }

    .premium-header-container .static-part {
        color: white;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: -10px;
    }

    .premium-header-container .gradient-part {
        font-size: 3.2rem !important; 
        font-weight: 800 !important;
        white-space: nowrap !important; 
        background-image: linear-gradient(120deg, #E05C6F 0%, #A378D1 35%, #6DAEED 70%, #6DAEED 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        color: transparent;
        animation: premiumGradientSweep 6s linear infinite;
        line-height: 1.2;
    }

    @keyframes premiumGradientSweep {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* ==========================================
       THE BULLETPROOF HTML CARD GLOW
       ========================================== */
    .custom-glow-card {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        background-color: #0E1117; /* Matches standard Streamlit dark mode */
        transition: all 0.3s ease-in-out !important;
        box-sizing: border-box;
        height: 100%;
    }

    .custom-glow-card:hover {
        border-color: #00E5FF !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.4), inset 0 0 10px rgba(0, 229, 255, 0.15) !important;
        transform: translateY(-4px) !important;
    }

    /* Ensure standard Streamlit widgets at the bottom keep glowing */
    [data-testid="stExpander"] details,
    [data-testid="stStatusWidget"] div[role="status"] {
        border-radius: 16px !important;
        transition: all 0.3s ease-in-out !important;
    }

    [data-testid="stExpander"] details:hover,
    [data-testid="stStatusWidget"] div[role="status"]:hover {
        border-color: #00E5FF !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.4), inset 0 0 10px rgba(0, 229, 255, 0.15) !important;
        transform: translateY(-4px) !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Navigation Header Row
col_header, col_badge = st.columns([4, 1])
with col_header:
    premium_header_html = """
    <div class='premium-header-container'>
        <div class='static-part'>🔮 ChurnSense AI &mdash;</div>
        <div class='gradient-part'>Customer Intelligence Hub</div>
    </div>
    """
    st.markdown(premium_header_html, unsafe_allow_html=True)
    st.caption("Identify customer risk, understand key drivers, and view suggested retention strategies.")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.status("Model Version v2.1", state="complete")

st.markdown("---")

# ==========================================
# 2. CACHED MODEL & PREPROCESSOR ARTIFACTS
# ==========================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, scaler, encoders

try:
    model, scaler, encoders = load_artifacts()
except Exception as e:
    st.error(f"❌ Critical error loading model artifacts: {e}")
    st.stop()

# Feature order expected by scaler and model
FEATURE_ORDER = [
    'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
    'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
    'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount'
]

# ==========================================
# 3. INTERACTIVE SIDEBAR CONFIGURATOR
# ==========================================
st.sidebar.markdown("### 🎯 **Customer Profile**")

# Toggle for Auto vs Manual Execution
auto_mode = st.sidebar.toggle("⚡ Auto-Update Predictions", value=True)
analyze_btn = st.sidebar.button("🔍 Analyze Profile", disabled=auto_mode, type="primary")

st.sidebar.markdown("---")

# Primary High-Impact Features
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=60, value=2)
complain = st.sidebar.segmented_control("Customer Has Active Complaint?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", default=0)
days_since_last_order = st.sidebar.number_input("Days Since Last Order", min_value=0, max_value=60, value=12)
satisfaction_score = st.sidebar.slider("Satisfaction Score", 1, 5, 2)
number_of_address = st.sidebar.number_input("Number of Addresses Registered", min_value=1, max_value=20, value=3)
cashback_amount = st.sidebar.number_input("Cashback Amount ($)", min_value=0.0, max_value=500.0, value=120.0)
preferred_order_cat = st.sidebar.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
preferred_payment = st.sidebar.selectbox("Preferred Payment Mode", ["Debit Card", "Credit Card", "E Wallet", "UPI", "COD"])

# Collapsed Secondary Features
with st.sidebar.expander("⚙️ Additional Profile Details", expanded=False):
    login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer", "Phone"])
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    warehouse_to_home = st.number_input("Warehouse to Home Distance (km)", min_value=1, max_value=150, value=15)
    gender = st.selectbox("Gender", ["Female", "Male"])
    hours_spend_app = st.slider("Hours Spent on App / Week", 0, 10, 3)
    devices_registered = st.number_input("Devices Registered", min_value=1, max_value=10, value=3)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    order_hike = st.number_input("Order Amount Hike From Last Year (%)", min_value=0, max_value=50, value=15)
    coupon_used = st.number_input("Coupons Used", min_value=0, max_value=30, value=1)
    order_count = st.number_input("Total Order Count", min_value=1, max_value=100, value=2)

# Execution Gate: Stop rendering if manual mode is on and button hasn't been clicked
if not auto_mode and not analyze_btn:
    st.info("👈 Update the customer details in the sidebar, then click **'Analyze Profile'** to view the churn risk and recommendations.")
    st.stop()

# ==========================================
# 4. RUN INFERENCE & PROCESS DATA
# ==========================================
# Collect Raw Inputs
raw_inputs = {
    'Tenure': tenure, 'PreferredLoginDevice': login_device, 'CityTier': city_tier,
    'WarehouseToHome': warehouse_to_home, 'PreferredPaymentMode': preferred_payment,
    'Gender': gender, 'HourSpendOnApp': hours_spend_app, 'NumberOfDeviceRegistered': devices_registered,
    'PreferedOrderCat': preferred_order_cat, 'SatisfactionScore': satisfaction_score,
    'MaritalStatus': marital_status, 'NumberOfAddress': number_of_address, 'Complain': complain,
    'OrderAmountHikeFromlastYear': order_hike, 'CouponUsed': coupon_used, 'OrderCount': order_count,
    'DaySinceLastOrder': days_since_last_order, 'CashbackAmount': cashback_amount
}

# Encode Categoricals Safely
encoded_inputs = raw_inputs.copy()
for col in ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']:
    if col in encoders:
        try:
            encoded_inputs[col] = encoders[col].transform([raw_inputs[col]])[0]
        except Exception:
            encoded_inputs[col] = 0

# Convert to structured frame mapping
input_df = pd.DataFrame([encoded_inputs])[FEATURE_ORDER]
scaled_features = scaler.transform(input_df)
scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_ORDER)

with st.spinner("Analyzing customer profile..."):
    if not auto_mode:
        time.sleep(0.5) 
    churn_prob = float(model.predict_proba(scaled_df)[0][1])
    churn_percent = churn_prob * 100

# Compute SHAP math elements behind scenes
explainer = shap.TreeExplainer(model)
shap_vals = explainer(scaled_df)
sample_shap = pd.Series(shap_vals.values[0], index=FEATURE_ORDER)
top_risk_factors = sample_shap[sample_shap > 0].sort_values(ascending=False)

# ==========================================
# 5. SPLIT-PANEL CORE DISPLAY ANALYSIS GRID (BULLETPROOF HTML)
# ==========================================
col_left, col_right = st.columns([1, 1.1])

# Dynamic Status Math and CSS generation
if churn_prob >= 0.5:
    alert_bg = "rgba(255, 75, 75, 0.15)"
    alert_border = "rgba(255, 75, 75, 0.4)"
    alert_title = "🚨 High Churn Risk"
    alert_msg = "Alert: This customer is likely to leave."
    alert_text = "#ff4b4b"
    delta_bg = "rgba(255, 75, 75, 0.2)"
    delta_color = "#ff4b4b"
    delta_arrow = "↑"
else:
    alert_bg = "rgba(9, 171, 59, 0.15)"
    alert_border = "rgba(9, 171, 59, 0.4)"
    alert_title = "✅ Low Churn Risk"
    alert_msg = "Status: This customer is currently safe."
    alert_text = "#09ab3b"
    delta_bg = "rgba(9, 171, 59, 0.2)"
    delta_color = "#09ab3b"
    delta_arrow = "↓"

delta_val = f"{delta_arrow} {abs(churn_percent - 50):.1f}% vs baseline"

with col_left:
    # Removed indentation so Markdown parser doesn't convert it into a code block
    left_card_html = f"""<div class="custom-glow-card">
<h3 style="margin-top:0; font-size: 1.5rem; font-weight: 600; color: white;">🎯 Risk Assessment</h3>
<div style="background-color: {alert_bg}; border: 1px solid {alert_border}; border-radius: 8px; padding: 16px; margin: 16px 0;">
<h4 style="margin: 0 0 8px 0; color: {alert_text}; font-size: 1.25rem;">{alert_title}</h4>
<p style="margin: 0; font-size: 1rem; color: #e0e0e0;">{alert_msg}</p>
<p style="margin: 12px 0 0 0; font-size: 1rem; color: #e0e0e0;">Probability of leaving: <strong>{churn_percent:.1f}%</strong></p>
</div>
<div style="display: flex; flex-direction: column; margin-top: 24px;">
<span style="font-size: 0.85rem; color: #8A99AD; text-transform: uppercase; letter-spacing: 0.08rem;">Overall Risk Score</span>
<span style="font-size: 2.3rem; font-weight: 700; margin: 6px 0; color: white;">{churn_percent:.1f}%</span>
<span style="color: {delta_color}; font-weight: 600; font-size: 0.95rem; background: {delta_bg}; width: fit-content; padding: 4px 10px; border-radius: 12px;">{delta_val}</span>
</div>
</div>"""
    st.markdown(left_card_html, unsafe_allow_html=True)

with col_right:
    factors_html = ""
    if len(top_risk_factors) > 0:
        for feat, val in top_risk_factors.head(4).items():
            raw_val = raw_inputs[feat]
            factors_html += f'<div style="margin-bottom: 16px; color: #e0e0e0; font-size: 1rem;">🔹 <strong>{feat}</strong>: <span style="color: #00E5FF; background: rgba(0, 229, 255, 0.1); padding: 4px 8px; border-radius: 6px; font-family: monospace; font-size: 0.95rem;">{raw_val}</span></div>'
    else:
        factors_html = '<div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px; color: #e0e0e0;">No major risk factors detected for this profile.</div>'

    # Removed indentation so Markdown parser doesn't convert it into a code block
    right_card_html = f"""<div class="custom-glow-card">
<h3 style="margin-top:0; font-size: 1.5rem; font-weight: 600; color: white;">💡 Main Risk Factors</h3>
<p style="color: #8A99AD; font-size: 1rem; margin-bottom: 24px;">The main reasons influencing this customer's risk score:</p>
{factors_html}
</div>"""
    st.markdown(right_card_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. SUGGESTED RECOMMENDATIONS 
# ==========================================
st.subheader("📋 **Suggested Actions**")

recs = []
if complain == 1:
    recs.append(("🚨 Resolve Complaint", "This customer has an active complaint. Reach out immediately to resolve their issue and offer a small store credit as an apology."))
if satisfaction_score <= 2:
    recs.append(("⭐ Improve Satisfaction", "The customer's satisfaction rating is low. Send a brief follow-up email to understand what went wrong during their last experience."))
if days_since_last_order > 14:
    recs.append(("🛒 Re-engagement Offer", f"It has been a while since their last order. Send them a personalized discount code for their favorite category: **{preferred_order_cat}**."))
if tenure <= 3:
    recs.append(("🆕 Welcome Follow-up", "This is a new customer. Send them a welcome guide or a special perk to build early loyalty."))
if cashback_amount < 100:
    recs.append(("💰 Offer Incentives", "Increase their cashback reward slightly on their next purchase to encourage them to buy again."))

if not recs:
    with st.container(border=True):
        st.balloons() 
        st.success("🎉 **Customer is Healthy:** No immediate action is required. Maintain standard marketing communications.")
else:
    with st.status("Generating recommendations based on profile...", expanded=True) as playbook_status:
        if auto_mode:
            time.sleep(0.3)
        for label, description in recs:
            with st.container(border=True):
                st.markdown(f"**{label}**\n\n{description}")

st.markdown("---")

# ==========================================
# 7. ADVANCED SHAP ANALYSIS (COLLAPSIBLE)
# ==========================================
with st.expander("🔍 **Advanced Analysis (SHAP Graph)**", expanded=False):
    st.caption("This visualization breaks down exactly how much each feature contributed to the final probability calculation.")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_vals[0], show=False)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    st.pyplot(fig)
