import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(
    page_title="ChurnSense AI — Intelligence Dashboard",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 ChurnSense AI — Customer Churn Intelligence")
st.write("Predict customer churn risk, identify top risk factors, and view automated retention strategies.")

# 2. Load Model & Preprocessors
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, scaler, encoders

try:
    model, scaler, encoders = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Feature order expected by scaler and model
FEATURE_ORDER = [
    'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
    'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
    'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount'
]

# 3. Sidebar Inputs
st.sidebar.header("🎯 Primary Customer Profile")

# Primary High-Impact Features
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=60, value=2)
complain = st.sidebar.selectbox("Customer Has Active Complaint?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
days_since_last_order = st.sidebar.number_input("Days Since Last Order", min_value=0, max_value=60, value=12)
satisfaction_score = st.sidebar.slider("Satisfaction Score (1 = Low, 5 = High)", 1, 5, 2)
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

# Collect Raw Inputs
raw_inputs = {
    'Tenure': tenure,
    'PreferredLoginDevice': login_device,
    'CityTier': city_tier,
    'WarehouseToHome': warehouse_to_home,
    'PreferredPaymentMode': preferred_payment,
    'Gender': gender,
    'HourSpendOnApp': hours_spend_app,
    'NumberOfDeviceRegistered': devices_registered,
    'PreferedOrderCat': preferred_order_cat,
    'SatisfactionScore': satisfaction_score,
    'MaritalStatus': marital_status,
    'NumberOfAddress': number_of_address,
    'Complain': complain,
    'OrderAmountHikeFromlastYear': order_hike,
    'CouponUsed': coupon_used,
    'OrderCount': order_count,
    'DaySinceLastOrder': days_since_last_order,
    'CashbackAmount': cashback_amount
}

# Encode Categoricals
encoded_inputs = raw_inputs.copy()
for col in ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']:
    if col in encoders:
        try:
            encoded_inputs[col] = encoders[col].transform([raw_inputs[col]])[0]
        except Exception:
            encoded_inputs[col] = 0

# Convert to DataFrame in exact feature order
input_df = pd.DataFrame([encoded_inputs])[FEATURE_ORDER]

# Scale features
scaled_features = scaler.transform(input_df)
scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_ORDER)

# 4. Model Prediction
churn_prob = float(model.predict_proba(scaled_df)[0][1])
churn_percent = churn_prob * 100

st.divider()

# 5. Top Section: Prediction & Key Risk Drivers
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("🎯 Risk Assessment")
    if churn_prob >= 0.5:
        st.error(f"### 🚨 High Churn Risk\n**Probability of Churning:** {churn_percent:.1f}%")
    else:
        st.success(f"### ✅ Low Churn Risk\n**Probability of Churning:** {churn_percent:.1f}%")

    st.metric(
        label="Overall Churn Score", 
        value=f"{churn_percent:.1f}%", 
        delta=f"{'+' if churn_prob >= 0.5 else '-'}{abs(churn_percent - 50):.1f}% relative to baseline threshold"
    )

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_vals = explainer(scaled_df)
sample_shap = pd.Series(shap_vals.values[0], index=FEATURE_ORDER)

# Positive SHAP = Increasing Churn Risk
top_risk_factors = sample_shap[sample_shap > 0].sort_values(ascending=False)

with col_right:
    st.subheader("💡 Key Drivers Behind This Risk")
    if len(top_risk_factors) > 0:
        st.write("Top factors increasing churn probability for this customer:")
        for feat, val in top_risk_factors.head(4).items():
            raw_val = raw_inputs[feat]
            st.markdown(f"• **{feat}** (Current Value: `{raw_val}`)")
    else:
        st.write("No severe risk factors detected for this customer profile.")

st.divider()

# 6. Retention Action Recommendations
st.subheader("📋 Recommended Retention Actions")
recs = []

if complain == 1:
    recs.append("🚨 **Unresolved Complaint:** Escalate open issue to priority support immediately and offer goodwill store credit.")
if satisfaction_score <= 2:
    recs.append("⭐ **Low Satisfaction Score:** Trigger automated feedback outreach with direct support follow-up.")
if days_since_last_order > 14:
    recs.append(f"🛒 **Inactivity Alert:** Send a personalized re-engagement campaign with a discount code for `{preferred_order_cat}`.")
if tenure <= 3:
    recs.append("🆕 **Early-Stage Risk:** Enroll customer in early-stage onboarding flow with special welcome perks.")
if cashback_amount < 100:
    recs.append("💰 **Incentive Push:** Offer temporary cashback boost to encourage higher purchase frequency.")

if not recs:
    recs.append("✅ **Healthy Profile:** Customer displays positive retention indicators. Maintain normal engagement.")

for rec in recs:
    st.info(rec)

# 7. Collapsible Advanced SHAP Visualizer
with st.expander("🔍 View Technical SHAP Breakdown (For ML Engineers)", expanded=False):
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_vals[0], show=False)
    st.pyplot(fig)
