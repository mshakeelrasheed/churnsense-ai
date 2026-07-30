import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

st.set_page_config(page_title="ChurnSense AI", page_icon="🛒", layout="wide")

CATEGORICAL_COLS = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']

# Best-guess column order: dataset order minus CustomerID and Churn.
# If you still get a ValueError, paste it back — the message reveals the real order and we fix it in one shot.
FEATURE_ORDER = [
    'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
    'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
    'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount'
]

# --- 1. Load the Machine Learning Artifacts (cached so they load only once) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('xgboost_model.pkl')   # rename to match your actual filename
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

# --- 2. Prediction and Explanation Function ---
def predict_and_explain(inputs: dict):
    input_data = pd.DataFrame([inputs])[FEATURE_ORDER]

    for col in CATEGORICAL_COLS:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    scaled_data = scaler.transform(input_data)
    scaled_df = pd.DataFrame(scaled_data, columns=input_data.columns)

    churn_probability = model.predict_proba(scaled_df)[0][1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(scaled_df)

    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Why did the model make this prediction?")
    plt.tight_layout()

    return churn_probability, fig

# --- 3. Dashboard Layout ---
st.title("🛒 ChurnSense AI")
st.markdown("Explainable churn prediction for e-commerce customers — powered by XGBoost + SHAP.")

with st.sidebar:
    st.header("⚙️ Customer Profile")

    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    number_of_address = st.number_input("Number of Addresses", min_value=0, value=2, step=1)

    st.subheader("App Engagement")
    preferred_login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Phone", "Computer"])
    hour_spend_on_app = st.slider("Hours Spent on App", 0.0, 10.0, 3.0, 0.5)
    number_of_devices = st.number_input("Number of Devices Registered", min_value=1, value=3, step=1)
    warehouse_to_home = st.number_input("Warehouse to Home Distance (km)", min_value=0.0, value=15.0)

    st.subheader("Orders & Payment")
    preferred_payment = st.selectbox(
        "Preferred Payment Mode",
        ["Debit Card", "Credit Card", "UPI", "CC", "Cash on Delivery", "COD", "E wallet"]
    )
    prefered_order_cat = st.selectbox(
        "Preferred Order Category",
        ["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion", "Grocery", "Others"]
    )
    order_count = st.number_input("Total Order Count", min_value=0.0, value=3.0)
    day_since_last_order = st.number_input("Days Since Last Order", min_value=0.0, value=5.0)
    coupon_used = st.number_input("Coupons Used", min_value=0.0, value=1.0)
    order_amount_hike = st.number_input("Order Amount Hike From Last Year (%)", min_value=0.0, value=15.0)
    cashback_amount = st.number_input("Cashback Amount", min_value=0.0, value=150.0)

    st.subheader("Satisfaction & Tenure")
    tenure = st.slider("Tenure (Months)", 0, 61, 12)
    satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
    complain = st.selectbox("Has Complained Recently?", ["No", "Yes"])

    predict_btn = st.button("🔮 Predict Churn & Explain", type="primary", use_container_width=True)

if predict_btn:
    inputs = {
        'Tenure': tenure,
        'PreferredLoginDevice': preferred_login_device,
        'CityTier': city_tier,
        'WarehouseToHome': warehouse_to_home,
        'PreferredPaymentMode': preferred_payment,
        'Gender': gender,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': number_of_devices,
        'PreferedOrderCat': prefered_order_cat,
        'SatisfactionScore': satisfaction_score,
        'MaritalStatus': marital_status,
        'NumberOfAddress': number_of_address,
        'Complain': 1 if complain == "Yes" else 0,
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': day_since_last_order,
        'CashbackAmount': cashback_amount,
    }

    try:
        churn_probability, fig = predict_and_explain(inputs)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📊 Prediction")
            if churn_probability > 0.5:
                st.error(f"### ⚠️ High Risk of Churn\n**Probability:** {churn_probability:.1%}")
            else:
                st.success(f"### ✅ Customer Likely to Stay\n**Probability:** {churn_probability:.1%}")

        with col2:
            st.subheader("🔍 Why This Prediction? (SHAP)")
            st.pyplot(fig)

    except ValueError as e:
        st.error("Feature mismatch between the form and the trained model.")
        st.code(str(e))
        st.info("Copy this exact error and send it back — it reveals the real column order needed.")
else:
    st.info("Fill in the customer profile on the left and click **Predict Churn & Explain**.")
