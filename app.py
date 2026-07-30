import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

st.set_page_config(page_title="ChurnSense AI", page_icon="🛒", layout="wide")

# --- 1. Load the Machine Learning Artifacts (cached so they load only once) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('xgboost_model.pkl')   # rename to match your actual filename
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')      # Dictionary of LabelEncoders
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

# --- 2. Prediction and Explanation Function ---
def predict_and_explain(tenure, satisfaction, order_count, membership_category):
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'satisfaction_score': [satisfaction],
        'total_orders': [order_count],
        'membership_level': [membership_category]
    })

    for col in ['membership_level']:
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
st.title("🛒 Explainable Churn & Retention for E-Commerce")
st.markdown("Analyze customer retention risk and understand the driving factors behind every prediction using SHAP (Explainable AI).")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Customer Profile")
    tenure_input = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
    satisfaction_input = st.slider("Satisfaction Score", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
    order_count_input = st.number_input("Total Order Count", value=5, step=1)
    category_input = st.selectbox("Membership Level", ["Basic", "Premium", "VIP"])
    predict_btn = st.button("Predict Churn & Generate Explanation", type="primary")

with col2:
    st.subheader("📊 AI Analysis")
    if predict_btn:
        churn_probability, fig = predict_and_explain(
            tenure_input, satisfaction_input, order_count_input, category_input
        )

        if churn_probability > 0.5:
            st.error(f"### ⚠️ High Risk of Churn\n**Probability:** {churn_probability:.1%}")
        else:
            st.success(f"### ✅ Customer is Likely to Stay\n**Probability:** {churn_probability:.1%}")

        st.subheader("🔍 SHAP Feature Impact")
        st.pyplot(fig)
    else:
        st.info("Fill in the customer profile and click Predict to see the analysis.")
