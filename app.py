
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Load model ---
model = joblib.load('insurance_model.pkl')

# --- Page config ---
st.set_page_config(
    page_title="Medical Insurance Cost Estimator",
    page_icon="🏥",
    layout="centered"
)

# --- Header ---
st.title("🏥 Medical Insurance Cost Estimator")
st.markdown("### Powered by Random Forest ML Model")
st.markdown(
    "Enter your details below to get an instant estimate "
    "of your annual medical insurance charges."
)
st.markdown("---")

# --- Input section ---
st.subheader("📋 Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=10.0,
                          max_value=60.0, value=25.0, step=0.1)
    children = st.slider("Number of Children",
                         min_value=0, max_value=5, value=0)

with col2:
    sex = st.radio("Sex", options=["female", "male"])
    smoker = st.radio("Smoker?", options=["no", "yes"])
    region = st.selectbox(
        "Region", options=["southwest", "southeast", "northwest", "northeast"])

st.markdown("---")

# --- Predict button ---
if st.button("Calculate My Estimate"):

    # --- Encode inputs exactly like training ---
    sex_encoded = 1 if sex == "male" else 0
    smoker_encoded = 1 if smoker == "yes" else 0

    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    # --- Build input dataframe ---
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex_encoded,
        'bmi': bmi,
        'children': children,
        'smoker': smoker_encoded,
        'region_northwest': region_northwest,
        'region_southeast': region_southeast,
        'region_southwest': region_southwest
    }])

    # --- Predict and reverse log ---
    log_prediction = model.predict(input_data)
    prediction = np.exp(log_prediction)[0]

    # --- Display result ---
    st.success(f"💰 Estimated Annual Insurance Cost: **${prediction:,.2f}**")

    # --- Risk breakdown ---
    st.markdown("#### 📊 Your Risk Profile")
    col3, col4, col5 = st.columns(3)

    with col3:
        smoker_risk = "🔴 High Risk" if smoker == "yes" else "🟢 Low Risk"
        st.metric("Smoking Status", smoker_risk)

    with col4:
        bmi_risk = "🔴 Obese" if bmi >= 30 else "🟡 Overweight" if bmi >= 25 else "🟢 Normal"
        st.metric("BMI Category", bmi_risk)

    with col5:
        age_risk = "🔴 Senior" if age >= 55 else "🟡 Middle" if age >= 35 else "🟢 Young"
        st.metric("Age Group", age_risk)

# --- Footer ---
st.markdown("---")
st.caption(
    "Model: Random Forest (100 trees) | R² = 0.896 | "
    "RMSE = $4,367 | Trained on 1,337 insurance records"
)

