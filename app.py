# Gender -> 1 Female | 0 Male
# Churn  -> 1 Yes    | 0 No
# Order of X -> ['Age', 'Gender', 'Tenure', 'MonthlyCharges']

import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# UI
st.title("📊 Customer Churn Prediction App")
st.divider()
st.write("Please enter the values and click **Predict** to get a churn prediction.")
st.divider()

# Inputs
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure (months)", min_value=0, max_value=130, value=10)
monthly_charge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150, value=50)

gender = st.selectbox("Select Gender", ["Male", "Female"])

st.divider()
predict_button = st.button("Predict")

# Prediction
if predict_button:
    gender_encoded = 1 if gender == "Female" else 0

    # 2D array: (1 sample, 4 features)
    X = np.array([[age, gender_encoded, tenure, monthly_charge]])

    # Scaling
    X_scaled = scaler.transform(X)

    # SVC decision score
    decision_score = model.decision_function(X_scaled)[0]

    # Convert score to probability-like value (sigmoid)
    churn_score = 1 / (1 + np.exp(-decision_score))

    # Threshold
    threshold = 0.8
    prediction = "Yes" if churn_score >= threshold else "No"

    # Display results
    st.subheader("📌 Prediction Result")
    st.write(f"**Churn Score:** {churn_score:.2%}")

    if prediction == "Yes":
        st.error(" Customer is likely to CHURN")
    else:
        st.success(" Customer is NOT likely to churn")

else:
    st.info("Please enter the values and click the **Predict** button.")
