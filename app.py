import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# --- TITLE ---
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)

# --- ACCURACY ---
st.subheader("📈 Model Performance")
st.info(f"Model Accuracy: {accuracy*100:.2f}%")

st.markdown("---")

# --- INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", value=50.0)

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer", "Credit card"
    ])
    total = st.number_input("Total Charges", value=500.0)

# --- ENCODING ---
def encode():
    return [
        0 if gender == "Female" else 1,
        1 if senior == "Yes" else 0,
        0, 0,  # Partner, Dependents default
        tenure,
        1, 0, 2,
        1, 1, 1, 1,
        1, 1,
        ["Month-to-month", "One year", "Two year"].index(contract),
        1,
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"].index(payment),
        monthly,
        total
    ]

# --- PREDICTION ---
if st.button("🚀 Predict Now"):

    data = np.array(encode()).reshape(1, -1)
    data = scaler.transform(data)

    pred = model.predict(data)
    prob = model.predict_proba(data)

    churn_prob = prob[0][1] * 100

    st.markdown("---")

    # RESULT
    if pred[0] == 1:
        st.error(f"⚠️ High Risk Customer ({churn_prob:.2f}%)")
    else:
        st.success(f"✅ Low Risk Customer ({churn_prob:.2f}%)")

    # RISK METER
    st.subheader("🎯 Customer Risk Meter")
    st.progress(int(churn_prob))

    # PIE CHART
    st.subheader("📊 Churn Probability Distribution")

    labels = ['Not Churn', 'Churn']
    values = [prob[0][0], prob[0][1]]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig)

    # FEATURE IMPORTANCE
    st.subheader("📊 Feature Importance")

    feature_names = [
        "Gender","SeniorCitizen","Partner","Dependents","Tenure",
        "PhoneService","MultipleLines","InternetService","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges"
    ]

    importance = model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(df_imp.set_index("Feature"))

    # --- TOP 3 REASONS (HUMAN FRIENDLY) ---
    st.subheader("🧠 Why This Customer May Churn")

    reasons = df_imp.head(3)["Feature"].tolist()

    for reason in reasons:
        if reason == "Tenure":
            st.write("🔹 Low customer tenure increases churn risk")
        elif reason == "Contract":
            st.write("🔹 Short-term contract increases churn risk")
        elif reason == "MonthlyCharges":
            st.write("🔹 High monthly charges increase churn risk")
        elif reason == "InternetService":
            st.write("🔹 Internet service type affects churn behavior")
        else:
            st.write(f"🔹 {reason} strongly influences churn")