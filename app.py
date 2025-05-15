import streamlit as st
import xgboost as xgb
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.special import expit  # Sigmoid function

import os
print(os.getcwd())
# Load model and features
MODEL_PATH = os.path.join("src", "models", "tuned_xgb_fhir.pkl")
FEATURE_PATH = os.path.join("src", "models", "fhir_feature_order.pkl")

booster = joblib.load(MODEL_PATH)
feature_order = joblib.load(FEATURE_PATH)
imputer = SimpleImputer(strategy="mean")

st.set_page_config(page_title="CardioRisk Estimator", layout="centered")
st.title("CardioRisk Estimator")
st.markdown("Predict the Cardio-related risks in 10 years based on clinical biomarkers")

# Input fields
input_data = {}
for feat in feature_order:
    if feat == "age":
        input_data[feat] = st.number_input("Age", 18, 120, 55)
    elif "sysBP" in feat:
        input_data[feat] = st.number_input("Systolic Blood Pressure", 50, 250, 120)
    elif "diaBP" in feat:
        input_data[feat] = st.number_input("Diastolic Blood Pressure", 50, 250, 120)
    elif "chol" in feat.lower():
        input_data[feat] = st.number_input("Total Cholesterol", 100, 400, 200)
    else:
        input_data[feat] = st.number_input("Glucose", 30, 300, 100)

# Predict
if st.button("Predict"):
    df_input = pd.DataFrame([input_data])[feature_order]
    df_input = pd.DataFrame(imputer.fit_transform(df_input), columns=feature_order)
    dmat = xgb.DMatrix(df_input, feature_names=feature_order)
    pred = booster.predict(dmat)[0]
    st.metric(label="Cardio Risk Probability", value=f"{pred:.2%}")

    try:
        explainer = shap.Explainer(booster)
        shap_values = explainer(df_input)
        shap_vals = explainer.shap_values(df_input)
        expected_log_odds = explainer.expected_value
        expected_prob = expit(expected_log_odds)
        prob_contributions = []

        # Approximate marginal contributions (non-linear mapping)
        for i, feat in enumerate(feature_order):
            partial_log_odds = expected_log_odds + shap_vals[0][i]
            partial_prob = expit(partial_log_odds)
            delta_prob = partial_prob - expected_prob
            prob_contributions.append((feat, delta_prob * 100))

        df_shap = pd.DataFrame(prob_contributions, columns=["Feature", "Contribution (%)"])
        df_shap = df_shap.sort_values("Contribution (%)", ascending=True).set_index("Feature")

        fig, ax = plt.subplots(figsize=(7, 4))
        df_shap.plot(kind="barh", ax=ax, legend=False, color="teal")
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_title("Biomarkers attribution (%)")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Không thể tính SHAP: {e}")