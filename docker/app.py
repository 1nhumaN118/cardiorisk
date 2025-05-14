import streamlit as st
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

# Load model and features
MODEL_PATH = "src/models/tuned_xgb_fhir.pkl"
FEATURE_PATH = "src/models/fhir_feature_order.pkl"

booster = joblib.load(MODEL_PATH)
feature_order = joblib.load(FEATURE_PATH)
imputer = SimpleImputer(strategy="mean")

st.set_page_config(page_title="CardioRisk Estimator", layout="centered")
st.title("🫀 CardioRisk Estimator")
st.markdown("Dự đoán nguy cơ mắc bệnh tim mạch trong 10 năm dựa trên chỉ số sinh học lâm sàng.")

# Input fields
input_data = {}
for feat in feature_order:
    if feat == "age":
        input_data[feat] = st.number_input("Tuổi", 18, 120, 55)
    elif "BP" in feat:
        input_data[feat] = st.number_input(feat, 50, 250, 120)
    elif "chol" in feat.lower():
        input_data[feat] = st.number_input(feat, 100, 400, 200)
    else:
        input_data[feat] = st.number_input(feat, 30, 300, 100)

# Predict
if st.button("🩺 Dự đoán nguy cơ"):
    df_input = pd.DataFrame([input_data])[feature_order]
    df_input = pd.DataFrame(imputer.fit_transform(df_input), columns=feature_order)
    dmat = xgb.DMatrix(df_input, feature_names=feature_order)
    pred = booster.predict(dmat)[0]
    st.metric(label="Xác suất nguy cơ bệnh tim", value=f"{pred:.2%}")

    # Optional: Display bar chart of user inputs and static feature importance
    st.subheader("🧬 Các chỉ số đã nhập và tầm quan trọng")
    gain_importance = booster.get_score(importance_type="gain")
    importances = [gain_importance.get(f, 0) for f in feature_order]

    chart_data = pd.DataFrame({
        "Feature": feature_order,
        "Giá trị đã nhập": [input_data[f] for f in feature_order],
        "Tầm quan trọng": importances
    }).set_index("Feature")

    st.bar_chart(chart_data["Tầm quan trọng"])
    st.dataframe(chart_data)
