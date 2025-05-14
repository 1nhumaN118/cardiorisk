#  CardioRisk AI App

CardioRisk is an end-to-end Machine Learning app to estimate the 10-year cardiovascular risk using basic clinical features (e.g., age, blood pressure, cholesterol). It integrates a trained XGBoost model with a simple Streamlit user interface, supports FHIR-formatted data, and offers both visual explanations and Docker-based deployment.

---

##  Project Structure

```
cardiorisk_app/
├── app.py                     # Streamlit app (renamed from app_cardiorisk.py)
├── Dockerfile                 # Docker build file
├── requirements.txt           # Python dependencies (for Hugging Face or local use)
├── src/
│   └── models/
│       ├── tuned_xgb_fhir.pkl          # Trained XGBoost model
│       └── fhir_feature_order.pkl      # Ordered list of input features
├── notebooks/
│   ├── cardiorisk_baseline.ipynb       # (1) Initial risk model with Framingham data
│   ├── tuning.ipynb            		  # (2) Hyperparameter tuning using Optuna
│   ├── fhir_data.ipynb                 # (3) Generate FHIR Observation data (synthetic)
│   └── fhir_inference_visual.ipynb     # (4) Load FHIR + Predict + Visualize importance
```

---

##  Main App Features

- Input clinical values via Streamlit interface
- Predict 10-year heart disease risk using trained XGBoost model
- Display risk probability + bar chart of feature importance
- Docker-compatible & deployable to Hugging Face Spaces
- Process input manually or from FHIR JSON observations

---

##  Core Functions

| Function | Description |
|---------|-------------|
| `predict_risk()` | Load model and infer from manual inputs |
| `get_feature_importance()` | Extract gain-based feature impact |
| `process_fhir_json()` | (In notebooks) Convert FHIR observations to feature table |
| `tuning_optuna()` | (Notebook) Bayesian optimization for best model |
| `explain_model()` | SHAP or gain-based importance (fallback if SHAP not available) |

---

##  How to Run (Locally)

1. Clone or unzip this project.
2. Build Docker image:
   ```bash
   docker build -t cardiorisk-app .
   ```
3. Run the app:
   ```bash
   docker run -p 8501:8501 cardiorisk-app
   ```
4. Open browser at [http://localhost:8501](http://localhost:8501)

---

##  Version Tracking

| Version | Date | Notes |
|---------|------|-------|
| v0.1.0  | 2025-05-13 | First public demo version with full ML pipeline and UI |
| v0.2.0  | _Planned_ | Add support for FHIR API ingestion and explainability UI |
| v1.0.0  | _Future_  | Include HL7/FHIR compliant API endpoints and integration with hospitals |

---

##  Contact

Built by a medical data science specialist aiming to integrate AI with real-world clinical workflows. Suitable for research, demo, or MedTech applications.

