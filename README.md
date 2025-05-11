#  CardioRisk

A reproducible prototype to predict 10-year cardiovascular risk using structured health data (Framingham dataset), with transparent explainability using SHAP.

##  Purpose
This project aims to demonstrate:
- How structured medical data can be used to train a risk prediction model
- How to use **SHAP** to explain AI decisions in a clinical context
- How this prototype can be extended into real-world **FHIR**-based hospital systems

---

##  Dataset
**Framingham Heart Study Dataset**  
Source: [Kaggle – Framingham Risk Score](https://www.kaggle.com/datasets/imanmsharifi/framingham-heart-study-dataset)

Variables include: age, gender, cholesterol, glucose, hypertension, smoking status, and 10-year CHD outcome.

---

##  Features
-  Exploratory Data Analysis (EDA)
-  Train XGBoost classifier for risk prediction
-  Use **SHAP** to explain predictions at global and individual levels
-  Visualize SHAP summary and waterfall plots

---

##  Tech Stack
- `pandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `jupyter`

---

##  How to Run
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
