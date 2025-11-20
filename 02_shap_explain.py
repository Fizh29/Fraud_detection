import shap
import joblib
import pandas as pd
import numpy as np

model = joblib.load("model_ensemble.pkl")
df = pd.read_excel("dummy_claims_with_fraud_label.xlsx")
X = df.drop(columns=["fraud_label"])

explainer = shap.TreeExplainer(model.estimators_[0])  # pakai RF
shap_values = explainer.shap_values(X)

np.save("shap_values.npy", shap_values)
print("✔ SHAP values saved → shap_values.npy")
