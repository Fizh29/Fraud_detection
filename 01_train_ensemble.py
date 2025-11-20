import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_excel("dummy_claims_with_fraud_label.xlsx")

X = df.drop(columns=["fraud_label"])
y = df["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss"
)

model = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)

model.fit(X_train, y_train)

joblib.dump(model, "model_ensemble.pkl")
print("✔ Ensemble model saved as model_ensemble.pkl")
