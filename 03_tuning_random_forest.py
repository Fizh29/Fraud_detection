from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

df = pd.read_excel("dummy_claims_with_fraud_label.xlsx")

X = df.drop(columns=["fraud_label"])
y = df["fraud_label"]

param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [8, 10, 12, 15],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3]
}

rf = RandomForestClassifier()

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='f1_macro',
    verbose=2,
    random_state=42
)

random_search.fit(X, y)

best_rf = random_search.best_estimator_

joblib.dump(best_rf, "model_rf_tuned.pkl")
print("✔ Tuned RF model saved as model_rf_tuned.pkl")
