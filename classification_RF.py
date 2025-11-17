import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("dummy_claims_with_fraud_label.xlsx")

features = [
    'age','NIK_valid','biometric_flag','num_diagnoses','num_procedures',
    'length_of_stay','total_claim_amount','diagnosis_cost_ratio',
    'avg_cost_per_procedure','verification_delay_days',
    'duplicate_ID_count','duplicate_ID_count_month','time_between_admissions',
    'num_claims_last_30d_by_provider','num_unique_patients_last_30d',
    'provider_claim_rate_vs_peer','claim_rejection_rate_provider',
    'month_over_month_claim_growth','sudden_spike_flag',
    'avg_claim_per_patient','claim_fragmentation_score','service_mix_index'
]

X = df[features]
y = df['fraud_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === SMOTE (oversample kelas HIGH)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# === Random Forest dengan class_weight
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight={'HIGH': 4, 'MEDIUM': 1.5, 'NORMAL': 1},
    random_state=42
)

model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test)

print("\n=== Classification Report (IMPROVED) ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
