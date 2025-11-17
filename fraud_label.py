import pandas as pd
import numpy as np

df = pd.read_excel("dummy_claims_with_features.xlsx")

# === Fraud Rule Scoring ===
df['fraud_score'] = 0
df['fraud_score'] += np.where(df['duplicate_ID_count'] > 2, 20, 0)
df['fraud_score'] += np.where(df['duplicate_ID_count_month'] > 1, 15, 0)
df['fraud_score'] += np.where(df['time_between_admissions'] < 5, 15, 0)
df['fraud_score'] += np.where(df['diagnosis_cost_ratio'] > 2, 10, 0)
df['fraud_score'] += np.where(df['verification_delay_days'] > 30, 10, 0)
df['fraud_score'] += np.where(df['month_over_month_claim_growth'] > 1.5, 10, 0)
df['fraud_score'] += np.where(df['sudden_spike_flag'] == 1, 20, 0)
df['fraud_score'] += np.where(df['avg_claim_per_patient'] > 5_000_000, 10, 0)
df['fraud_score'] += np.where(df['claim_fragmentation_score'] >= 2, 20, 0)
df['fraud_score'] += np.where(df['service_mix_index'] < 0.6, 10, 0)
df['fraud_score'] += np.where(df['NIK_valid'] == 0, 10, 0)
df['fraud_score'] += np.where(df['biometric_flag'] == 0, 10, 0)

def assign_label(score):
    if score >= 60:
        return "HIGH"
    elif score >= 35:
        return "MEDIUM"
    else:
        return "NORMAL"

df['fraud_label'] = df['fraud_score'].apply(assign_label)

# === SPLIT 5030 → 5000 (training) + 30 (new) ===
training_df = df.iloc[:5000].copy()
new_claims_df = df.iloc[5000:].copy()

training_df.to_excel("dummy_claims_with_fraud_label.xlsx", index=False)
new_claims_df.to_excel("new_claims_30.xlsx", index=False)

print("✔ Fraud scoring completed")
print("✔ 5000 training claims saved!")
print("✔ 30 new claims saved as new_claims_30.xlsx")
