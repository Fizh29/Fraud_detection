import pandas as pd
import numpy as np

df = pd.read_excel("dummy_claims_with_features.xlsx")
diagnosis_groups = {
    "ISPA": ['J00', 'J18.9', 'J45.9'],
    "Metabolic": ['E11', 'E66.9'],
    "Cardio": ['I10', 'I25.1'],
    "Gastro": ['K29.7', 'K21.0', 'A09'],
    "Urinary": ['N39.0'],
    "Musculoskeletal": ['M54.5'],
    "Eye": ['H25.9'],
    "Obstetri": ['O80', 'Z34.0'],
    "General": ['R50.9', 'Z03.8', 'Z00.0', 'R10.4']
}

# === Fraud Rule Scoring ===
df['fraud_score'] = 0  # <--- HARUS ADA

df['fraud_score'] += np.where(df['duplicate_ID_count'] > 2, 10, 0)
df['fraud_score'] += np.where(df['duplicate_ID_count_month'] > 1, 15, 0)
df['fraud_score'] += np.where(df['time_between_admissions'] < 5, 10, 0)
df['fraud_score'] += np.where(df['diagnosis_cost_ratio'] > 2, 17, 0)
df['fraud_score'] += np.where(df['verification_delay_days'] > 30, 10, 0)
df['fraud_score'] += np.where(df['month_over_month_claim_growth'] > 1.5, 10, 0)
df['fraud_score'] += np.where(df['sudden_spike_flag'] == 1, 14, 0)
df['fraud_score'] += np.where(df['avg_claim_per_patient'] > 5_000_000, 20, 0)
df['fraud_score'] += np.where(df['claim_fragmentation_score'] >= 2, 12, 0)
df['fraud_score'] += np.where(df['service_mix_index'] < 0.6, 10, 0)
df['fraud_score'] += np.where(df['NIK_valid'] == 0, 15, 0)
df['fraud_score'] += np.where(df['biometric_flag'] == 0, 25, 0)
df['fraud_score'] += np.where(df['claim_date'].dt.weekday == 6, 25, 0)
# Map LOS distribusi normal per group
los_normal_map = {
    "ISPA": (0, 4), "Metabolic": (0, 6), "Cardio": (0, 7), "Gastro": (0, 4),
    "Urinary": (0, 5), "Musculoskeletal": (0, 2), "Eye": (0, 1), "Obstetri": (1, 4),
    "General": (0, 2)
}
def get_diag_group(code):
        for group, codes in diagnosis_groups.items():
            if code in codes:
                return group
        return "General"
def los_anomaly_flag(row):
    group = get_diag_group(row['diagnosis_code'])
    min_los, max_los = los_normal_map.get(group, (0,2))
    return 1 if row['length_of_stay'] < min_los or row['length_of_stay'] > max_los else 0

df['los_anomaly_flag'] = df.apply(los_anomaly_flag, axis=1)
df['fraud_score'] += df['los_anomaly_flag'] * 25  # misal +15 poin

def provider_balance_score(nik):
    providers = df.loc[df['NIK'] == nik, 'provider_id']
    counts = providers.value_counts()
    if len(counts) == 1:
        return 0
    total = counts.sum()
    top_ratio = counts.max() / total
    # Lebih seimbang → lebih aneh
    if top_ratio <= 0.6:
        return 20
    elif top_ratio <= 0.8:
        return 8
    else:
        return 1
df['provider_balance_score'] = df['NIK'].map(provider_balance_score)
df['fraud_score'] += df['provider_balance_score']


# Cap maksimal skor di 100
df['fraud_score'] = df['fraud_score'].clip(upper=100)  
# === Assign Fraud Label berdasarkan skor ===


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
