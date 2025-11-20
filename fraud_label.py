import pandas as pd
import numpy as np

df = pd.read_csv("dummy_claims_with_features.csv", dtype={"NIK": str})

# Convert tanggal → datetime wajib
df['claim_date'] = pd.to_datetime(df['claim_date'])
df['service_date'] = pd.to_datetime(df['service_date'])
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

df['provider_id_default'] = df.groupby('NIK')['provider_id'].transform('first')

# ==========================================================
#                HARD TUNING FRAUD SCORING
# ==========================================================

df['fraud_score'] = 0

# === CRITICAL RULES (12–20) ===
df['fraud_score'] += np.where(df['biometric_flag'] == 0, 20, 0)
df['fraud_score'] += np.where(df['NIK_valid'] == 0, 15, 0)
df['fraud_score'] += np.where(df['duplicate_ID_count_month'] > 1, 14, 0)
df['fraud_score'] += np.where(df['diagnosis_cost_ratio'] > 3.0, 14, 0)
df['fraud_score'] += np.where(df['total_claim_amount'] > df['tarif_standar_diagnosis'] * 2.5, 16, 0)

# === MAJOR RULES (6–10) ===
df['fraud_score'] += np.where(df['duplicate_ID_count'] > 2, 8, 0)
df['fraud_score'] += np.where(df['claim_fragmentation_score'] >= 2, 10, 0)
df['fraud_score'] += np.where(df['diagnosis_cost_ratio'] > 2.0, 9, 0)
df['fraud_score'] += np.where(df['avg_claim_per_patient'] > 5_000_000, 10, 0)
df['fraud_score'] += np.where(df['month_over_month_claim_growth'] > 1.5, 8, 0)
df['fraud_score'] += np.where(df['service_mix_index'] < 0.55, 8, 0)

# Provider anomaly
df['fraud_score'] += np.where(df['provider_id'] != df['provider_id_default'], 6, 0)

# === MINOR RULES (2–5) ===
df['fraud_score'] += np.where(df['time_between_admissions'] < 5, 5, 0)
df['fraud_score'] += np.where(
    (df['diagnosis_cost_ratio'] > 1.1) & (df['diagnosis_cost_ratio'] <= 2.0), 4, 0
)
df['fraud_score'] += np.where(
    (df['verification_delay_days'] > 30), 4, 0
)
df['fraud_score'] += np.where(df['claim_date'].dt.weekday == 6, 3, 0)
df['fraud_score'] += np.where(
    (df['claim_date'] - df['service_date']).dt.days <= 1, 3, 0
)
df['fraud_score'] += np.where(df['sudden_spike_flag'] == 1, 3, 0)

# ==========================================================
#              LOS anomaly (major category)
# ==========================================================

los_normal_map = {
    "ISPA": (0, 4), "Metabolic": (0, 6), "Cardio": (0, 7), "Gastro": (0, 4),
    "Urinary": (0, 5), "Musculoskeletal": (0, 2), "Eye": (0, 1),
    "Obstetri": (1, 4), "General": (0, 2)
}

def get_diag_group(code):
    for group, codes in diagnosis_groups.items():
        if code in codes:
            return group
    return "General"

def los_anomaly_flag(row):
    group = get_diag_group(row['diagnosis_code'])
    lo, hi = los_normal_map.get(group, (0, 3))
    return 1 if row['length_of_stay'] < lo or row['length_of_stay'] > hi else 0

df['los_anomaly_flag'] = df.apply(los_anomaly_flag, axis=1)
df['fraud_score'] += df['los_anomaly_flag'] * 10  # MAJOR

# ==========================================================
#        PROVIDER BALANCE SCORE (Critical but scaled)
# ==========================================================

def provider_balance_score(nik):
    providers = df.loc[df['NIK'] == nik, 'provider_id']
    counts = providers.value_counts()
    if len(counts) == 1:
        return 0
    total = counts.sum()
    top_ratio = counts.max() / total

    if top_ratio <= 0.55:
        return 12
    elif top_ratio <= 0.75:
        return 6
    else:
        return 2

df['provider_balance_score'] = df['NIK'].map(provider_balance_score)
df['fraud_score'] += df['provider_balance_score']

# ==========================================================
#               HARD TUNING NORMALIZATION
# ==========================================================

# Step 1: clip max to avoid outlier explosion
df['fraud_score'] = df['fraud_score'].clip(upper=100)

# Step 2: optional scaling → normalizing to 0–100
df['fraud_score'] = (df['fraud_score'] / df['fraud_score'].max()) * 100
df['fraud_score'] = df['fraud_score'].round(2)

# ==========================================================
#                   FINAL LABELING
# ==========================================================

def assign_label(score):
    if score >= 66:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "NORMAL"

df['fraud_label'] = df['fraud_score'].apply(assign_label)

# SPLIT DATA
training_df = df.iloc[:5000].copy()
new_claims_df = df.iloc[5000:].copy()

training_df.to_csv("dummy_claims_with_fraud_label.csv", index=False)
new_claims_df.to_csv("new_claims_30.csv", index=False)

print("✔ Hard tuning completed.")
print("✔ 5000 labeled claims saved.")
print("✔ 30 evaluation claims saved.")
