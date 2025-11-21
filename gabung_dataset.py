import pandas as pd

# ===============================
# 1. LOAD DATA
# ===============================
print("Loading datasets...")
df_old = pd.read_csv("Data_Claims_classification.csv", dtype={"NIK": str})
df_new = pd.read_csv("new_claims.csv", dtype={"NIK": str})

# ===============================
# 2. CLEANING DUMMY OLD DATA
# ===============================
cols_to_drop_old = [
    "fraud_score",
    "los_anomaly_flag",
    "provider_balance_score"
]

df_old = df_old.drop(columns=[c for c in cols_to_drop_old if c in df_old.columns])


# ===============================
# 3. CLEANING NEW DATA
# ===============================
cols_to_drop_new = [
    "fraud_score",
    "fraud_label",
    "fraud_prob_HIGH",
    "los_anomaly_flag","provider_balance_score"
]

df_new = df_new.drop(columns=[c for c in cols_to_drop_new if c in df_new.columns])

# rename predicted_label → fraud_label
if "predicted_label" in df_new.columns:
    df_new = df_new.rename(columns={"predicted_label": "fraud_label"})


# ===============================
# 4. GABUNGKAN DATASET
# ===============================
df_combined = pd.concat([df_old, df_new], ignore_index=True)

print("Total rows after merge:", len(df_combined))


# ===============================
# 5. SAVE FINAL DATA
# ===============================
output_name = "Data_Claims_classification.csv"
df_combined.to_csv(output_name, index=False)

print(f"File berhasil dibuat: {output_name}")
print("✔ Dataset gabungan disimpan.")
