import pandas as pd
import numpy as np

# === 1️⃣ BACA DATA DUMMY HASIL GENERATOR ===
df = pd.read_excel('dummy_claims_2024_2025.xlsx')

# Pastikan format tanggal
df['claim_date'] = pd.to_datetime(df['claim_date'])
df['service_date'] = pd.to_datetime(df['service_date'])

# Urutkan data
df = df.sort_values(by=['provider_id', 'claim_date']).reset_index(drop=True)

# === 2️⃣ FITUR TURUNAN DASAR ===
df['avg_cost_per_procedure'] = df['total_claim_amount'] / df['num_procedures']
df['diagnosis_cost_ratio'] = df['total_claim_amount'] / df['tarif_standar_diagnosis']
df['verification_delay_days'] = (df['claim_date'] - df['service_date']).dt.days

# === 3️⃣ FITUR LANJUTAN ===

# --- 3.1 Duplicate_ID_count (TOTAL DUPLICATE)
nik_counts = df['NIK'].value_counts()
df['duplicate_ID_count'] = df['NIK'].map(lambda x: nik_counts[x] - 1)
df['duplicate_ID_count'] = df['duplicate_ID_count'].astype(int)

# --- 3.1b Duplicate_ID_count_month (ROLLING 30 DAYS)
def duplicate_last_30d(g):
    g = g.sort_values('claim_date')
    g['duplicate_ID_count_month'] = [
        g.loc[(g['claim_date'] <= d) &
              (g['claim_date'] > d - pd.Timedelta(days=30)), 'NIK'].value_counts().get(nik, 1) - 1
        for d, nik in zip(g['claim_date'], g['NIK'])
    ]
    return g

df = df.groupby('NIK', group_keys=False).apply(duplicate_last_30d)


# --- 3.2 Time_between_admissions
df['time_between_admissions'] = df.groupby('NIK')['claim_date'].diff().dt.days.fillna(0)

# --- 3.3 Num_claims_last_30d_by_provider (tanpa error monotonic)
def rolling_claims_count(g):
    g = g.sort_values('claim_date')
    g['num_claims_last_30d_by_provider'] = [
        ((g['claim_date'] <= d) & (g['claim_date'] > d - pd.Timedelta(days=30))).sum()
        for d in g['claim_date']
    ]
    return g

df = df.groupby('provider_id', group_keys=False).apply(rolling_claims_count)

# --- 3.4 Num_unique_patients_last_30d
def rolling_unique_patients(g):
    g = g.sort_values('claim_date')
    g['num_unique_patients_last_30d'] = [
        g.loc[(g['claim_date'] <= d) & (g['claim_date'] > d - pd.Timedelta(days=30)), 'NIK'].nunique()
        for d in g['claim_date']
    ]
    return g

df = df.groupby('provider_id', group_keys=False).apply(rolling_unique_patients)

# --- 3.5 Provider_claim_rate_vs_peer
provider_total = df.groupby('provider_id')['total_claim_amount'].sum()
peer_avg = provider_total.mean()
df['provider_claim_rate_vs_peer'] = df['provider_id'].map(lambda x: provider_total[x] / peer_avg)

# --- 3.6 Claim_rejection_rate_provider (simulasi)
np.random.seed(42)
df['claim_rejection_rate_provider'] = df['provider_id'].map(lambda x: np.random.uniform(0.02, 0.15))

# --- 3.7 Month_over_month_claim_growth + penanganan inf
df['year_month'] = df['claim_date'].dt.to_period('M')
monthly_claims = df.groupby(['provider_id', 'year_month']).size().unstack(fill_value=0)
mom_growth = monthly_claims.pct_change(axis=1)
mom_growth_long = mom_growth.stack().rename('month_over_month_claim_growth')

# Join ke df utama
df = df.join(mom_growth_long, on=['provider_id', 'year_month'])

# Tangani inf dan tambahkan flag lonjakan
df['sudden_spike_flag'] = np.where(np.isinf(df['month_over_month_claim_growth']), 1, 0)
df['month_over_month_claim_growth'].replace([np.inf, -np.inf], np.nan, inplace=True)
# FLAG provider baru muncul bulan itu
df['new_provider_month_flag'] = df['month_over_month_claim_growth'].isna().astype(int)

# Replace NaN MoM growth → 0 (stabil untuk klasifikasi)
df['month_over_month_claim_growth'] = df['month_over_month_claim_growth'].fillna(0)


# --- 3.8 Avg_claim_per_patient
df['avg_claim_per_patient'] = df['total_claim_amount'] / df['num_unique_patients_last_30d'].replace(0, np.nan)

# --- 3.9 Claim_fragmentation_score
df['claim_fragmentation'] = (
    df.groupby(['NIK', 'diagnosis_code'])['claim_date']
      .diff()
      .dt.days
      .apply(lambda x: 1 if pd.notna(x) and 0 < x <= 7 else 0)
      .fillna(0)
)
df['claim_fragmentation_score'] = (
    df.groupby(['NIK', 'diagnosis_code'])['claim_date']
      .diff().dt.days
      .apply(lambda x:
             3 if pd.notna(x) and 0 < x <= 3 else
             1 if pd.notna(x) and 3 < x <= 7 else
             0)
      .fillna(0)
)


# --- 3.10 Service_mix_index (entropi diagnosis per RS)
def service_mix_entropy(sub_df):
    p = sub_df['diagnosis_code'].value_counts(normalize=True)
    return -np.sum(p * np.log2(p))

entropy_map = df.groupby('provider_id').apply(service_mix_entropy)
df['service_mix_index'] = df['provider_id'].map(entropy_map)

# === 4️⃣ HAPUS KOLOM BANTUAN DAN SIMPAN ===
df = df.drop(columns=['days_diff'], errors='ignore')
df = df.sort_values(by='claim_date')
df.to_excel('dummy_claims_with_features.xlsx', index=False)

# === 5️⃣ TAMPILKAN CONTOH HASIL ===
cols_show = [
    'provider_id', 'NIK', 'duplicate_ID_count', 'time_between_admissions',
    'num_claims_last_30d_by_provider', 'num_unique_patients_last_30d',
    'provider_claim_rate_vs_peer', 'claim_rejection_rate_provider',
    'month_over_month_claim_growth', 'sudden_spike_flag',
    'avg_claim_per_patient', 'claim_fragmentation_score', 'service_mix_index'
]
print("\n✅ Fitur lanjutan berhasil dihitung tanpa error dan disimpan ke dummy_claims_with_features.xlsx")
print(df[cols_show].head(10))
