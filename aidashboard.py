
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google import generativeai as genai

# ===================== SETUP AI =====================
genai.configure(api_key="AIzaSyDWASI1ydh57y3BA415_wXVEsjJzM-Mavs")
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# Load dataset
df = pd.read_csv("Data_Claims_classification.csv", dtype={"NIK": str})

# Pastikan kolom tanggal dalam datetime
df['service_date'] = pd.to_datetime(df['service_date'])
df['claim_date'] = pd.to_datetime(df['claim_date'])

# Kolom tambahan
df['year'] = df['claim_date'].dt.year
df['month'] = df['claim_date'].dt.month
df['year_month'] = df['claim_date'].dt.to_period('M')
# 1. Total klaim per provider
claims_per_provider = df.groupby('provider_id').size().sort_values(ascending=False)
claims_per_provider.plot(kind='bar')
plt.title('Total Klaim per Provider')
plt.ylabel('Jumlah Klaim')
plt.xlabel('Provider ID')
plt.savefig("total_claims_per_provider.png", dpi=300, bbox_inches='tight')
plt.clf()

# 2. Gender distribution
df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribusi Gender Pasien')
plt.ylabel('')
plt.savefig("gender_distribution.png", dpi=300, bbox_inches='tight')
plt.clf()

# 3. Top 10 diagnosis
top_diag_overall = df['diagnosis_code'].value_counts().nlargest(10)
top_diag_overall.plot(kind='bar')
plt.title('Top 10 Diagnosis')
plt.ylabel('Jumlah Klaim')
plt.xlabel('Kode Diagnosis')
plt.savefig("top_10_diagnosis.png", dpi=300, bbox_inches='tight')
plt.clf()

# 4. Klaim bulanan
monthly_claims = df.groupby('year_month').size()
monthly_claims.plot(kind='line', marker='o')
plt.title('Total Klaim per Bulan')
plt.ylabel('Jumlah Klaim')
plt.xlabel('Bulan')
plt.xticks(rotation=45)
plt.savefig("monthly_claims_trend.png", dpi=300, bbox_inches='tight')
plt.clf()

# 5. Total claim amount per provider
claims_amount = df.groupby('provider_id')['total_claim_amount'].sum().sort_values(ascending=False)
claims_amount.plot(kind='bar')
plt.title('Total Claim Amount per Provider')
plt.ylabel('Jumlah Rupiah')
plt.xlabel('Provider ID')
plt.savefig("total_claim_amount_per_provider.png", dpi=300, bbox_inches='tight')
plt.clf()

# 6. Distribusi fraud label
df['fraud_label'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribusi Fraud Label')
plt.ylabel('')
plt.savefig("fraud_label_distribution.png", dpi=300, bbox_inches='tight')
plt.clf()

# 7. Diagnosis vs Fraud Level

fraud_diag = df.groupby(['diagnosis_code','fraud_label']).size().unstack(fill_value=0)
top_diag_fraud = fraud_diag.sum(axis=1).nlargest(10).index
fraud_diag.loc[top_diag_fraud].plot(kind='bar', stacked=True)
plt.title('Top 10 Diagnosis Berdasarkan Fraud Level')
plt.savefig("top_10_diagnosis_by_fraud_level.png", dpi=300, bbox_inches='tight')
plt.clf()


# 8. Diagnosis High Risk
high_diag = df[df['fraud_label']=='HIGH']['diagnosis_code'].value_counts().nlargest(10)
if not high_diag.empty:
    high_diag.plot(kind='bar')
    plt.title('Top 10 Diagnosis High Risk')
    plt.ylabel('Jumlah Klaim High')
    plt.xlabel('Kode Diagnosis')
    plt.savefig("top_high_risk_diagnosis.png", dpi=300, bbox_inches='tight')
    plt.clf()

# 9. Fraud per provider
fraud_provider = df.groupby(['provider_id','fraud_label']).size().unstack(fill_value=0)
fraud_provider.plot(kind='bar', stacked=True)
plt.title('Jumlah Fraud per Provider')
plt.ylabel('Jumlah Klaim')
plt.xlabel('Provider ID')
plt.savefig("fraud_per_provider.png", dpi=300, bbox_inches='tight')
plt.clf()

# 10. Age group analysis
df['age_group'] = pd.cut(df['age'], bins=[0,20,40,60,90], labels=['0-20','21-40','41-60','61+'])
df.groupby('age_group').size().plot(kind='bar')
plt.title('Jumlah Klaim per Age Group')
plt.ylabel('Jumlah Klaim')
plt.xlabel('Age Group')
plt.savefig("claims_by_age_group.png", dpi=300, bbox_inches='tight')
plt.clf()

# 11. Scatterplot LOS vs claim amount
sns.scatterplot(x='length_of_stay', y='total_claim_amount', hue='fraud_label', data=df)
plt.title('LOS vs Total Claim Amount')
plt.savefig("los_vs_claim_amount.png", dpi=300, bbox_inches='tight')
plt.clf()


# ===================== AI SUMMARY =====================
# Kita bikin summary untuk seluruh dataset / dashboard
dashboard_summary_prompt = f"""
Berikan analisis profesional dan ringkasan insight dari dataset klaim asuransi berikut:
Berikut adalah seluruh ringkasan data yang digunakan untuk 11 grafik:

1. Total klaim per provider: {claims_per_provider.to_dict()}
2. Distribusi gender: {df['gender'].value_counts().to_dict()}
3. Top 10 diagnosis: {top_diag_overall.to_dict()}
4. Klaim bulanan: {monthly_claims.to_dict()}
5. Total claim amount per provider: {claims_amount.to_dict()}
6. Distribusi fraud label: {df['fraud_label'].value_counts().to_dict()}
7. Diagnosis per fraud label (stacked): (stacked): {fraud_diag.loc[top_diag_fraud].to_dict()}
8. Top diagnosis high risk: {high_diag.to_dict()}
9. Fraud per provider: {fraud_provider.to_dict()}
10. Distribusi umur: {df.groupby('age_group').size().to_dict()}
11. Scatter LOS vs Claim Amount:
    - korelasi_length_of_stay_claim_amount = {df['length_of_stay'].corr(df['total_claim_amount'])}

Jelaskan secara profesional:
1. Pola umum klaim dan pasien.
2. Insight fraud: kategori High/Medium/Normal, diagnosis yang paling rawan fraud, provider bermasalah.
3. Tren klaim bulanan dan tahunan.
4. Rekomendasi tindakan untuk auditor dan manajemen untuk mengurangi risiko fraud.

Tulis ringkasan padat, jelas, actionable, dan mudah dimengerti.
"""

    
try:
    resp = model.generate_content(dashboard_summary_prompt)

    with open("AI_SUMMARY_DASHBOARD.txt", "w", encoding="utf-8") as f:
        f.write(resp.text)

    print("✔ AI summary saved to: AI_SUMMARY_DASHBOARD.txt")

except Exception as e:
    print("⚠️ Gagal generate AI summary:", e)

