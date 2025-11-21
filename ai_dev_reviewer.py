import pandas as pd
from google import generativeai as genai
import glob

# SETUP GEMINI
genai.configure(api_key="AIzaSyDWASI1ydh57y3BA415_wXVEsjJzM-Mavs")
model = genai.GenerativeModel("models/gemini-2.5-flash")

# ===================== 1. BACA DATASET =====================
df = pd.read_csv("Data_Claims_classification.csv", dtype={"NIK": str})

# Rapikan NIK
df["NIK"] = df["NIK"].str.replace(".0", "", regex=False).str.strip()

# Buat ringkasan dataset (biar AI bisa membaca)
dataset_summary = f"""
--- DATASET SUMMARY ---
Total Rows: {len(df)}
Total Pasien: {df['NIK'].nunique()}
Fraud Label Distribution: {df['fraud_label'].value_counts().to_dict()}
Columns: {list(df.columns)}

Head (5 rows):
{df.head().to_string()}

Describe Numerical:
{df.describe().to_string()}

Top Diagnosis (10):
{df['diagnosis_code'].value_counts().head(10).to_dict()}
"""

# ===================== 2. BACA SEMUA FILE PY =====================
files = glob.glob("**/*.py", recursive=True)

code_text = ""
for f in files:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
        code_text += f"\n\n===== FILE: {f} =====\n{content}\n\n"

# ===================== 3. COMBINE PROMPT =====================
prompt = f"""
Kamu adalah AI reviewer proyek fraud detection.

Analisis berikut ini mencakup:
1. Dataset klaim asuransi.
2. Seluruh file kode Python di project ini.

=== DATASET ===
{dataset_summary}

=== SOURCE CODE ===
{code_text}

Berikan analisis profesional yang mencakup:

1. Analisis kualitas data (missing value, distribusi, potensi bias).
2. Validitas fitur fraud_label & potensi masalah modeling.
3. Pola fraud yang terlihat dari ringkasan dataset.
4. Alur kerja project secara keseluruhan (dari data → fitur → model → dashboard).
5. Review semua file Python (bugs, redundansi, code smell).
6. Rekomendasi perbaikan data pipeline.
7. Rekomendasi perbaikan kode dan struktur folder.
8. Ide optimasi untuk developer agar sistem lebih scalable.
9. dan paling penting ada untuk improvable fraud label (fraud_label.py) dari yang sudah ada berdasarkan dari data data yang sudah dianalisis apa yang bisa diubah dari fraud label apa yang bisa dihapus apa yang bisa ditambahkan dan apa yang bisa membuat fraud label bisa lebih dynamis dan bisa membuat labelling makin detailed dan dynamis untuk data baru nanti


Jawaban sangat detail, kritis, teknis, dan actionable.
"""
resp = model.generate_content(prompt)

with open("AI_DEV_REVIEW.txt", "w", encoding="utf-8") as f:
    f.write(resp.text)

print("✔ AI developer review saved to: AI_DEV_REVIEW.txt")
