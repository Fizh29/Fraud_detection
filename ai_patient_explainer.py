import pandas as pd
from google import generativeai as genai

# ========== SETUP GEMINI ==========
genai.configure(api_key="AIzaSyDWASI1ydh57y3BA415_wXVEsjJzM-Mavs")
model = genai.GenerativeModel("models/gemini-2.5-flash")

# ========== LOAD FINAL DATASET (PAKSA STRING) ==========
df = pd.read_csv(
    "Data_Claims_classification.csv",
    dtype={"NIK": str}  # <--- PENTING!
)

# Hapus spasi dan .0 jika pernah terjadi
df["NIK"] = df["NIK"].str.replace(".0", "", regex=False).str.strip()

# ========== INPUT NIK ==========
nik_input = input("Masukkan NIK untuk dianalisis: ").strip()
rows = df[df["NIK"] == nik_input].sort_values("claim_date")

if rows.empty:
    print(f"❌ NIK tidak ditemukan: {nik_input}")
    print("Coba cek 5 NIK pertama dari file:")
    print(df['NIK'].head())
    exit()

# ------------------------------
# Ambil semua klaim NIK tertentu
# ------------------------------
# Convert semua baris klaim ke list of dict
all_claims = rows.to_dict(orient="records")

# Buat string yang rapi untuk prompt
claims_text = "\n\n".join([str(c) for c in all_claims])

prompt = f"""
Berikan analisis lengkap untuk 1 pasien asuransi berdasarkan semua klaim berikut:

{claims_text}

Jelaskan secara profesional dalam bahasa Indonesia:

1. Penyebab fraud_score pasien ini (jelaskan fitur apa yang mempengaruhi).
2. Apakah pasien ini HIGH / MEDIUM / NORMAL risk dan alasannya.
3. Pola-pola mencurigakan dari seluruh riwayat klaim NIK ini.
4. Risiko tindak kecurangan yang mungkin terjadi.
5. Rekomendasi tindakan lanjut untuk auditor.

Tulis padat, jelas, dan actionable.
"""

resp = model.generate_content(prompt)

print("\n==============================")
print("📌 ANALISIS SPESIFIK PASIEN")
print("==============================")
print(resp.text)
