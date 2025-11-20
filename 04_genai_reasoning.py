import google.generativeai as genai
import pandas as pd
import numpy as np
import joblib

# Load Gemini
genai.configure(api_key="AIzaSyDWASI1ydh57y3BA415_wXVEsjJzM-Mavs")

model = joblib.load("model_ensemble.pkl")
shap_values = np.load("shap_values.npy")
df = pd.read_excel("dummy_claims_with_fraud_label.xlsx")

# Ambil 1 klaim untuk penjelasan AI
row_id = 10
record = df.iloc[row_id]
features = record.drop("fraud_label")
prediction = model.predict([features])[0]

# SHAP untuk kasus ini
shap_single = shap_values[1][row_id]  # index class HIGH/MEDIUM/NORMAL

top_contrib = sorted(
    list(zip(features.index, shap_single)),
    key=lambda x: abs(x[1]),
    reverse=True
)[:5]

prompt = f"""
Berikan penjelasan analitis tentang potensi fraud pada klaim asuransi berikut:

Prediksi model: {prediction}
Fitur penting:
{top_contrib}

Berikan analisis:
1. Kenapa klaim ini terdeteksi HIGH / MEDIUM / NORMAL
2. Faktor biaya / medis / perilaku yang mencurigakan
3. Rekomendasi tindak lanjut investigasi

Sampaikan dalam bahasa Indonesia profesional yang mudah dipahami auditor.
"""

result = genai.GenerativeModel("gemini-pro").generate_content(prompt)

print("\n=== Gemini Fraud Analysis ===\n")
print(result.text)
