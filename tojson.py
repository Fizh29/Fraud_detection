import pandas as pd

df = pd.read_csv("Data_Claims_classification.csv")

df.to_json("Data_Claims_classification.json", orient="records", indent=4)

print("✔ Konversi CSV → JSON berhasil!")

