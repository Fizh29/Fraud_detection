import os

# print("=== RUNNING Dummy Generation ===")
# os.system("python dummy_gene.py")

# print("=== RUNNING Feature Engineering ===")
# os.system("python feature_engineering.py")

# print("=== RUNNING Fraud Labeling ===")
# os.system("python fraud_label.py")

# print("=== RUNNING Train RF Classifier ===")
# os.system("python classification_RF.py")

# print("=== RUNNING Predict New Claims ===")
# os.system("python new_claims.py")

# print("=== Menggabung Dataset ===")
# os.system("python gabung_dataset.py")

# print("=== melakukan analasis dan visualisasi data ===")
# os.system("python aidashboard.py")

# print("=== Masukkan NIK untuk analisis dari salah satu NIK ===")
# os.system("python ai_patient_explainer.py")

# print("=== Untuk developer  ===")
# os.system("python ai_dev_reviewer.py")
# os.system("python dashboard/dashboard.py")

os.system("streamlit run dashboard/dashboard.py --server.address=0.0.0.0 --server.port=8501")

print("=== PIPELINE COMPLETE ===")
