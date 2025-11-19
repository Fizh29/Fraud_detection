import os

print("=== RUNNING MODULE 1: Dummy Generation ===")
os.system("python dummy_gene.py")

print("=== RUNNING MODULE 2: Feature Engineering ===")
os.system("python feature_engineering.py")

print("=== RUNNING MODULE 3: Fraud Labeling ===")
os.system("python fraud_label.py")

print("=== RUNNING MODULE 4: Train RF Classifier ===")
os.system("python classification_RF.py")

print("=== RUNNING MODULE 5: Predict New Claims ===")
os.system("python new_claims.py")

print("=== PIPELINE COMPLETE ===")
