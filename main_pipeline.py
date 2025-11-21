import os

print("=== RUNNING Feature Engineering ===")
os.system("python feature_engineering.py")

print("=== RUNNING Train RF Classifier ===")
os.system("python classification_RF.py")

print("=== Menggabung Dataset ===")
os.system("python gabung_dataset.py")

print("=== PIPELINE COMPLETE ===")
