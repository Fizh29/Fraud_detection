import pandas as pd
from faker import Faker
import random
import datetime as dt
import numpy as np

fake = Faker('id_ID')

# Tarif INA-CBGs dummy
tarif_standar = {
    'J00': 500_000, 'J18.9': 4_200_000, 'I10': 2_000_000, 'E11': 2_500_000,
    'N39.0': 2_200_000, 'M54.5': 1_800_000, 'H25.9': 7_500_000, 'K29.7': 1_700_000,
    'A09': 1_300_000, 'R50.9': 1_000_000, 'Z03.8': 1_200_000, 'O80': 3_500_000,
    'Z34.0': 800_000, 'R10.4': 1_200_000, 'K21.0': 1_400_000, 'F32.9': 2_800_000,
    'J45.9': 2_300_000, 'I25.1': 5_000_000, 'E66.9': 1_000_000, 'Z00.0': 700_000
}

# Procedure mapping per diagnosis
procedure_map = {
    'J00': ['93.9'], 'J18.9': ['96.04', '93.94'], 'I10': ['99.04'],
    'E11': ['99.10'], 'N39.0': ['57.94', '57.98'], 'M54.5': ['93.39'],
    'H25.9': ['13.1', '13.4', '13.5'], 'K29.7': ['45.16'], 'A09': ['45.23'],
    'R50.9': ['99.19'], 'Z03.8': ['93.91'], 'O80': ['72.0'], 'Z34.0': ['73.0'],
    'R10.4': ['45.23', '87.09'], 'K21.0': ['45.16'], 'F32.9': ['94.39'],
    'J45.9': ['93.96'], 'I25.1': ['36.15', '99.24'], 'E66.9': ['99.28'],
    'Z00.0': ['93.99']
}

# ----------------------------
# 1. Diagnosis Groups (untuk buat dummy yang realistis)
# ----------------------------
diagnosis_groups = {
    "ISPA": ['J00', 'J18.9', 'J45.9'],
    "Metabolic": ['E11', 'E66.9'],
    "Cardio": ['I10', 'I25.1'],
    "Gastro": ['K29.7', 'K21.0', 'A09'],
    "Urinary": ['N39.0'],
    "Musculoskeletal": ['M54.5'],
    "Eye": ['H25.9'],
    "Obstetri": ['O80', 'Z34.0'],
    "General": ['R50.9', 'Z03.8', 'Z00.0', 'R10.4']
}

# Flatten list diagnosis
all_diag_codes = [d for v in diagnosis_groups.values() for d in v]

# ----------------------------
# 2. Setting tanggal dataset
# ----------------------------
start_date = dt.date(2024, 1, 1)
today = dt.date.today()

TARGET_ROWS = 5030
num_patients = 1500

# Provider list
provider_list = [f"RS{str(i).zfill(3)}" for i in range(1, 11)]

# ----------------------------
# 3. Generate patient master → identity konsisten
# ----------------------------
def generate_nik():
    raw = str(fake.random_number(digits=16))
    return raw.zfill(16)

patient_NIKs = [generate_nik() for _ in range(num_patients)]

patient_master = {}

for nik in patient_NIKs:
    base_diag = random.choice(all_diag_codes)

    patient_master[nik] = {
        "NIK_valid": random.choice([1, 1, 1, 0]),
        "biometric_flag": random.choice([1, 1, 1, 0]),
        "age": random.randint(1, 90),
        "gender": random.choice(['M','F']),
        "provider_default": random.choice(provider_list),
        "base_diag": base_diag
    }

# ----------------------------
# 4. Tentukan jumlah kunjungan per pasien
# ----------------------------
visit_dist = random.choices(
    population=[1,2,3,4],
    weights=[0.70,0.20,0.08,0.02],
    k=num_patients
)

NIK_list = []
for nik, visits in zip(patient_NIKs, visit_dist):
    NIK_list.extend([nik]*visits)

# Pastikan jumlah row = TARGET_ROWS
if len(NIK_list) < TARGET_ROWS:
    NIK_list.extend(random.choices(patient_NIKs, k=TARGET_ROWS - len(NIK_list)))

random.shuffle(NIK_list)
NIK_list = NIK_list[:TARGET_ROWS]

# ----------------------------
# 5. Generate claim rows realistis
# ----------------------------
data = []
def generate_non_sunday(start, end):
    d = fake.date_between(start, end)
    while d.weekday() == 6:  # 6 = Minggu
        d = fake.date_between(start, end)
    return d

for nik in NIK_list:
    p = patient_master[nik]

    # Provider: 97% tetap, 3% rujukan
    if random.random() < 0.97:
        provider = p["provider_default"]
    else:
        others = [x for x in provider_list if x != p["provider_default"]]
        provider = random.choice(others)

    # Tentukan diagnosis (70% sama, 25% mirip, 5% random)
    base = p["base_diag"]
    r = random.random()

    if r < 0.70:
        diag_code = base
    elif r < 0.95:
        for group, codes in diagnosis_groups.items():
            if base in codes:
                diag_code = random.choice(codes)
                break
    else:
        diag_code = random.choice(all_diag_codes)

    proc_code = random.choice(procedure_map[diag_code])
    service_date = generate_non_sunday(start_date, today - dt.timedelta(days=14))
    claim_date = generate_non_sunday(service_date + dt.timedelta(days=3), today)

    

    total_claim_amount = random.randint(400_000, 10_000_000)
        # --- Length of Stay Realistic Mapping ---
    los_distribution = {
        "ISPA":             [0]*55 + [1]*30 + [2]*10 + [3]*4 + [4]*1,
        "Metabolic":        [0]*15 + [1]*35 + [2]*30 + [3]*10 + [4]*7 + [5]*2 + [6]*1,
        "Cardio":           [0]*5  + [1]*15 + [2]*25 + [3]*30 + [4]*15 + [5]*5 + [6]*3 + [7]*2,
        "Gastro":           [0]*40 + [1]*30 + [2]*20 + [3]*8 + [4]*2,
        "Urinary":          [0]*35 + [1]*25 + [2]*25 + [3]*10 + [4]*3 + [5]*2,
        "Musculoskeletal":  [0]*70 + [1]*20 + [2]*10,
        "Eye":              [0]*85 + [1]*15,
        "Obstetri":         [1]*20 + [2]*50 + [3]*20 + [4]*10,
        "General":          [0]*80 + [1]*15 + [2]*5
    }

    # Cari group diagnosis dari patient
    def get_diag_group(code):
        for group, codes in diagnosis_groups.items():
            if code in codes:
                return group
        return "General"

    diag_group = get_diag_group(diag_code)
    length_of_stay = random.choice(los_distribution[diag_group])

    data.append({
        "NIK": nik,
        "NIK_valid": p["NIK_valid"],
        "biometric_flag": p["biometric_flag"],
        "age": p["age"],
        "gender": p["gender"],
        "provider_id": provider,
        "diagnosis_code": diag_code,
        "procedure_code": proc_code,
        "num_diagnoses": random.randint(1, 3),
        "num_procedures": random.randint(1, 4),
        'length_of_stay': length_of_stay,
        "total_claim_amount": total_claim_amount,
        "service_date": service_date,
        "claim_date": claim_date,
        "verification_method": random.choice(["auto","manual"]),
        "tarif_standar_diagnosis": tarif_standar[diag_code],
        "diagnosis_cost_ratio": total_claim_amount / tarif_standar[diag_code]
    })


df = pd.DataFrame(data)
df = df.sort_values("claim_date")

# ----------------------------
# 6. Inject noise sesuai proporsi
# ----------------------------
num_rows = len(df)

# --- 0.5% Emergency Sunday Service bukan fraud ---
emergency_idx = np.random.choice(df.index, size=int(0.0001 * num_rows), replace=False)

for i in emergency_idx:
    original_service = df.at[i, 'service_date']
    
    # Dapatkan minggu terdekat (Minggu ke depan)
    shift_to_sunday = (6 - original_service.weekday()) % 7
    new_service_date = original_service + dt.timedelta(days=shift_to_sunday)

    df.at[i, 'service_date'] = new_service_date

# 0.5% Sunday claim fraud
sunday_fraud_idx = np.random.choice(df.index, size=int(0.002 * num_rows), replace=False)

for i in sunday_fraud_idx:
    # Tentukan claim_date Minggu terdekat (ke depan)
    base_date = df.at[i, 'claim_date']
    shift_to_sunday = (6 - base_date.weekday()) % 7
    new_claim_date = base_date + dt.timedelta(days=shift_to_sunday)

    # Service date dipaksa Jumat (dua hari sebelum Minggu)
    new_service_date = new_claim_date - dt.timedelta(days=2)

    df.at[i, 'claim_date'] = new_claim_date
    df.at[i, 'service_date'] = new_service_date

# 5% LOS anomaly (bisa naik/turun drastis tapi realistis)
los_anomaly_idx = np.random.choice(df.index, size=int(0.0003*num_rows), replace=False)
for i in los_anomaly_idx:
    actual_los = df.at[i, 'length_of_stay']
    if actual_los <= 2:
        df.at[i, 'length_of_stay'] = actual_los + random.choice([1,2,3])
    else:
        df.at[i, 'length_of_stay'] = max(1, actual_los - random.randint(1, actual_los-1))

# 3% inflated cost
inflated_cost_idx = np.random.choice(df.index, size=int(0.0001*num_rows), replace=False)
df.loc[inflated_cost_idx, 'total_claim_amount'] = (
    df.loc[inflated_cost_idx, 'total_claim_amount'] * random.uniform(1.5, 3.0)
).astype(int)


# 2% provider anomaly
provider_anomaly_idx = np.random.choice(df.index, size=int(0.005*num_rows), replace=False)
for i in provider_anomaly_idx:
    original_provider = df.at[i, 'provider_id']
    alternatives = [p for p in provider_list if p != original_provider]
    df.at[i, 'provider_id'] = random.choice(alternatives)

# 5% realistic fraud pattern (gabungan beberapa field, misal LOS tinggi tapi tarif standar rendah + inflated cost)
fraud_idx = np.random.choice(df.index, size=int(0.001*num_rows), replace=False)
for i in fraud_idx:
    df.at[i, 'length_of_stay'] += random.randint(1,3)
    df.at[i, 'total_claim_amount'] = int(
    df.at[i, 'total_claim_amount'] * random.uniform(1.2, 2.5)
    )

# -----------------------------------------
# 7. Late claim (0.07%)
# -----------------------------------------
late_claim_idx = np.random.choice(df.index, size=int(0.0007 * num_rows), replace=False)
for i in late_claim_idx:
    df.at[i, 'claim_date'] = df.at[i, 'service_date'] + dt.timedelta(days=random.randint(40, 90))


# -----------------------------------------
# 8. Gender error (0.04%)
# -----------------------------------------
gender_error_idx = np.random.choice(df.index, size=int(0.0004 * num_rows), replace=False)
for i in gender_error_idx:
    df.at[i, 'gender'] = 'M' if df.at[i, 'gender'] == 'F' else 'F'


# -----------------------------------------
# 9. Medical diagnosis typo (0.06%) 
# Diagnosis diganti ke kode 1 prefix sama (typo realistis)
# -----------------------------------------
def get_similar_diag(code):
    prefix = code[0]   # ambil huruf pertama
    same_prefix = [d for d in all_diag_codes if d[0] == prefix and d != code]
    return random.choice(same_prefix) if same_prefix else code

med_typo_idx = np.random.choice(df.index, size=int(0.0006 * num_rows), replace=False)
for i in med_typo_idx:
    df.at[i, 'diagnosis_code'] = get_similar_diag(df.at[i, 'diagnosis_code'])


# -----------------------------------------
# 10. Missing procedure (0.01%)
# -----------------------------------------
missing_proc_idx = np.random.choice(df.index, size=int(0.0001 * num_rows), replace=False)
for i in missing_proc_idx:
    df.at[i, 'procedure_code'] = None
    df.at[i, 'num_procedures'] = max(0, df.at[i, 'num_procedures'] - 1)


# -----------------------------------------
# 11. High-frequency visitor (0.09%)
# Tambah kunjungan palsu → service_date mundur 1–5 hari
# -----------------------------------------
high_visit_idx = np.random.choice(df.index, size=int(0.0009 * num_rows), replace=False)
for i in high_visit_idx:
    df.at[i, 'service_date'] = df.at[i, 'claim_date'] - dt.timedelta(days=random.randint(0, 2))


# -----------------------------------------
# 12. Diagnosis–procedure mismatch (0.02%)
# diagnosis sengaja diganti ke group lain
# -----------------------------------------
def get_wrong_diag(diag_code):
    for group, codes in diagnosis_groups.items():
        if diag_code in codes:
            other_groups = {g: codes2 for g, codes2 in diagnosis_groups.items() if g != group}
            return random.choice(random.choice(list(other_groups.values())))
    return diag_code

mismatch_idx = np.random.choice(df.index, size=int(0.0002 * num_rows), replace=False)
for i in mismatch_idx:
    df.at[i, 'diagnosis_code'] = get_wrong_diag(df.at[i, 'diagnosis_code'])


# -----------------------------------------
# 13. Upgrade diagnosis (0.09%)
# Naikkan severity → contoh J00 -> J18.9
# -----------------------------------------
upgrade_map = {
    'J00': 'J18.9',  # common ISPA → pneumonia
    'J18.9': 'J18.9',
    'I10': 'I25.1',
    'E11': 'E66.9',
    'N39.0': 'N39.0',
    'K29.7': 'K21.0',
    'A09': 'K29.7',
    'R50.9': 'J18.9',
    'Z03.8': 'R50.9',
    'Z34.0': 'O80',
}

upgrade_idx = np.random.choice(df.index, size=int(0.0009 * num_rows), replace=False)
for i in upgrade_idx:
    old = df.at[i, 'diagnosis_code']
    df.at[i, 'diagnosis_code'] = upgrade_map.get(old, old)


# -----------------------------------------
# 14. Diagnosis count inflated (0.04%)
# num_diagnoses = 4–6
# -----------------------------------------
diag_inflate_idx = np.random.choice(df.index, size=int(0.0004 * num_rows), replace=False)
for i in diag_inflate_idx:
    df.at[i, 'num_diagnoses'] = random.randint(4, 6)

df['NIK'] = df['NIK'].astype(str)

print(df['NIK'].head())
print("✔ Noise injected according to your proportions!")
df.to_csv("dummy_claims_2024_2025.csv", index=False)

print("✔ 5030 dummy claims generated realistically!")
print(df.head())
