import pandas as pd
from faker import Faker
import random
import datetime as dt

fake = Faker('id_ID')

# Tarif standar diagnosis (perkiraan INA-CBGs)
tarif_standar = {
    'J00': 500_000, 'J18.9': 4_200_000, 'I10': 2_000_000, 'E11': 2_500_000,
    'N39.0': 2_200_000, 'M54.5': 1_800_000, 'H25.9': 7_500_000, 'K29.7': 1_700_000,
    'A09': 1_300_000, 'R50.9': 1_000_000, 'Z03.8': 1_200_000, 'O80': 3_500_000,
    'Z34.0': 800_000, 'R10.4': 1_200_000, 'K21.0': 1_400_000, 'F32.9': 2_800_000,
    'J45.9': 2_300_000, 'I25.1': 5_000_000, 'E66.9': 1_000_000, 'Z00.0': 700_000
}

# Mapping diagnosis ke prosedur
procedure_map = {
    'J00': ['93.9'], 'J18.9': ['96.04', '93.94'], 'I10': ['99.04'],
    'E11': ['99.10'], 'N39.0': ['57.94', '57.98'], 'M54.5': ['93.39'],
    'H25.9': ['13.1', '13.4', '13.5'], 'K29.7': ['45.16'], 'A09': ['45.23'],
    'R50.9': ['99.19'], 'Z03.8': ['93.91'], 'O80': ['72.0'], 'Z34.0': ['73.0'],
    'R10.4': ['45.23', '87.09'], 'K21.0': ['45.16'], 'F32.9': ['94.39'],
    'J45.9': ['93.96'], 'I25.1': ['36.15', '99.24'], 'E66.9': ['99.28'],
    'Z00.0': ['93.99']
}

# === Range waktu ===
start_date = dt.date(2024, 1, 1)
today = dt.date.today()

# === 1) Generate NIK LIST (REALISTIK) ===
num_patients = 1500   # hanya 1500 pasien, tapi total claim 5000 (realisitk)

def generate_nik():
    nik_raw = str(fake.random_number(digits=16))
    return nik_raw.zfill(16)  # pastikan selalu 16 digit

patient_NIKs = [generate_nik() for _ in range(num_patients)]

# Distribusi kunjungan per pasien
visit_distribution = random.choices(
    population=[1, 2, 3, 4],
    weights=[0.70, 0.20, 0.08, 0.02], 
    k=num_patients
)

# Expand ke list final NIK → ini menghasilkan >5000 rows
NIK_list = []
for nik, visits in zip(patient_NIKs, visit_distribution):
    NIK_list.extend([nik] * visits)

# Pastikan panjang minimal 5000
if len(NIK_list) < 5000:
    needed = 5000 - len(NIK_list)
    NIK_list.extend(random.choices(patient_NIKs, k=needed))

random.shuffle(NIK_list)
NIK_list = NIK_list[:5000]

# === 2) Generate DUMMY CLAIMS ===
diagnosis_codes = list(tarif_standar.keys())
data = []

for i in range(5000):
    nik = NIK_list[i]

    age = random.randint(1, 90)
    gender = random.choice(['M', 'F'])
    provider_id = f"RS{random.randint(1, 10):03d}"

    total_claim_amount = random.randint(500_000, 10_000_000)
    length_of_stay = random.choice([0, 1, 2, 3, 5, 7])
    diag_code = random.choices(
        diagnosis_codes,
        weights=[10, 8, 12, 10, 7, 6, 5, 5, 5, 4, 3, 2, 2, 3, 3, 2, 2, 1, 1, 4],
        k=1
    )[0]
    proc_code = random.choice(procedure_map[diag_code])

    # Service & claim date (lebih realistis)
    service_date = fake.date_between(start_date=start_date, end_date=today - dt.timedelta(days=14))
    claim_date = fake.date_between(start_date=service_date + dt.timedelta(days=3), end_date=today)

    record = {
        'NIK': nik,
        'NIK_valid': random.choice([0, 1]),
        'biometric_flag': random.choice([0, 1]),
        'age': age,
        'gender': gender,
        'provider_id': provider_id,
        'diagnosis_code': diag_code,
        'procedure_code': proc_code,
        'num_diagnoses': random.randint(1, 3),
        'num_procedures': random.randint(1, 4),
        'length_of_stay': length_of_stay,
        'total_claim_amount': total_claim_amount,
        'service_date': service_date,
        'claim_date': claim_date,
        'verification_method': random.choice(['auto', 'manual']),
        'tarif_standar_diagnosis': tarif_standar[diag_code],
        'diagnosis_cost_ratio': total_claim_amount / tarif_standar[diag_code]
    }

    data.append(record)

df = pd.DataFrame(data)
df = df.sort_values(by='claim_date')
df.to_excel('dummy_claims_2024_2025.xlsx', index=False)

print("✅ Dummy data klaim REALISTIK berhasil dibuat!")
print(df.head(10))
