import os
import json
import hashlib
import pandas as pd
import numpy as np
from typing import List, Dict

import ollama   # <--- gunakan ollama

# ============ CONFIG ============
MODEL_NAME = "llama3"  # ganti jika mau: "llama3:instruct" / "qwen2.5"

INPUT_XLSX = "dummy_claims_with_fraud_label.xlsx"
OUTPUT_RULES_JSON = "ai_rule_suggestions.json"
OUTPUT_EXPL_CSV = "ai_claim_explanations.csv"

SAMPLE_PER_CLASS = 8
NUM_RULES_REQUEST = 6

# ============ HELPERS ============
def anonymize_nik(nik: str) -> str:
    try:
        s = str(int(float(nik)))
    except:
        s = str(nik)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def summarize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols].describe().transpose()[["count","mean","std","min","25%","50%","75%","max"]]


def df_sample_for_prompt(df: pd.DataFrame, features: List[str], label_col="fraud_label", n=SAMPLE_PER_CLASS):
    samples = []
    for lab in df[label_col].unique():
        df_lab = df[df[label_col] == lab].sample(
            min(n, len(df[df[label_col] == lab])),
            random_state=42
        )

        for _, r in df_lab.iterrows():
            record = {}
            for f in features:
                val = r[f]
                if pd.api.types.is_numeric_dtype(df[f]):
                    record[f] = None if pd.isna(val) else float(val)
                else:
                    record[f] = str(val)
            record[label_col] = lab
            samples.append(record)
    return samples


# ============ LLM CALL ============
def call_llm(prompt: str, model=MODEL_NAME, max_tokens=1500):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ============ MAIN ============
def main():

    df = pd.read_excel(INPUT_XLSX)

    if "NIK" in df.columns:
        df["NIK_h"] = df["NIK"].astype(str).apply(anonymize_nik)

    features = [
        'NIK_valid','biometric_flag','age','gender','provider_id','diagnosis_code','procedure_code',
        'num_diagnoses','num_procedures','length_of_stay','total_claim_amount','diagnosis_cost_ratio',
        'avg_cost_per_procedure','verification_delay_days','duplicate_ID_count','duplicate_ID_count_month',
        'time_between_admissions','num_claims_last_30d_by_provider','num_unique_patients_last_30d',
        'provider_claim_rate_vs_peer','claim_rejection_rate_provider','month_over_month_claim_growth',
        'sudden_spike_flag','new_provider_month_flag','avg_claim_per_patient',
        'claim_fragmentation_score','service_mix_index'
    ]

    features = [f for f in features if f in df.columns]

    label_col = "fraud_label"

    stats_per_label = {}
    for lab in df[label_col].unique():
        numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        stats_per_label[lab] = summarize_numeric(
            df[df[label_col] == lab], numeric_cols
        ).fillna(0).to_dict()

    sample_records = df_sample_for_prompt(df, features, label_col, SAMPLE_PER_CLASS)

    # ===================== BUILD PROMPT =====================
    prompt = f"""
You are an expert fraud analyst. You will receive:
1) Summary statistics of healthcare claim data (per label).
2) Sample claim records.

Your task:
Create up to {NUM_RULES_REQUEST} fraud-detection rules.

Each rule must include:
- id
- human_readable_description
- logical_condition (pandas query string)
- reason
- expected_direction
- weight (0-100)
- examples (subset of provided samples)

Return STRICT JSON ONLY with this structure:

{{
  "rules": [
    {{
      "id": "string",
      "human_readable_description": "string",
      "logical_condition": "string",
      "reason": "string",
      "expected_direction": "string",
      "weight": 0,
      "examples": []
    }}
  ],
  "notes": "string"
}}

Summary statistics per label:
{json.dumps(stats_per_label)}

Sample records:
{json.dumps(sample_records)}
"""

    # ===================== CALL LLM =====================
    print("Calling LLM (Ollama)...")
    raw = call_llm(prompt)

    try:
        suggestions = json.loads(raw)
    except:
        print("LLM did not return valid JSON. Raw output:")
        print(raw)
        return

    with open(OUTPUT_RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2)

    print(f"Saved rule suggestions to {OUTPUT_RULES_JSON}")

    # ===================== VALIDATION =====================
    validated = []

    for r in suggestions.get("rules", []):
        cond = r.get("logical_condition", "")
        if not cond:
            continue

        try:
            matched = df.query(cond)
        except Exception as e:
            print(f"Error evaluating rule `{r.get('id')}`: {e}")
            continue

        support = len(matched)
        precision = None
        if support > 0:
            precision = (matched[label_col] == "HIGH").sum() / support

        validated.append({
            "id": r.get("id"),
            "description": r.get("human_readable_description"),
            "condition": cond,
            "support": support,
            "precision": precision,
            "weight": r.get("weight")
        })

    pd.DataFrame(validated).to_excel("ai_rule_validation.xlsx", index=False)
    print("Saved rule validation to ai_rule_validation.xlsx")

    # ===================== EXPLANATIONS (OPTIONAL) =====================
    high_df = df[df[label_col] == "HIGH"].head(50)
    explanations = []

    for idx, row in high_df.iterrows():
        abr = {k: row[k] for k in features}

        p = f"""
Explain why the following healthcare claim may be HIGH risk.
Return JSON:
{{
  "claim_id": "{idx}",
  "explanation": ["bullet1", "bullet2"]
}}

Record:
{json.dumps(abr)}
"""
        raw_expl = call_llm(p)
        try:
            explanations.append(json.loads(raw_expl))
        except:
            explanations.append({"claim_id": idx, "raw": raw_expl})

    pd.DataFrame(explanations).to_csv(OUTPUT_EXPL_CSV, index=False)
    print("Saved explanations to ai_claim_explanations.csv")


if __name__ == "__main__":
    main()
