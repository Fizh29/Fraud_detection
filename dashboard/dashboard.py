import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from google import generativeai as genai
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon=":bar_chart:",
    initial_sidebar_state="expanded"
)

st.title("💡 Fraud Detection Claims Dashboard")

# Hanya gunakan file utama hasil gabungan
data_file = "Data_Claims_classification.csv"

# Load data
try:
    df = pd.read_csv(data_file, dtype={"NIK": str})
except Exception as e:
    st.error(f"Gagal load file: {e}")
    st.stop()

# ===== Summary Metrics =====
st.markdown("### 📊 Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Claims", f"{len(df):,}")
col2.metric("total nominal klaim ", f"Rp {df['total_claim_amount'].sum():,.0f}" if "total_claim_amount" in df.columns else "-")
col3.metric("Unique Patients", df["NIK"].nunique() if "NIK" in df.columns else "-")
col4.metric("Unique Providers", df["provider_id"].nunique() if "provider_id" in df.columns else "-")

# Fraud metrics
if "fraud_label" in df.columns:
    fraud_count = (df["fraud_label"] != "NORMAL").sum()
    fraud_rate = 100 * fraud_count / len(df) if len(df) > 0 else 0
    st.info(f"Fraudulent Claims: *{fraud_count:,}* &nbsp;&nbsp;|&nbsp;&nbsp; Fraud Rate: *{fraud_rate:.2f}%*")

st.markdown("---")
# ===== VISUALISASI TAMBAHAN =====
st.markdown("### 📈 Visualisasi Umum")

# ========================================
# VISUALISASI DINAMIS (PLOTLY + GRID 2×2)
# ========================================
import plotly.express as px



row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# ======================
# 1. Total Klaim per Provider
# ======================
if "provider_id" in df.columns:
    with row1_col1:
        st.markdown("#### Total Klaim per Provider")
        claims_per_provider = df.groupby("provider_id").size().reset_index(name="count")

        fig = px.bar(
            claims_per_provider,
            x="provider_id",
            y="count",
            title="Total Klaim per Provider",
            labels={"count": "Jumlah Klaim", "provider_id": "Provider ID"},
            color="count"
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================
# 2. Distribusi Gender
# ======================
if "gender" in df.columns:
    with row1_col2:
        st.markdown("#### Distribusi Gender")
        fig = px.pie(
            df,
            names="gender",
            title="Distribusi Gender",
            color="gender",
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================
# 3. Top 10 Diagnosis
# ======================
if "diagnosis_code" in df.columns:
    with row2_col1:
        st.markdown("#### Top 10 Diagnosis")
        top_diag = df["diagnosis_code"].value_counts().nlargest(10).reset_index()
        top_diag.columns = ["diagnosis_code", "count"]

        fig = px.bar(
            top_diag,
            x="diagnosis_code",
            y="count",
            title="Top 10 Diagnosis",
            labels={"count": "Jumlah Klaim"},
            color="count"
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# ======================
# 4. Tren Klaim Bulanan
# ======================
if "claim_date" in df.columns:
    with row2_col2:
        st.markdown("#### Tren Klaim Bulanan")

        # Convert to datetime
        df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")

        # --- GROUP BY MONTH ---
        monthly = (
            df.groupby(df["claim_date"].dt.to_period("M"))
              .agg(
                  count=("claim_date", "size"),
                  total_amount=("total_claim_amount", "sum")
              )
              .reset_index()
        )

        # Convert Period → Timestamp (avoid 1970 issue)
        monthly["claim_date"] = monthly["claim_date"].dt.to_timestamp()

        # --- PLOT ---
        fig = px.line(
            monthly,
            x="claim_date",
            y="count",
            markers=True,
            title="Total Klaim per Bulan",
            labels={"claim_date": "Bulan", "count": "Jumlah Klaim"},
        )

        # Tambahkan hover custom dengan format duit
        fig.update_traces(
            hovertemplate=
            "<b>Bulan:</b> %{x|%Y-%m}<br>" +
            "<b>Total Klaim:</b> %{y}<br>" +
            "<b>Total Nominal:</b> Rp %{customdata:,}<extra></extra>",
            customdata=monthly["total_amount"]
        )

        # Axis formatting
        fig.update_layout(
            xaxis_tickformat="%Y-%m",
            yaxis_title="Jumlah Klaim",
        )

        st.plotly_chart(fig, use_container_width=True)

 
# ======================
# 5. Total Claim Amount per Provider
# ======================
if "provider_id" in df.columns and "total_claim_amount" in df.columns:
    st.markdown("#### Total Nominal Klaim per Provider")

    # Pastikan angka bersih dan numeric
    df["total_claim_amount"] = pd.to_numeric(df["total_claim_amount"], errors="coerce").fillna(0)

    claims_amount = (
        df.groupby("provider_id")["total_claim_amount"]
        .sum()
        .reset_index()
        .sort_values("total_claim_amount", ascending=False)
    )

    fig = px.bar(
        claims_amount,
        x="provider_id",
        y="total_claim_amount",
        title="Total Nominal Klaim per Provider",
        labels={"provider_id": "Provider", "total_claim_amount": "Total Rupiah"},
        color="total_claim_amount",
    )

    # === Hover + formatting rupiah ===
    fig.update_traces(
        hovertemplate=
        "<b>Provider:</b> %{x}<br>" +
        "<b>Total Nominal:</b> Rp %{y:,.0f}<extra></extra>"
    )

    # Rapihin grafik
    fig.update_layout(
        xaxis_title="Provider ID",
        yaxis_title="Total Nominal Klaim (Rp)",
        bargap=0.2,
        xaxis_tickangle=-45,
    )

    st.plotly_chart(fig, use_container_width=True)



# ===== AI SUMMARY DASHBOARD =====

# ===== AI SUMMARY DASHBOARD (reworked — menyamakan dengan aidashboard.py & pakai visualisasi 1-5) =====
st.markdown("### 🧠 AI Summary Dashboard")

with st.expander("Generate / Refresh AI Summary", expanded=True):
    st.write("Summary akan dibangun dari dashboard ini menggunakan beberapa visualisasi sebagai input ringkasan data.")
    if st.button("Regenerate AI Summary"):
        try:
            # Pastikan kolom-kolom penting ada dan format benar
            if "claim_date" in df.columns:
                df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")
            if "total_claim_amount" in df.columns:
                df["total_claim_amount"] = pd.to_numeric(df["total_claim_amount"], errors="coerce").fillna(0)

            # Visualisasi 1-5 summary data (sesuai aidashboard.py)
            claims_per_provider = df.groupby("provider_id").size().sort_values(ascending=False) if "provider_id" in df.columns else pd.Series(dtype=int)
            gender_dist = df["gender"].value_counts() if "gender" in df.columns else pd.Series(dtype=int)
            top_diag_overall = df["diagnosis_code"].value_counts().nlargest(10) if "diagnosis_code" in df.columns else pd.Series(dtype=int)
            monthly_claims = df.groupby(df["claim_date"].dt.to_period("M")).size() if "claim_date" in df.columns else pd.Series(dtype=int)
            claims_amount = df.groupby("provider_id")["total_claim_amount"].sum().sort_values(ascending=False) if ("provider_id" in df.columns and "total_claim_amount" in df.columns) else pd.Series(dtype=float)

            # Build prompt (padat dan berdasarkan ringkasan 1-5)
            dashboard_summary_prompt = f"""
Berikan analisis profesional, ringkasan insight, dan rekomendasi dari dataset klaim asuransi berdasarkan ringkasan berikut (gunakan Bahasa Indonesia, padat & actionable):

1) Total klaim per provider (top 10): {claims_per_provider.head(10).to_dict()}
2) Distribusi gender: {gender_dist.to_dict()}
3) Top 10 diagnosis: {top_diag_overall.to_dict()}
4) Klaim bulanan (jumlah per bulan): {monthly_claims.to_dict()}
5) Total nominal klaim per provider (top 10): {claims_amount.head(10).to_dict()}

Jelaskan:
- Pola umum klaim dan pasien yang terlihat.
- Diagnosis dan provider yang paling rawan fraud.
- Tren bulanan penting yang perlu diperhatikan.
- 5 rekomendasi tindakan untuk auditor/manajemen untuk mengurangi risiko fraud.
Ringkas, terstruktur, dan actionable.
"""

            # Panggil AI (Gemini) — gunakan konfigurasi dari file lain (sudah diimport)
            with st.spinner("Menghasilkan ringkasan AI..."):
                genai.configure(api_key="")
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                resp = model.generate_content(dashboard_summary_prompt)
                ai_summary_text = resp.text

            # Simpan dan tampilkan
            with open("AI_SUMMARY_DASHBOARD.txt", "w", encoding="utf-8") as f:
                f.write(ai_summary_text)

            st.success("AI summary berhasil di-generate dan disimpan.")
            st.markdown(
                """
                <div style="border:2px solid #90caf9; border-radius:8px; padding:12px; background-color:#f7fbff; max-height:400px; overflow:auto; font-size:15px;">
                <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">{}</pre>
                </div>
                """.format(ai_summary_text),
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Gagal generate AI summary: {e}")

# Jika sudah ada file summary lama, tampilkan (read-only)
try:
    if "ai_summary_text" not in locals():
        with open("AI_SUMMARY_DASHBOARD.txt", "r", encoding="utf-8") as f:
            ai_summary = f.read()
        st.markdown(
            """
            <div style="border:2px solid #90caf9; border-radius:8px; padding:12px; background-color:#f7fbff; max-height:350px; overflow:auto; font-size:15px;">
            <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">{}</pre>
            </div>
            """.format(ai_summary),
            unsafe_allow_html=True
        )
except Exception:
    st.info("AI summary dashboard belum tersedia. Klik 'Regenerate AI Summary' untuk membuatnya.")

st.markdown("---")

# ===== Charts Layout =====
chart_col1, chart_col2 = st.columns([2, 1])
with chart_col1:
    if "claim_date" in df.columns and "total_claim_amount" in df.columns:
        st.markdown("#### Klaim per Bulan")

        df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")

        # 🔥 Grouping lengkap: jumlah klaim + total nominal per bulan
        monthly = (
            df.groupby(df["claim_date"].dt.to_period("M"))
              .agg(
                  count=("NIK", "size"),
                  total_amount=("total_claim_amount", "sum")
              )
              .reset_index()
        )

        # Convert period → string biar bar gak kurus
        monthly["bulan"] = (
            monthly["claim_date"]
            .dt.to_timestamp()
            .dt.strftime("%Y-%m")
        )

        # Chart
        fig = px.bar(
            monthly,
            x="bulan",
            y="count",
            title="Total Klaim per Bulan",
            labels={"bulan": "Bulan", "count": "Jumlah Klaim"},
        )

        # Hover custom — tampilkan duit juga
        fig.update_traces(
            width=0.6,
            hovertemplate=
                "<b>Bulan:</b> %{x}<br>" +
                "<b>Jumlah Klaim:</b> %{y}<br>" +
                "<b>Total Nominal:</b> Rp %{customdata:,.0f}<br>" +
                "<extra></extra>",
            customdata=monthly["total_amount"],   # inject total nominal
        )

        fig.update_layout(
            bargap=0.2,
            bargroupgap=0.1,
            xaxis_type="category",
            xaxis_title="Bulan",
            yaxis_title="Jumlah Klaim",
        )

        st.plotly_chart(fig, use_container_width=True)



with chart_col2:
    # Distribusi fraud label
    if "fraud_label" in df.columns:
        st.markdown("#### Distribusi Fraud Label")
        label_counts = df["fraud_label"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        label_counts.plot.pie(autopct='%1.1f%%', ax=ax, colors=["#90caf9", "#f48fb1", "#ffe082"])
        ax.set_ylabel("")
        st.pyplot(fig)
    # Top provider
    elif "provider" in df.columns:
        st.markdown("#### Top 5 Provider by Claims")
        top_prov = df["provider"].value_counts().head(5)
        st.bar_chart(top_prov)



# #
# # ===== Filter Data (Opsional) =====
# with st.expander("🔍 Filter Data", expanded=False):
#     columns = df.columns.tolist()
#     for col in columns:
#         if df[col].dtype == "object":
#             options = df[col].unique().tolist()
#             selected = st.multiselect(f"{col}", options, default=options)
#             df = df[df[col].isin(selected)]
#         elif df[col].dtype in ["int64", "float64"]:
#             min_val, max_val = df[col].min(), df[col].max()
#             if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val:
#                 selected = st.slider(f"{col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
#                 df = df[(df[col] >= selected[0]) & (df[col] <= selected[1])]
#             elif pd.notna(min_val) and pd.notna(max_val) and min_val == max_val:
#                 st.info(f"Kolom '{col}' hanya memiliki satu nilai: {min_val}")
#             else:
#                 st.info(f"Kolom '{col}' tidak memiliki data untuk difilter.")


# # ===== Data Table =====
# st.markdown("### 🗂 Data Sample")
# st.dataframe(df.head(30), use_container_width=True)

# # ===== Statistik Dasar =====
# with st.expander("📈 Statistik Dasar", expanded=False):
#     st.write(df.describe())

#     # ===== AI PATIENT ANALYSIS REALTIME =====
# st.markdown("### 🧑‍⚕ AI Patient Analysis (Realtime)")

# with st.form("ai_patient_form"):
#     nik_input = st.text_input("Masukkan NIK untuk dianalisis", "")
#     submitted = st.form_submit_button("Analisa Sekarang")
#     analysis_result = None

#     if submitted and nik_input:
#         rows = df[df["NIK"] == nik_input.strip()].sort_values("claim_date")
#         if rows.empty:
#             analysis_result = f"❌ NIK tidak ditemukan: {nik_input}\n\nCoba cek 5 NIK pertama:\n{df['NIK'].head().to_list()}"
#         else:
#             # Siapkan prompt seperti di ai_patient_explainer.py
#             all_claims = rows.to_dict(orient="records")
#             claims_text = "\n\n".join([str(c) for c in all_claims])
#             prompt = f"""
# Berikan analisis lengkap untuk 1 pasien asuransi berdasarkan semua klaim berikut:

# {claims_text}

# Jelaskan secara profesional dalam bahasa Indonesia:

# 1. Penyebab fraud_score pasien ini (jelaskan fitur apa yang mempengaruhi).
# 2. Apakah pasien ini HIGH / MEDIUM / NORMAL risk dan alasannya.
# 3. Pola-pola mencurigakan dari seluruh riwayat klaim NIK ini.
# 4. Risiko tindak kecurangan yang mungkin terjadi.
# 5. Rekomendasi tindakan lanjut untuk auditor.

# Tulis padat, jelas, dan actionable.
# """
#             # Proses dengan Gemini jika tersedia, jika tidak tampilkan prompt saja
#             try:
               
#                 with st.spinner("Sedang menganalisis dengan AI..."):
#                     genai.configure(api_key="")
#                     model = genai.GenerativeModel("models/gemini-2.5-flash")
#                     resp = model.generate_content(prompt)
#                     analysis_result = resp.text
#             except Exception as e:
#                 analysis_result = f"Gagal memproses analisis AI secara realtime.\n\nPrompt yang digunakan:\n{prompt}\n\nError: {e}"

#     if submitted:
#         st.markdown(
#             """
#             <div style="border:2px solid #f48fb1; border-radius:8px; padding:12px; background-color:#fff7fb; max-height:500px; overflow:auto; font-size:15px;">
#             <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">{}</pre>
#             </div>
#             """.format(analysis_result if analysis_result else "Silakan masukkan NIK dan klik Analisa."),
#             unsafe_allow_html=True
#         )



# Keep original full dataset for searches / AI analysis (do not mutate)
df_all = df.copy()

st.markdown("---")

# ===== Lightweight global search (applies only to view, not to df_all) =====
search_query = st.text_input("Cari (global across semua kolom, kosongkan untuk tampilkan semua)", "")

if search_query:
    # cari di semua kolom sebagai string (case-insensitive)
    mask = df_all.astype(str).apply(lambda col: col.str.contains(search_query, case=False, na=False))
    mask = mask.any(axis=1)
    df_view = df_all[mask].copy()
    st.info(f"Menampilkan {len(df_view):,} baris yang cocok untuk query: '{search_query}'")
else:
    df_view = df_all.copy()

# ===== Paginated / Excel-like Data Table (AgGrid jika tersedia, fallback ke st.dataframe dengan pagination) =====
page_size = st.number_input("Rows per page", min_value=10, max_value=200, value=30, step=10)

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode

    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    grid_opts = gb.build()

    st.markdown("### 🗂 Data (klik header untuk filter/sort — menggunakan AgGrid)")
    grid_response = AgGrid(
        df_view,
        gridOptions=grid_opts,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=False,
        height=500,
    )

    # Ambil selected row (jika perlu)
    selected = grid_response.get("selected_rows", [])
    if selected:
        st.write("Selected row detail:")
        st.json(selected[0])

except Exception:
    # fallback: manual pagination with st.dataframe
    st.markdown("### 🗂 Data (fallback: paginated view)")
    total_rows = len(df_view)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size
    st.write(f"Menampilkan rows {start+1} - {min(end, total_rows)} dari {total_rows}")
    st.dataframe(df_view.iloc[start:end], use_container_width=True)

# ===== Statistik Dasar =====
with st.expander("📈 Statistik Dasar", expanded=False):
    st.write(df_all.describe(include='all'))

# ===== AI PATIENT ANALYSIS (Realtime) - gunakan df_all agar tidak terpengaruh view =====
st.markdown("### 🧑‍⚕ AI Patient Analysis (Realtime)")

with st.form("ai_patient_form"):
    nik_input = st.text_input("Masukkan NIK untuk dianalisis", "")
    submitted = st.form_submit_button("Analisa Sekarang")
    analysis_result = None

    if submitted and nik_input:
        rows = df_all[df_all["NIK"] == nik_input.strip()].sort_values("claim_date")
        if rows.empty:
            analysis_result = f"❌ NIK tidak ditemukan: {nik_input}\n\nCoba cek 5 NIK pertama:\n{df_all['NIK'].head().to_list()}"
        else:
            all_claims = rows.to_dict(orient="records")
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
            try:
                with st.spinner("Sedang menganalisis dengan AI..."):
                    genai.configure(api_key="")
                    model = genai.GenerativeModel("models/gemini-2.5-flash")
                    resp = model.generate_content(prompt)
                    analysis_result = resp.text
            except Exception as e:
                analysis_result = f"Gagal memproses analisis AI secara realtime.\n\nError: {e}"

    if submitted:
        st.markdown(
            """
            <div style="border:2px solid #f48fb1; border-radius:8px; padding:12px; background-color:#fff7fb; max-height:500px; overflow:auto; font-size:15px;">
            <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">{}</pre>
            </div>
            """.format(analysis_result if analysis_result else "Silakan masukkan NIK dan klik Analisa."),
            unsafe_allow_html=True
        )

st.markdown("---")



with st.expander("📥 Upload New Claims CSV & Run Pipeline", expanded=False):
    st.write("Upload file CSV klaim baru. Gunakan template header jika butuh format.")
    template_cols = [
        "NIK","NIK_valid","biometric_flag",
        "age","gender","provider_id","diagnosis_code",
        "procedure_code","num_diagnoses","num_procedures","length_of_stay",
        "total_claim_amount","service_date","claim_date","verification_method",
        "tarif_standar_diagnosis","diagnosis_cost_ratio"

    ]
    csv_template = (",".join(template_cols) + "\n").encode("utf-8")
    st.download_button("Download CSV template (header only)", data=csv_template,
                       file_name="new_claims_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV (wajib .csv)", type=["csv"])
    if uploaded:
        st.write("Uploaded:", uploaded.name)
        if st.button("Process uploaded CSV & Update Dataset"):
            import os
            import subprocess
            import joblib
            from sklearn.metrics import classification_report
            try:
                save_name = "dummy_claims_2024_2025.csv"
                with open(save_name, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.info(f"Saved uploaded file as: {save_name}")

                # 1) Feature engineering (will read dummy_claims_2024_2025.csv and output dummy_claims_with_features.csv)
                st.info("Running feature_engineering.py ...")
                subprocess.run(["python","feature_engineering.py"], check=True)
                st.success("Feature engineering finished.")

                # 2) Train / retrain classifier (classification_RF.py will produce fraud_model.pkl)
                st.info("Running classification_RF.py (train model) ...")
                subprocess.run(["python","classification_RF.py"], check=True)
                st.success("Model training finished and fraud_model.pkl saved.")

                # 3) Load model and run prediction on newly engineered data
                st.info("Loading model and predicting labels for new claims ...")
                model = joblib.load("fraud_model.pkl")

                new_feat_path = "dummy_claims_with_features.csv"
                if not os.path.exists(new_feat_path):
                    st.error(f"File not found after feature engineering: {new_feat_path}")
                else:
                    new_df = pd.read_csv(new_feat_path, dtype={"NIK": str})

                    # Feature list must match classification_RF.py features
                    features = [
                        'age','NIK_valid','biometric_flag','num_diagnoses','num_procedures',
                        'length_of_stay','total_claim_amount','diagnosis_cost_ratio',
                        'avg_cost_per_procedure','verification_delay_days',
                        'duplicate_ID_count','duplicate_ID_count_month','time_between_admissions',
                        'num_claims_last_30d_by_provider','num_unique_patients_last_30d',
                        'provider_claim_rate_vs_peer','claim_rejection_rate_provider',
                        'month_over_month_claim_growth','sudden_spike_flag',
                        'avg_claim_per_patient','claim_fragmentation_score','service_mix_index'
                    ]

                    missing_feats = [c for c in features if c not in new_df.columns]
                    if missing_feats:
                        st.warning(f"Beberapa fitur tidak ditemukan di hasil feature engineering: {missing_feats}")

                    X_new = new_df.reindex(columns=features).fillna(0)
                    preds = model.predict(X_new)
                    # Save predictions in column 'predicted_label' so gabung_dataset.py can rename if needed
                    new_df["predicted_label"] = preds

                    # Optional: evaluation if ground-truth fraud_label present in uploaded file (preserved through FE)
                    eval_report = None
                    if "fraud_label" in new_df.columns:
                        report = classification_report(new_df["fraud_label"], preds, output_dict=False)
                        eval_report = report
                        st.text("Evaluation (uploaded truth vs predicted):")
                        st.text(eval_report)

                    # Save to filename used by gabung_dataset.py
                    new_out = "new_claims.csv"
                    new_df.to_csv(new_out, index=False)
                    st.success(f"Predictions saved to: {new_out}")

                # 4) Merge with existing dataset (gabung_dataset.py will create/overwrite Data_Claims_classification.csv)
                st.info("Merging new claims into main dataset (running gabung_dataset.py) ...")
                subprocess.run(["python","gabung_dataset.py"], check=True)
                st.success("Merge complete. Data_Claims_classification.csv updated.")

                # 5) reload main data into dashboard (best-effort)
                try:
                    df_updated = pd.read_csv("Data_Claims_classification.csv", dtype={"NIK": str})
                    st.success(f"Reloaded Data_Claims_classification.csv ({len(df_updated):,} rows).")
                except Exception:
                    st.info("Silakan refresh dashboard untuk melihat data terbaru.")

            except subprocess.CalledProcessError as e:
                st.error(f"Pipeline failed while running subprocess: {e}")
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
