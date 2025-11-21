import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon=":bar_chart:",
    initial_sidebar_state="expanded"
)

st.title("💡 Fraud Detection Claims Dashboard")

# Pilih file data
data_file = st.selectbox(
    "📁 Pilih data yang ingin ditampilkan:",
    [
        "dummy_claims_with_fraud_label.csv",
        "prediction_result.csv",
        "dummy_claims_2024_2025.csv"
    ]
)

# Load data
try:
    df = pd.read_csv(data_file)
except Exception as e:
    st.error(f"Gagal load file: {e}")
    st.stop()


# ===== HARD FIX STREAMLIT + ARROW (FORCE ALL DATETIME TO STRING) =====
for col in df.columns:
    # detect otomatis: tipe datetime, atau string tapi mirip datetime, atau kolom berisi Timestamps
    if df[col].dtype == "datetime64[ns]" or df[col].dtype == "datetime64[ms]" or "date" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].astype(str)  # **PAKSA STRING**
        except:
            df[col] = df[col].astype(str)

# ===== Summary Metrics =====
st.markdown("### 📊 Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Claims", f"{len(df):,}")
if "fraud_label" in df.columns:
    fraud_count = (df["fraud_label"] != "NORMAL").sum()
    fraud_rate = 100 * fraud_count / len(df)
    col2.metric("Fraudulent Claims", f"{fraud_count:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Unique Patients", df["NIK"].nunique() if "NIK" in df.columns else "-")
else:
    col2.metric("Unique Patients", df["NIK"].nunique() if "NIK" in df.columns else "-")
    col3.metric("Unique Providers", df["provider"].nunique() if "provider" in df.columns else "-")
    col4.metric("Total Biaya", f"Rp {df['biaya'].sum():,.0f}" if "biaya" in df.columns else "-")

st.markdown("---")

# ===== Charts Layout =====
chart_col1, chart_col2 = st.columns([2, 1])
with chart_col1:
    if "claim_date" in df.columns:
        st.markdown("#### Klaim per Bulan")
        
        temp_claim_date = pd.to_datetime(df["claim_date"], errors="coerce")
        monthly = df.groupby(temp_claim_date.dt.to_period("M")).size()

        st.bar_chart(monthly)


with chart_col2:
    if "fraud_label" in df.columns:
        st.markdown("#### Distribusi Fraud Label")
        label_counts = df["fraud_label"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        label_counts.plot.pie(autopct='%1.1f%%', ax=ax, colors=["#90caf9", "#f48fb1", "#ffe082"])
        ax.set_ylabel("")
        st.pyplot(fig)

st.markdown("---")

# ===== Data Table =====
st.markdown("### 🗂️ Data Sample")
st.dataframe(df.head(30), use_container_width=True)

# ===== Filter Data (Opsional) =====
with st.expander("🔍 Filter Data", expanded=False):
    columns = df.columns.tolist()
    for col in columns:
        if df[col].dtype == "object":
            options = df[col].unique().tolist()
            selected = st.multiselect(f"{col}", options, default=options)
            df = df[df[col].isin(selected)]
        elif df[col].dtype in ["int64", "float64"]:
            min_val, max_val = df[col].min(), df[col].max()
            if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val:
                selected = st.slider(f"{col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
                df = df[(df[col] >= selected[0]) & (df[col] <= selected[1])]
            elif pd.notna(min_val) and pd.notna(max_val) and min_val == max_val:
                st.info(f"Kolom '{col}' hanya memiliki satu nilai: {min_val}")
            else:
                st.info(f"Kolom '{col}' tidak memiliki data untuk difilter.")

# ===== Statistik Dasar =====
with st.expander("📈 Statistik Dasar", expanded=False):
    st.write(df.describe())