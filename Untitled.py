import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Ecommerce AI Dashboard", layout="wide")
st.title("🛒 Ecommerce Customer Intelligence Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Ecommerce Customers.csv")

# =========================
# MODEL TRAINING
# =========================
FEATURES = [
    'Avg. Session Length',
    'Time on App',
    'Time on Website',
    'Length of Membership'
]

df = None
if os.path.exists("Ecommerce Customers.csv"):
    df = load_data()

if df is not None:
    X = df[FEATURES]
    y = df['Yearly Amount Spent']

    model = LinearRegression()
    model.fit(X, y)
else:
    model = None
    st.error("Dataset not found!")

# =========================
# SIDEBAR
# =========================
option = st.sidebar.radio(
    "📊 Select Mode",
    ["🔮 Manual Prediction", "📂 CSV Upload Analysis", "🔍 Bulk Scanner"]
)

# ==========================================================
# 🔮 MANUAL PREDICTION
# ==========================================================
if option == "🔮 Manual Prediction":

    st.header("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        avg_session = st.number_input("Avg. Session Length", 0.0, 50.0, 30.0)
        time_app = st.number_input("Time on App", 0.0, 50.0, 12.0)

    with col2:
        time_web = st.number_input("Time on Website", 0.0, 50.0, 40.0)
        membership = st.number_input("Length of Membership", 0.0, 10.0, 3.0)

    if st.button("Predict Spending") and model:

        input_df = pd.DataFrame([{
            'Avg. Session Length': avg_session,
            'Time on App': time_app,
            'Time on Website': time_web,
            'Length of Membership': membership
        }])

        prediction = model.predict(input_df)[0]

        if prediction < 0:
            st.error("❌ Invalid prediction")
        else:
            st.success(f"💰 Estimated Spending: ${prediction:,.2f}")

# ==========================================================
# 🔍 BULK SCANNER
# ==========================================================
elif option == "🔍 Bulk Scanner":

    st.header("🔎 Bulk Customer Scanner")

    # Sample file
    sample_df = pd.DataFrame([{
        'Avg. Session Length': 30,
        'Time on App': 12,
        'Time on Website': 40,
        'Length of Membership': 3
    }])

    st.subheader("1. Download Sample Templates")

    c1, c2, c3 = st.columns(3)

    csv_sample = sample_df.to_csv(index=False).encode('utf-8')
    c1.download_button("📄 CSV Sample", csv_sample, "sample.csv")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        sample_df.to_excel(writer, index=False)
    c2.download_button("📊 Excel Sample", buffer.getvalue(), "sample.xlsx")

    c3.download_button("📦 JSON Sample", sample_df.to_json(), "sample.json")

    st.divider()

    # Upload
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "json"])

    if uploaded_file and model:

        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_json(uploaded_file)

        st.dataframe(data.head())

        if st.button("🚀 Start Bulk Prediction"):

            if all(col in data.columns for col in FEATURES):

                preds = model.predict(data[FEATURES])
                data["Predicted Spending"] = preds

                st.success("✅ Completed")
                st.dataframe(data.head(10))

                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "results.csv")

            else:
                st.error("Missing required columns!")

# ==========================================================
# 📂 CSV ANALYSIS DASHBOARD (SMART)
# ==========================================================
elif option == "📂 CSV Upload Analysis":

    st.header("📊 Smart Data Dashboard")

    data_source = st.radio(
        "Select Data Source",
        ["Use Default Dataset", "Upload CSV"]
    )

    data = None

    if data_source == "Use Default Dataset":
        if df is not None:
            data = df.copy()
            st.info("Using Ecommerce Customers dataset")
        else:
            st.error("Dataset not found")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            data = pd.read_csv(uploaded)

    if data is not None:

        # Sampling for performance
        if len(data) > 5000:
            st.warning("Large dataset, sampling 5000 rows")
            data = data.sample(5000)

        # Prediction
        if model and all(col in data.columns for col in FEATURES):
            data["Predicted Spending"] = model.predict(data[FEATURES])

        # ================= KPI CARDS =================
        st.subheader("📌 Key Insights")

        c1, c2, c3, c4 = st.columns(4)

        if "Predicted Spending" in data.columns:
            c1.metric("Avg Spending", f"${data['Predicted Spending'].mean():,.0f}")
            c2.metric("Max Spending", f"${data['Predicted Spending'].max():,.0f}")
        else:
            c1.metric("Avg Membership", f"{data['Length of Membership'].mean():.2f}")
            c2.metric("Max Membership", f"{data['Length of Membership'].max():.2f}")

        c3.metric("Avg App Time", f"{data['Time on App'].mean():.2f}")
        c4.metric("Total Records", f"{len(data):,}")

        # ================= DYNAMIC VISUALIZATION =================
        st.subheader("📊 Interactive Visualization")

        selected_col = st.selectbox("Select Column", data.columns)

        if data[selected_col].dtype in ['int64', 'float64']:

            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.histogram(data, x=selected_col, title="Distribution")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = px.box(data, y=selected_col, title="Spread")
                st.plotly_chart(fig2, use_container_width=True)

        else:
            vc = data[selected_col].value_counts().reset_index()
            vc.columns = [selected_col, "Count"]

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(px.bar(vc, x=selected_col, y="Count"), use_container_width=True)

            with col2:
                st.plotly_chart(px.pie(vc, names=selected_col, values="Count"), use_container_width=True)

        # ================= CORRELATION HEATMAP =================
        st.subheader("🌡️ Correlation Heatmap")

        numeric_df = data.select_dtypes(include=['number'])

        fig = px.imshow(
            numeric_df.corr(),
            text_auto=True,
            title="Feature Correlation"
        )
        st.plotly_chart(fig, use_container_width=True)
