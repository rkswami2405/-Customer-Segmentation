import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Ecommerce Dashboard", layout="wide")

# =========================
# LOAD DATA & TRAIN MODEL
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Ecommerce Customers.csv")

df = load_data()

X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

model = LinearRegression()
model.fit(X, y)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Select Mode")

mode = st.sidebar.radio(
    "",
    ["Manual Prediction", "CSV Upload Analysis", "🔍 Bulk Scanner"]
)

# =========================
# MANUAL PREDICTION
# =========================
if mode == "Manual Prediction":

    st.title("🛒 Ecommerce Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        avg_session = st.number_input("Avg. Session Length", min_value=0.0)
        time_app = st.number_input("Time on App", min_value=0.0)

    with col2:
        time_web = st.number_input("Time on Website", min_value=0.0)
        membership = st.number_input("Length of Membership", min_value=0.0)

    if st.button("Predict Spending"):

        input_data = pd.DataFrame({
            'Avg. Session Length': [avg_session],
            'Time on App': [time_app],
            'Time on Website': [time_web],
            'Length of Membership': [membership]
        })

        prediction = model.predict(input_data)[0]

        if prediction < 0:
            st.error("❌ Invalid prediction! Please enter realistic values.")
        else:
            st.success(f"💰 Estimated Spending: ${prediction:.2f}")

# =========================
# CSV ANALYSIS
# =========================
elif mode == "CSV Upload Analysis":

    st.title("📂 CSV Upload Analysis")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        st.subheader("Preview")
        st.dataframe(data.head())

        if st.button("Run Predictions"):
            predictions = model.predict(data)

            data['Predicted Spending'] = predictions
            st.dataframe(data)

# =========================
# BULK SCANNER (LIKE YOUR IMAGE)
# =========================
elif mode == "🔍 Bulk Scanner":

    st.title("💰 Ecommerce Spending Prediction Dashboard")
    st.markdown("### 🔍 Bulk Customer Scanner")

    # =========================
    # DOWNLOAD SECTION
    # =========================
    st.markdown("## 1. Download Sample Templates")

    sample_df = pd.DataFrame({
        'Avg. Session Length': [30],
        'Time on App': [12],
        'Time on Website': [40],
        'Length of Membership': [3]
    })

    csv = sample_df.to_csv(index=False).encode('utf-8')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button("📄 Download CSV Sample", csv, "sample.csv", "text/csv")

    with col2:
        st.download_button("📊 Download Excel Sample", csv, "sample.xlsx")

    with col3:
        st.download_button("📦 Download JSON Sample", sample_df.to_json(), "sample.json")

    st.divider()

    # =========================
    # UPLOAD SECTION
    # =========================
    st.markdown("## 2. Upload File to Scan")

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel / JSON",
        type=["csv", "xlsx", "json"]
    )

    if uploaded_file is not None:

        # Detect file type
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_json(uploaded_file)

        st.subheader("📊 Uploaded Data")
        st.dataframe(data.head())

        if st.button("🔍 Run Bulk Prediction"):

            try:
                predictions = model.predict(data)
                data['Predicted Spending'] = predictions

                st.success("✅ Prediction Completed")
                st.dataframe(data)

            except Exception as e:
                st.error(f"❌ Error: {e}")
