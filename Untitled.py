import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Ecommerce Dashboard", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Ecommerce Customers.csv")

df = load_data()

# =========================
# TRAIN MODEL (ONLY REQUIRED FEATURES)
# =========================
FEATURES = [
    'Avg. Session Length',
    'Time on App',
    'Time on Website',
    'Length of Membership'
]

X = df[FEATURES]
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
# 1. MANUAL PREDICTION
# =========================
if mode == "Manual Prediction":

    st.title("🛒 Customer Spending Prediction")

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
            st.error("❌ Invalid prediction! Spending cannot be negative.")
        else:
            st.success(f"💰 Estimated Spending: ${prediction:.2f}")

# =========================
# 2. CSV ANALYSIS
# =========================
elif mode == "CSV Upload Analysis":

    st.title("📂 CSV Upload Analysis")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        st.subheader("Preview")
        st.dataframe(data.head())

        if st.button("Run Predictions"):

            missing_cols = [col for col in FEATURES if col not in data.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                input_data = data[FEATURES]

                predictions = model.predict(input_data)

                data['Predicted Spending'] = predictions
                st.dataframe(data)

# =========================
# 3. BULK SCANNER
# =========================
elif mode == "🔍 Bulk Scanner":

    st.title("💰 Ecommerce Prediction Dashboard")
    st.markdown("### 🔍 Bulk Customer Scanner")

    # =========================
    # DOWNLOAD SAMPLE
    # =========================
    st.markdown("## 1. Download Sample Template")

    sample_df = pd.DataFrame({
        'Avg. Session Length': [30],
        'Time on App': [12],
        'Time on Website': [40],
        'Length of Membership': [3]
    })

    csv = sample_df.to_csv(index=False).encode('utf-8')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button("📄 CSV Sample", csv, "sample.csv")

    with col2:
        st.download_button("📊 Excel Sample", csv, "sample.xlsx")

    with col3:
        st.download_button("📦 JSON Sample", sample_df.to_json(), "sample.json")

    st.divider()

    # =========================
    # UPLOAD FILE
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

            # ✅ Check required columns
            missing_cols = [col for col in FEATURES if col not in data.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                st.stop()

            # ✅ Select correct columns
            input_data = data[FEATURES]

            try:
                predictions = model.predict(input_data)

                data['Predicted Spending'] = predictions

                # ✅ Handle negative values
                data['Predicted Spending'] = data['Predicted Spending'].apply(
                    lambda x: x if x >= 0 else None
                )

                st.success("✅ Prediction Completed")
                st.dataframe(data)

            except Exception as e:
                st.error(f"❌ Error: {e}")
