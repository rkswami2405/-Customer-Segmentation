import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Ecommerce Dashboard", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Ecommerce Customers.csv")

df = load_data()

# ✅ FIX: Define min_spending HERE (VERY IMPORTANT)
min_spending = df['Yearly Amount Spent'].min()

# =========================
# MODEL TRAINING
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
st.sidebar.title("📊 Navigation")

mode = st.sidebar.radio(
    "Select Mode",
    ["Manual Prediction", "CSV Upload Analysis", "📊 Visualization", "🔍 Bulk Scanner"]
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

        # ✅ Prevent empty input
        if (avg_session == 0 and time_app == 0 and 
            time_web == 0 and membership == 0):
            st.warning("⚠️ Please enter valid input values.")
            st.stop()

        input_data = pd.DataFrame({
            'Avg. Session Length': [avg_session],
            'Time on App': [time_app],
            'Time on Website': [time_web],
            'Length of Membership': [membership]
        })

        prediction = model.predict(input_data)[0]

        # ✅ FIX: Replace negative/zero
        prediction = prediction if prediction > 0 else min_spending

        st.success(f"💰 Estimated Spending: ${prediction:,.2f}")

# =========================
# 2. CSV UPLOAD ANALYSIS
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
                predictions = model.predict(data[FEATURES])

                # ✅ FIX: Replace negative/zero
                predictions = np.where(predictions <= 0, min_spending, predictions)

                data['Predicted Spending'] = predictions

                st.success("✅ Prediction Completed")
                st.dataframe(data)

# =========================
# 3. VISUALIZATION
# =========================
elif mode == "📊 Visualization":

    st.title("📊 Interactive Data Visualization Dashboard")

    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.scatter(df, x='Time on App', y='Yearly Amount Spent'),
                        use_container_width=True)

    with col2:
        st.plotly_chart(px.scatter(df, x='Time on Website', y='Yearly Amount Spent'),
                        use_container_width=True)

    st.plotly_chart(px.histogram(df, x='Yearly Amount Spent'),
                    use_container_width=True)

# =========================
# 4. BULK SCANNER
# =========================
elif mode == "🔍 Bulk Scanner":

    st.title("🔍 Bulk Customer Scanner")

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel / JSON",
        type=["csv", "xlsx", "json"]
    )

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_json(uploaded_file)

        st.dataframe(data.head())

        if st.button("🚀 Run Bulk Prediction"):

            missing_cols = [col for col in FEATURES if col not in data.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                st.stop()

            try:
                predictions = model.predict(data[FEATURES])

                # ✅ FIX: Replace negative/zero
                predictions = np.where(predictions <= 0, min_spending, predictions)

                data['Predicted Spending'] = predictions

                st.success("✅ Prediction Completed")
                st.dataframe(data)

            except Exception as e:
                st.error(f"❌ Error: {e}")
