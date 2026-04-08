import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Ecommerce AI Dashboard", layout="wide")

st.title("🛒 Customer Spending Prediction Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Ecommerce Customers.csv")

df = load_data()

# =========================
# MODEL TRAINING (UPGRADED)
# =========================
FEATURES = [
    'Avg. Session Length',
    'Time on App',
    'Time on Website',
    'Length of Membership'
]

X = df[FEATURES]
y = df['Yearly Amount Spent']

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

# ✅ Minimum realistic value
min_spending = df['Yearly Amount Spent'].min()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 Navigation")

mode = st.sidebar.radio(
    "Select Mode",
    ["🔮 Manual Prediction", "📂 CSV Upload Analysis", "📊 Visualization", "🔍 Bulk Scanner"]
)

# =========================
# 🔮 MANUAL PREDICTION
# =========================
if mode == "🔮 Manual Prediction":

    st.header("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        avg_session = st.number_input("Avg. Session Length", min_value=0.0, step=0.1)
        time_app = st.number_input("Time on App", min_value=0.0, step=0.1)

    with col2:
        time_web = st.number_input("Time on Website", min_value=0.0, step=0.1)
        membership = st.number_input("Length of Membership", min_value=0.0, step=0.1)

    if st.button("Predict Spending"):

        # ✅ Prevent empty input
        if (avg_session == 0 and time_app == 0 and 
            time_web == 0 and membership == 0):
            st.warning("⚠️ Please enter valid input values.")
            st.stop()

        input_df = pd.DataFrame([{
            'Avg. Session Length': avg_session,
            'Time on App': time_app,
            'Time on Website': time_web,
            'Length of Membership': membership
        }])

        prediction = model.predict(input_df)[0]

        # ✅ Fix negative/zero
        prediction = prediction if prediction > 0 else min_spending

        prediction = round(prediction, 2)

        st.success(f"💰 Estimated Spending: ${prediction:,.2f}")

# =========================
# 📂 CSV UPLOAD ANALYSIS
# =========================
elif mode == "📂 CSV Upload Analysis":

    st.header("CSV Prediction Tool")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        st.dataframe(data.head())

        if st.button("Run Predictions"):

            if not all(col in data.columns for col in FEATURES):
                st.error("❌ Missing required columns")
                st.stop()

            preds = model.predict(data[FEATURES])

            preds = np.where(preds <= 0, min_spending, preds)

            data['Predicted Spending'] = preds

            st.success("✅ Prediction Completed")
            st.dataframe(data)

# =========================
# 📊 VISUALIZATION
# =========================
elif mode == "📊 Visualization":

    st.header("Data Visualization Dashboard")

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

    # Feature Importance
    st.subheader("🔥 Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance.set_index("Feature"))

# =========================
# 🔍 BULK SCANNER
# =========================
elif mode == "🔍 Bulk Scanner":

    st.header("Bulk Prediction Tool")

    sample_df = pd.DataFrame({
        'Avg. Session Length': [30],
        'Time on App': [12],
        'Time on Website': [40],
        'Length of Membership': [3]
    })

    st.download_button(
        "📄 Download Sample CSV",
        sample_df.to_csv(index=False),
        "sample.csv"
    )

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

        if st.button("Run Bulk Prediction"):

            if not all(col in data.columns for col in FEATURES):
                st.error("❌ Missing required columns")
                st.stop()

            preds = model.predict(data[FEATURES])

            preds = np.where(preds <= 0, min_spending, preds)

            data["Predicted Spending"] = preds

            st.success("✅ Completed")
            st.dataframe(data.head(10))

            st.download_button(
                "📥 Download Results",
                data.to_csv(index=False),
                "results.csv"
            )
