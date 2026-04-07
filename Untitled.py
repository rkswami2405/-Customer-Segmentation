import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Ecommerce AI Dashboard", layout="wide")

st.title("🛒 Ecommerce Customer Intelligence Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("Ecommerce Customers.csv")

df = load_data()

# =========================
# FEATURES
# =========================
FEATURES = [
    'Avg. Session Length',
    'Time on App',
    'Time on Website',
    'Length of Membership'
]

X = df[FEATURES]
y = df['Yearly Amount Spent']

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL (UPGRADED)
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# MODEL EVALUATION
# =========================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

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
        avg_session = st.number_input("Avg. Session Length", 0.0, 50.0, 30.0)
        time_app = st.number_input("Time on App", 0.0, 50.0, 12.0)

    with col2:
        time_web = st.number_input("Time on Website", 0.0, 50.0, 40.0)
        membership = st.number_input("Length of Membership", 0.0, 10.0, 3.0)

    if st.button("Predict Spending"):

        input_df = pd.DataFrame([{
            'Avg. Session Length': avg_session,
            'Time on App': time_app,
            'Time on Website': time_web,
            'Length of Membership': membership
        }])

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated Spending: ${prediction:,.2f}")

# =========================
# 📂 CSV UPLOAD ANALYSIS
# =========================
elif mode == "📂 CSV Upload Analysis":

    st.header("📊 Smart Dashboard")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        data = pd.read_csv(file)

        st.dataframe(data.head())

        if all(col in data.columns for col in FEATURES):
            preds = model.predict(data[FEATURES])
            data["Predicted Spending"] = preds

            st.subheader("📊 Predictions")
            st.dataframe(data)

# =========================
# 📊 VISUALIZATION
# =========================
elif mode == "📊 Visualization":

    st.header("📊 Data Insights")

    # ================= KPI =================
    st.subheader("📌 Model Performance")

    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE", f"${mae:.2f}")

    # ================= FEATURE IMPORTANCE =================
    st.subheader("🔥 Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.plotly_chart(
        px.bar(importance, x="Feature", y="Importance",
               title="Feature Importance"),
        use_container_width=True
    )

    # ================= SCATTER =================
    st.subheader("📊 Relationships")

    st.plotly_chart(
        px.scatter(df, x='Time on App', y='Yearly Amount Spent'),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(df, x='Length of Membership', y='Yearly Amount Spent'),
        use_container_width=True
    )

# =========================
# 🔍 BULK SCANNER
# =========================
elif mode == "🔍 Bulk Scanner":

    st.header("🔎 Bulk Prediction Tool")

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

        if st.button("🚀 Run Prediction"):

            if not all(col in data.columns for col in FEATURES):
                st.error("❌ Missing required columns")
                st.stop()

            preds = model.predict(data[FEATURES])
            data["Predicted Spending"] = preds

            st.success("✅ Done")
            st.dataframe(data)

            st.download_button(
                "📥 Download Results",
                data.to_csv(index=False),
                "results.csv"
            )
