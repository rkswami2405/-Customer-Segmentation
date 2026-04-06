import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
                input_data = data[FEATURES]

                predictions = model.predict(input_data)

                data['Predicted Spending'] = predictions
                st.dataframe(data)

# =========================
# 3. VISUALIZATION DASHBOARD
# =========================
elif mode == "📊 Visualization":

    st.title("📊 Data Visualization Dashboard")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Scatter Plots
    st.subheader("🔵 Scatter Plots")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.scatter(df['Time on App'], df['Yearly Amount Spent'])
        ax1.set_title("Time on App vs Spending")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['Time on Website'], df['Yearly Amount Spent'])
        ax2.set_title("Website Time vs Spending")
        st.pyplot(fig2)

    # Histogram
    st.subheader("📊 Histogram")

    fig3, ax3 = plt.subplots()
    ax3.hist(df['Yearly Amount Spent'], bins=30)
    ax3.set_title("Spending Distribution")
    st.pyplot(fig3)

    # Line Plot
    st.subheader("📈 Line Plot")

    fig4, ax4 = plt.subplots()
    ax4.plot(df['Length of Membership'], df['Yearly Amount Spent'])
    ax4.set_title("Membership vs Spending")
    st.pyplot(fig4)

    # Bar Plot
    st.subheader("📊 Bar Plot")

    fig5, ax5 = plt.subplots()
    df.groupby('Length of Membership')['Yearly Amount Spent'].mean().plot(kind='bar', ax=ax5)
    ax5.set_title("Avg Spending by Membership")
    st.pyplot(fig5)

    # Heatmap
    st.subheader("🌡️ Heatmap")

    fig6, ax6 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax6)
    st.pyplot(fig6)

    # Pie Chart
    st.subheader("🥧 Spending Categories")

    df_temp = df.copy()
    df_temp['Category'] = pd.cut(
        df_temp['Yearly Amount Spent'],
        bins=3,
        labels=["Low", "Medium", "High"]
    )

    category_counts = df_temp['Category'].value_counts()

    fig7, ax7 = plt.subplots()
    ax7.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    ax7.set_title("Customer Spending Distribution")
    st.pyplot(fig7)

# =========================
# 4. BULK SCANNER
# =========================
elif mode == "🔍 Bulk Scanner":

    st.title("💰 Ecommerce Prediction Dashboard")
    st.markdown("### 🔍 Bulk Customer Scanner")

    # Download Sample
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

    # Upload
    st.markdown("## 2. Upload File to Scan")

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel / JSON",
        type=["csv", "xlsx", "json"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_json(uploaded_file)

        st.subheader("📊 Uploaded Data")
        st.dataframe(data.head())

        if st.button("🔍 Run Bulk Prediction"):

            missing_cols = [col for col in FEATURES if col not in data.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
                st.stop()

            input_data = data[FEATURES]

            try:
                predictions = model.predict(input_data)

                data['Predicted Spending'] = predictions

                # Handle negative values
                data['Predicted Spending'] = data['Predicted Spending'].apply(
                    lambda x: x if x >= 0 else None
                )

                st.success("✅ Prediction Completed")
                st.dataframe(data)

            except Exception as e:
                st.error(f"❌ Error: {e}")
