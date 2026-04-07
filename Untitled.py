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

# ✅ Get minimum valid spending (IMPORTANT FIX)
min_spending = df['Yearly Amount Spent'].min()

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

        # ✅ FIX: Replace negative or zero
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

                # ✅ FIX: Replace negative or zero values
                predictions = np.where(predictions <= 0, min_spending, predictions)

                data['Predicted Spending'] = predictions

                st.success("✅ Prediction Completed")
                st.dataframe(data)

# =========================
# 3. VISUALIZATION DASHBOARD
# =========================
elif mode == "📊 Visualization":

    st.title("📊 Interactive Data Visualization Dashboard")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Scatter
    st.subheader("🔵 Scatter Plots")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.scatter(df, x='Time on App', y='Yearly Amount Spent',
                       title="Time on App vs Spending"),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.scatter(df, x='Time on Website', y='Yearly Amount Spent',
                       title="Website Time vs Spending"),
            use_container_width=True
        )

    # Histogram
    st.subheader("📊 Distribution")

    st.plotly_chart(
        px.histogram(df, x='Yearly Amount Spent', nbins=30,
                     title="Spending Distribution"),
        use_container_width=True
    )

    # Line
    st.subheader("📈 Membership Impact")

    st.plotly_chart(
        px.line(df, x='Length of Membership', y='Yearly Amount Spent',
                title="Membership vs Spending"),
        use_container_width=True
    )

    # Bar
    st.subheader("📊 Average Spending")

    bar_data = df.groupby('Length of Membership')['Yearly Amount Spent'].mean().reset_index()

    st.plotly_chart(
        px.bar(bar_data, x='Length of Membership', y='Yearly Amount Spent',
               title="Avg Spending by Membership"),
        use_container_width=True
    )

    # Heatmap
    st.subheader("🌡️ Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['number'])

    st.plotly_chart(
        px.imshow(numeric_df.corr(), text_auto=True,
                  title="Correlation Matrix"),
        use_container_width=True
    )

    # Pie
    st.subheader("🥧 Spending Categories")

    df_temp = df.copy()
    df_temp['Category'] = pd.cut(
        df_temp['Yearly Amount Spent'],
        bins=3,
        labels=["Low", "Medium", "High"]
    )

    counts = df_temp['Category'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']

    st.plotly_chart(
        px.pie(counts, names='Category', values='Count',
               title="Customer Segments"),
        use_container_width=True
    )

# =========================
# 4. BULK SCANNER (FINAL FIX)
# =========================
elif mode == "🔍 Bulk Scanner":

    st.title("🔍 Bulk Customer Scanner")

    sample_df = pd.DataFrame({
        'Avg. Session Length': [30],
        'Time on App': [12],
        'Time on Website': [40],
        'Length of Membership': [3]
    })

    csv = sample_df.to_csv(index=False).encode('utf-8')

    st.download_button("📄 Download Sample CSV", csv, "sample.csv")

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

                # ✅ FINAL FIX: Replace negative & zero values
                predictions = np.where(predictions <= 0, min_spending, predictions)

                data['Predicted Spending'] = predictions

                st.success("✅ Prediction Completed")
                st.dataframe(data)

            except Exception as e:
                st.error(f"❌ Error: {e}")
