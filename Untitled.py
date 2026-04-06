import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Ecommerce ML App", layout="wide")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 Navigation")
option = st.sidebar.radio(
    "Go to:",
    ["Home", "Data Preview", "Visualization", "Clustering", "Models", "Prediction"]
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Ecommerce Customers.csv")
    return df

df = load_data()

# =========================
# HOME
# =========================
if option == "Home":
    st.title("🛒 Ecommerce Customer Analysis App")
    st.write("""
    This app performs:
    - Customer Segmentation (KMeans)
    - Regression (Spending Prediction)
    - Classification (Decision Tree & Random Forest)
    """)

# =========================
# DATA PREVIEW
# =========================
elif option == "Data Preview":
    st.title("📂 Dataset")
    st.dataframe(df.head())
    st.subheader("Statistics")
    st.write(df.describe())

# =========================
# VISUALIZATION
# =========================
elif option == "Visualization":
    st.title("📊 Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.scatter(df['Time on Website'], df['Yearly Amount Spent'])
        ax1.set_title("Website Time vs Spending")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['Time on App'], df['Yearly Amount Spent'])
        ax2.set_title("App Time vs Spending")
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(df['Yearly Amount Spent'], bins=30)
    ax3.set_title("Spending Distribution")
    st.pyplot(fig3)

# =========================
# CLUSTERING
# =========================
elif option == "Clustering":
    st.title("🔵 Customer Segmentation (KMeans)")

    df_encoded = df.copy()

    # Encoding
    le = LabelEncoder()
    df_encoded['Email'] = le.fit_transform(df_encoded['Email'])
    df_encoded['Address'] = le.fit_transform(df_encoded['Address'])
    df_encoded['Avatar'] = le.fit_transform(df_encoded['Avatar'])

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded)

    k = st.slider("Select number of clusters", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df_encoded['Cluster'] = kmeans.fit_predict(scaled_data)

    st.write("Clustered Data")
    st.dataframe(df_encoded.head())

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df_encoded['Time on App'],
        y=df_encoded['Yearly Amount Spent'],
        hue=df_encoded['Cluster'],
        ax=ax
    )
    st.pyplot(fig)

# =========================
# MODELS
# =========================
elif option == "Models":
    st.title("🤖 Machine Learning Models")

    df_model = df.copy()

    # Encoding
    le = LabelEncoder()
    df_model['Email'] = le.fit_transform(df_model['Email'])
    df_model['Address'] = le.fit_transform(df_model['Address'])
    df_model['Avatar'] = le.fit_transform(df_model['Avatar'])

    # KMeans for target
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_model)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df_model['Cluster'] = kmeans.fit_predict(scaled_data)

    # Classification
    X = df_model.drop('Cluster', axis=1)
    y = df_model['Cluster']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    st.subheader("Accuracy")
    st.write("Decision Tree:", accuracy_score(y_test, pred_dt))
    st.write("Random Forest:", accuracy_score(y_test, pred_rf))

    # Feature Importance
    st.subheader("Feature Importance")
    importances = rf.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(X.columns, importances)
    st.pyplot(fig)

# =========================
# PREDICTION
# =========================
elif option == "Prediction":
    st.title("🔮 Predict Customer Spending")

    avg_session = st.number_input("Avg. Session Length", 0.0)
    time_app = st.number_input("Time on App", 0.0)
    time_web = st.number_input("Time on Website", 0.0)
    membership = st.number_input("Length of Membership", 0.0)

    # Train model
    X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = df['Yearly Amount Spent']

    model = LinearRegression()
    model.fit(X, y)
        if st.button("Predict"):

            input_data = pd.DataFrame({
                'Avg. Session Length': [avg_session],
                'Time on App': [time_app],
                'Time on Website': [time_web],
                'Length of Membership': [membership]
            })
