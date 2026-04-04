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

st.title("Ecommerce Customer Analysis")

# =========================
# Load Data
# =========================
df = pd.read_csv("Ecommerce Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Info")
st.write(df.describe())

# =========================
# Visualization
# =========================
st.subheader("Visualizations")

fig1, ax1 = plt.subplots()
ax1.scatter(df['Time on Website'], df['Yearly Amount Spent'])
ax1.set_title("Website Time vs Spending")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.scatter(df['Time on App'], df['Yearly Amount Spent'])
ax2.set_title("App Time vs Spending")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.hist(df['Yearly Amount Spent'], bins=30)
ax3.set_title("Spending Distribution")
st.pyplot(fig3)

# =========================
# Encoding
# =========================
le = LabelEncoder()
df['Email'] = le.fit_transform(df['Email'])
df['Address'] = le.fit_transform(df['Address'])
df['Avatar'] = le.fit_transform(df['Avatar'])

# =========================
# Clustering
# =========================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

st.subheader("Clustered Data")
st.dataframe(df.head())

# =========================
# Better Regression Model (ONLY NUMERIC FEATURES)
# =========================
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

st.write("Model Score:", lr.score(X_test, y_test))

# =========================
# Classification
# =========================
X = df.drop(['Cluster'], axis=1)
y = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

st.subheader("Model Accuracy")
st.write("Decision Tree:", accuracy_score(y_test, pred_dt))
st.write("Random Forest:", accuracy_score(y_test, pred_rf))

# =========================
# Feature Importance
# =========================
importances = rf.feature_importances_
features = X.columns

fig4, ax4 = plt.subplots()
ax4.barh(features, importances)
ax4.set_title("Feature Importance")
st.pyplot(fig4)

# =========================
# USER INPUT PREDICTION
# =========================
st.subheader("🔮 Predict Customer Spending")

avg_session = st.number_input("Avg. Session Length", min_value=0.0)
time_app = st.number_input("Time on App", min_value=0.0)
time_web = st.number_input("Time on Website", min_value=0.0)
membership = st.number_input("Length of Membership", min_value=0.0)

if st.button("Predict Spending"):

    # Create input with ONLY required columns
    input_data = pd.DataFrame({
        'Avg. Session Length': [avg_session],
        'Time on App': [time_app],
        'Time on Website': [time_web],
        'Length of Membership': [membership]
    })

    prediction = lr.predict(input_data)

    st.success(f"Estimated Yearly Spending: ${prediction[0]:.2f}")
