import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML models
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# =========================
# Load Data
# =========================
df = pd.read_csv("Ecommerce Customers.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# =========================
# Visualization
# =========================
plt.figure(figsize=(8,5))
plt.scatter(df['Time on Website'], df['Yearly Amount Spent'])
plt.title("Time on Website vs Yearly Amount Spent")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df['Time on App'], df['Yearly Amount Spent'])
plt.title("Time on App vs Spending")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Yearly Amount Spent'], bins=30)
plt.title("Distribution of Yearly Amount Spent")
plt.show()

sns.pairplot(df)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# =========================
# OPTIONAL: Profiling (SAFE)
# =========================
try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df)
    profile.to_file("Ecommerce_Report.html")
    print("Profile report generated.")
except ImportError:
    print("ydata_profiling not installed. Skipping profiling...")

# =========================
# Encoding
# =========================
le = LabelEncoder()
df['Email'] = le.fit_transform(df['Email'])
df['Address'] = le.fit_transform(df['Address'])
df['Avatar'] = le.fit_transform(df['Avatar'])

# =========================
# Scaling + Clustering
# =========================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

sns.scatterplot(x=df.iloc[:,3], y=df.iloc[:,4], hue=df['Cluster'])
plt.show()

# =========================
# Linear Regression
# =========================
X = df.drop(['Yearly Amount Spent','Cluster'], axis=1)
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Linear Regression Score:", lr.score(X_test, y_test))

# =========================
# Classification (Cluster Prediction)
# =========================
X = df.drop(['Cluster'], axis=1)
y = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, pred_dt))
print(confusion_matrix(y_test, pred_dt))
print(classification_report(y_test, pred_dt))

plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=X.columns)
plt.show()

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

# Feature Importance
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.title("Feature Importance - Random Forest")
plt.show()

# =========================
# Model Comparison
# =========================
dt_acc = accuracy_score(y_test, pred_dt)
rf_acc = accuracy_score(y_test, pred_rf)

plt.figure(figsize=(6,4))
plt.bar(['Decision Tree', 'Random Forest'], [dt_acc, rf_acc])
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()

print("\nFinal Accuracy:")
print("Decision Tree:", dt_acc)
print("Random Forest:", rf_acc)
