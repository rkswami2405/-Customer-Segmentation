# Generated from: Untitled.ipynb
# Converted at: 2026-04-04T17:43:18.995Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML models
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Profiling
from ydata_profiling import ProfileReport

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Ecommerce Customers.csv")
df.head()

df.info()
df.describe()
df.isnull().sum()

plt.figure(figsize=(8,5))
plt.plot(df['Time on Website'], df['Yearly Amount Spent'])
plt.title("Time on Website vs Yearly Amount Spent")
plt.xlabel("Time on Website")
plt.ylabel("Yearly Amount Spent")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df['Time on App'], df['Yearly Amount Spent'])
plt.title("Time on App vs Spending")
plt.xlabel("Time on App")
plt.ylabel("Yearly Amount Spent")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Yearly Amount Spent'], bins=30)
plt.title("Distribution of Yearly Amount Spent")
plt.xlabel("Spending")
plt.ylabel("Frequency")
plt.show()

sns.pairplot(df)
plt.show()

profile = ProfileReport(df)
profile.to_file("Ecommerce_Report.html")

plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

le = LabelEncoder()
df['Email'] = le.fit_transform(df['Email'])
df['Address'] = le.fit_transform(df['Address'])
df['Avatar'] = le.fit_transform(df['Avatar'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5)
df['Cluster'] = kmeans.fit_predict(scaled_data)

sns.scatterplot(x=df.iloc[:,3], y=df.iloc[:,4], hue=df['Cluster'])
plt.show()

X = df.drop(['Yearly Amount Spent','Cluster'], axis=1)
y = df['Yearly Amount Spent']

X

y

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

X_train

X_test

lr = LinearRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)
print("Score:", lr.score(X_test, y_test))

y_pred = lr.predict(X_test)

#mae = mean_absolute_error(y_test,y_pred)

#mae

X = df.drop(['Cluster'], axis=1)
y = df['Cluster']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, pred_dt))
print(confusion_matrix(y_test, pred_dt))
print(classification_report(y_test, pred_dt))

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=X.columns)
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))
print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.title("Feature Importance - Random Forest")
plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(rf.estimators_[0], filled=True, feature_names=X.columns)
plt.show()

dt_acc = accuracy_score(y_test, pred_dt)
rf_acc = accuracy_score(y_test, pred_rf)

models = ['Decision Tree', 'Random Forest']
scores = [dt_acc, rf_acc]

plt.figure(figsize=(6,4))
plt.bar(models, scores)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    x=X_test['Time on App'], 
    y=X_test['Yearly Amount Spent'], 
    hue=pred_rf
)
plt.title("Random Forest Predictions (Test Data)")
plt.show()

importances = rf.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()

plt.figure(figsize=(15,10))
plot_tree(rf.estimators_[0], filled=True)
plt.show()

print("Decision Tree Accuracy:", accuracy_score(y_test, pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))