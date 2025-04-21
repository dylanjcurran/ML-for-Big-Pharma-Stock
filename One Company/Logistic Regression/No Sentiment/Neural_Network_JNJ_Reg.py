import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

#--- Variables ---
CSV_FILE = "JNJ.csv"

#--- Load & Preprocess Data ---
df = pd.read_csv(CSV_FILE)

"""
# Get counts for each class
count_up = sum(df["Output"] == 1)
count_down = sum(df["Output"] == 0)

# Get the smaller class count
n_samples = min(count_up, count_down)

# Sample from both classes equally
df_up = df[df["Output"] == 1].sample(n=n_samples, random_state=42)
df_down = df[df["Output"] == 0].sample(n=n_samples, random_state=42)

# Combine and shuffle
df = pd.concat([df_up, df_down])
df = df.sort_values(by=["StartDate"]).reset_index(drop=True)
"""

df.columns = df.columns.str.strip()

# Drop leaky features
drop_features = ['Simple Moving Average', 'Bollinger Upper Band', 'Bollinger Lower Band']
features = [col for col in df.columns if col not in drop_features and col not in ["Output", "StartDate", "EndDate", "Ticker", "Company"]]

# Shift features to prevent data leakage
df[features] = df[features].shift(1)

# Drop missing values
df.dropna(inplace=True)

# Sort by date
df = df.sort_values(by=["StartDate", "Ticker"]).reset_index(drop=True)

# Drop metadata columns
df = df.drop(columns=["Ticker", "Company", "StartDate", "EndDate"] + drop_features)

# Split features and target
X = df[features]
y = (df["Output"] > 0).astype(int)

# 80/20 split
train_size = int(len(df) * 0.8)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--- Logistic Regression ---
clf = LogisticRegression(class_weight='balanced')

print(y.value_counts())
clf.fit(X_train_scaled, y_train)

#--- Predictions & Evaluation ---
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#--- Plots ---

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("Confusion_Matrix_Heatmap.png", dpi=300)
plt.close()

# Probability histogram
plt.figure(figsize=(7, 4))
plt.hist(y_prob, bins=20)
plt.xlabel("Predicted Probability of Class 1")
plt.ylabel("Count")
plt.title("Probability Distribution from Logistic Regression")
plt.tight_layout()
plt.savefig("Probability_Histogram.png", dpi=300)
plt.close()

#Results
df_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability": y_prob
})
df_results.to_csv("Model_Predictions.csv", index=False)
