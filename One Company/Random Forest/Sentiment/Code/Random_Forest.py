import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ----------------------------
# CONFIG
# ----------------------------

CSV_FILE = "JNJSent.csv"
USE_ONLY_SENTIMENT_ROWS = False # Toggle this ON/OFF
TARGET_COL = "Output"
DROP_COLS = ["StartDate", "EndDate", "Ticker", "Company"]  # Drop if they exist

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv(CSV_FILE)

if USE_ONLY_SENTIMENT_ROWS:
    df = df[df["num_mentions"] > 0].copy()

# Drop unused columns
for col in DROP_COLS:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Drop rows with NaNs just in case
df = df.dropna()

# ----------------------------
# TRAIN-TEST SPLIT
# ----------------------------

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# TRAIN MODEL
# ----------------------------

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # for AUC

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# FEATURE IMPORTANCES (Optional)
# ----------------------------

importances = pd.Series(clf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print("\nTop Features:\n", importances.head(10))
