# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

# 2. Load Dataset
df = pd.read_csv('/mnt/data/extracted_dataset/spam.csv', encoding='latin1')
df = df[['v1', 'v2']]  # Keep only necessary columns
df.columns = ['label', 'message']

# 3. Preprocessing
# Encode labels (ham = 0, spam = 1)
le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])  # ham=0, spam=1

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Collect performance metrics
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# 6. Display Results
for model_name, metrics in results.items():
    print(f"\n=== {model_name} ===")
    for metric_name, value in metrics.items():
        if metric_name != "Confusion Matrix":
            print(f"{metric_name}: {value:.4f}")
    print("Confusion Matrix:\n", metrics["Confusion Matrix"])

# 7. Plot ROC Curve
plt.figure(figsize=(10, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# 8. Optional: Feature Importance for Random Forest
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20 features
top_features = np.array(vectorizer.get_feature_names_out())[indices]

plt.figure(figsize=(10, 6))
plt.barh(top_features, importances[indices])
plt.xlabel("Importance")
plt.title("Top 20 Important Features (Random Forest)")
plt.tight_layout()
plt.show()

# 9. Optional: GridSearchCV for best Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='f1')
grid.fit(X_train, y_train)
print("\nBest Parameters from GridSearchCV for Random Forest:")
print(grid.best_params_)
