# =========================
# 1. Import Required Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# =========================
# 2. Load Dataset
# =========================
# Upload your dataset in Colab and update the file name and target column
data = pd.read_csv("dataset.csv")   # <-- change file name if needed

# Display first rows
data.head()

# =========================
# 3. Separate Features and Target
# =========================
X = data.drop("target", axis=1)   # <-- replace "target" with actual label column
y = data["target"]

# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. Decision Tree (Baseline Model)
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# =========================
# 6. Random Forest
# =========================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# =========================
# 7. AdaBoost
# =========================
ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
ada.fit(X_train, y_train)

ada_pred = ada.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(y_test, ada_pred))
print(confusion_matrix(y_test, ada_pred))
print(classification_report(y_test, ada_pred))

# =========================
# 8. Stacking Classifier
# =========================
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)

print("Stacking Accuracy:", accuracy_score(y_test, stack_pred))
print(confusion_matrix(y_test, stack_pred))
print(classification_report(y_test, stack_pred))

# =========================
# 9. Accuracy Comparison
# =========================
models = ["Decision Tree", "Random Forest", "AdaBoost", "Stacking"]
accuracies = [
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, ada_pred),
    accuracy_score(y_test, stack_pred)
]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
