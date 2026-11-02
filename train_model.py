import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

sns.set(style="white")

# 1. Load Dataset
dataset = pd.read_csv("iris.csv")

# 2. Clean Column Names
dataset.columns = (
    dataset.columns
    .str.replace(" (cm)", "", regex=False)
    .str.strip()
    .str.replace(" ", "_")
)

# 3. Feature Engineering
dataset["sepal_length_width_ratio"] = (
    dataset["sepal_length"] / dataset["sepal_width"]
)

dataset["petal_length_width_ratio"] = (
    dataset["petal_length"] / dataset["petal_width"]
)

# 4. Select Features
X = dataset.drop("target", axis=1)
y = dataset["target"]

# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=44,
    stratify=y
)

# 6. Logistic Regression
logreg = LogisticRegression(
    C=0.0001,
    solver="lbfgs",
    max_iter=1000
)

logreg.fit(X_train, y_train)

predictions_lr = logreg.predict(X_test)

train_acc_lr = accuracy_score(
    y_train, logreg.predict(X_train)
) * 100

test_acc_lr = accuracy_score(
    y_test, predictions_lr
) * 100

f1_lr = f1_score(y_test, predictions_lr, average="weighted")
precision_lr = precision_score(y_test, predictions_lr, average="weighted")
recall_lr = recall_score(y_test, predictions_lr, average="weighted")

# 7. Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=44
)

rf_model.fit(X_train, y_train)

predictions_rf = rf_model.predict(X_test)

train_acc_rf = accuracy_score(
    y_train, rf_model.predict(X_train)
) * 100

test_acc_rf = accuracy_score(
    y_test, predictions_rf
) * 100

f1_rf = f1_score(y_test, predictions_rf, average="weighted")
precision_rf = precision_score(y_test, predictions_rf, average="weighted")
recall_rf = recall_score(y_test, predictions_rf, average="weighted")

# 8. Generate Confusion Matrix
cm = confusion_matrix(y_test, predictions_lr)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Setosa", "Versicolor", "Virginica"],
    yticklabels=["Setosa", "Versicolor", "Virginica"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig("ConfusionMatrix.png", dpi=150)
plt.close()

# 9. Feature Importance
importances = rf_model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_df
)

plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig("FeatureImportance.png", dpi=150)
plt.close()

# 10. Save Model Accuracy and Evaluation Metrics
with open("scores.txt", "w") as score:

    score.write("MACHINE LEARNING MODEL EVALUATION\n")
    score.write("=" * 45 + "\n\n")

    score.write("LOGISTIC REGRESSION\n")
    score.write("-" * 30 + "\n")
    score.write(f"Training Accuracy: {train_acc_lr:.2f}%\n")
    score.write(f"Testing Accuracy: {test_acc_lr:.2f}%\n")
    score.write(f"F1 Score: {f1_lr:.4f}\n")
    score.write(f"Precision: {precision_lr:.4f}\n")
    score.write(f"Recall: {recall_lr:.4f}\n\n")

    score.write("RANDOM FOREST CLASSIFIER\n")
    score.write("-" * 30 + "\n")
    score.write(f"Training Accuracy: {train_acc_rf:.2f}%\n")
    score.write(f"Testing Accuracy: {test_acc_rf:.2f}%\n")
    score.write(f"F1 Score: {f1_rf:.4f}\n")
    score.write(f"Precision: {precision_rf:.4f}\n")
    score.write(f"Recall: {recall_rf:.4f}\n")

print("Model Training Completed Successfully!")
print("\nLogistic Regression Testing Accuracy:", f"{test_acc_lr:.2f}%")
print("Random Forest Testing Accuracy:", f"{test_acc_rf:.2f}%")
print("\nScores saved successfully in scores.txt")
print("Confusion Matrix generated successfully.")
print("Feature Importance generated successfully.")
