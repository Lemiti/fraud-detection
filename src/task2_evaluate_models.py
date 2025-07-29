import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


with open("../data/processed/X_test_fraud.pkl", "rb") as f: X_test_fraud = pickle.load(f)
with open("../data/processed/y_test_fraud.pkl", "rb") as f: y_test_fraud = pickle.load(f)

# Load test data for creditcard
with open("../data/processed/X_test_creditcard.pkl", "rb") as f: X_test_creditcard = pickle.load(f)
with open("../data/processed/y_test_creditcard.pkl", "rb") as f: y_test_creditcard = pickle.load(f)

# Load trained models
with open("../data/processed/log_reg_fraud_model.pkl", "rb") as f: log_reg_fraud = pickle.load(f)
with open("../data/processed/rf_fraud_model.pkl", "rb") as f: rf_fraud = pickle.load(f)
with open("../data/processed/log_reg_creditcard_model.pkl", "rb") as f: log_reg_creditcard = pickle.load(f)
with open("../data/processed/rf_creditcard_model.pkl", "rb") as f: rf_creditcard = pickle.load(f)

# --- Evaluate and Compare Models for Fraud_Data.csv ---
print("\n--- Evaluating and Comparing Models for Fraud_Data.csv ---")

# Get predictions and probabilities for Logistic Regression (Fraud_Data)
y_pred_fraud_lr = log_reg_fraud.predict(X_test_fraud)
y_proba_fraud_lr = log_reg_fraud.predict_proba(X_test_fraud)[:, 1]

# Get predictions and probabilities for Random Forest (Fraud_Data)
y_pred_fraud_rf = rf_fraud.predict(X_test_fraud)
y_proba_fraud_rf = rf_fraud.predict_proba(X_test_fraud)[:, 1]

# Calculate metrics for Logistic Regression (Fraud_Data)
roc_auc_fraud_lr = roc_auc_score(y_test_fraud, y_proba_fraud_lr)
precision_fraud_lr, recall_fraud_lr, _ = precision_recall_curve(y_test_fraud, y_proba_fraud_lr)
auc_pr_fraud_lr = auc(recall_fraud_lr, precision_fraud_lr)

# Calculate metrics for Random Forest (Fraud_Data)
roc_auc_fraud_rf = roc_auc_score(y_test_fraud, y_proba_fraud_rf)
precision_fraud_rf, recall_fraud_rf, _ = precision_recall_curve(y_test_fraud, y_proba_fraud_rf)
auc_pr_fraud_rf = auc(recall_fraud_rf, precision_fraud_rf)

print("\nFraud_Data Model Comparison:")
print("----------------------------")
print(f"Logistic Regression: ROC AUC = {roc_auc_fraud_lr:.4f}, AUC-PR = {auc_pr_fraud_lr:.4f}")
print(f"Random Forest:       ROC AUC = {roc_auc_fraud_rf:.4f}, AUC-PR = {auc_pr_fraud_rf:.4f}")

# Determine best model for Fraud_Data based on AUC-PR (more suitable for imbalanced data)
best_model_fraud = "Random Forest" if auc_pr_fraud_rf > auc_pr_fraud_lr else "Logistic Regression"
print(f"Best model for Fraud_Data: {best_model_fraud}")

# Plot combined Precision-Recall Curve for Fraud_Data
plt.figure(figsize=(10, 7))
plt.plot(recall_fraud_lr, precision_fraud_lr, label=f"Logistic Regression (AUC-PR = {auc_pr_fraud_lr:.2f})")
plt.plot(recall_fraud_rf, precision_fraud_rf, label=f"Random Forest (AUC-PR = {auc_pr_fraud_rf:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (Fraud_Data)")
plt.legend()
plt.grid(True)
plt.savefig("notebooks/pr_curves_fraud_comparison.png")
# plt.show()

# --- Evaluate and Compare Models for creditcard.csv ---
print("\n--- Evaluating and Comparing Models for creditcard.csv ---")

# Get predictions and probabilities for Logistic Regression (creditcard)
y_pred_creditcard_lr = log_reg_creditcard.predict(X_test_creditcard)
y_proba_creditcard_lr = log_reg_creditcard.predict_proba(X_test_creditcard)[:, 1]

# Get predictions and probabilities for Random Forest (creditcard)
y_pred_creditcard_rf = rf_creditcard.predict(X_test_creditcard)
y_proba_creditcard_rf = rf_creditcard.predict_proba(X_test_creditcard)[:, 1]

# Calculate metrics for Logistic Regression (creditcard)
roc_auc_creditcard_lr = roc_auc_score(y_test_creditcard, y_proba_creditcard_lr)
precision_creditcard_lr, recall_creditcard_lr, _ = precision_recall_curve(y_test_creditcard, y_proba_creditcard_lr)
auc_pr_creditcard_lr = auc(recall_creditcard_lr, precision_creditcard_lr)

# Calculate metrics for Random Forest (creditcard)
roc_auc_creditcard_rf = roc_auc_score(y_test_creditcard, y_proba_creditcard_rf)
precision_creditcard_rf, recall_creditcard_rf, _ = precision_recall_curve(y_test_creditcard, y_proba_creditcard_rf)
auc_pr_creditcard_rf = auc(recall_creditcard_rf, precision_creditcard_rf)

print("\ncreditcard Model Comparison:")
print("----------------------------")
print(f"Logistic Regression: ROC AUC = {roc_auc_creditcard_lr:.4f}, AUC-PR = {auc_pr_creditcard_lr:.4f}")
print(f"Random Forest:       ROC AUC = {roc_auc_creditcard_rf:.4f}, AUC-PR = {auc_pr_creditcard_rf:.4f}")

# Determine best model for creditcard based on AUC-PR
best_model_creditcard = "Random Forest" if auc_pr_creditcard_rf > auc_pr_creditcard_lr else "Logistic Regression"
print(f"Best model for creditcard: {best_model_creditcard}")

# Plot combined Precision-Recall Curve for creditcard
plt.figure(figsize=(10, 7))
plt.plot(recall_creditcard_lr, precision_creditcard_lr, label=f"Logistic Regression (AUC-PR = {auc_pr_creditcard_lr:.2f})")
plt.plot(recall_creditcard_rf, precision_creditcard_rf, label=f"Random Forest (AUC-PR = {auc_pr_creditcard_rf:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (creditcard)")
plt.legend()
plt.grid(True)
plt.savefig("notebooks/pr_curves_creditcard_comparison.png")
# plt.show()

print("\nModel evaluation and comparison complete. Comparison plots saved.")


