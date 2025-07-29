import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data for Fraud_Data
with open("../data/processed/X_train_fraud.pkl", "rb") as f: X_train_fraud = pickle.load(f)
with open("../data/processed/X_test_fraud.pkl", "rb") as f: X_test_fraud = pickle.load(f)
with open("../data/processed/y_train_fraud.pkl", "rb") as f: y_train_fraud = pickle.load(f)
with open("../data/processed/y_test_fraud.pkl", "rb") as f: y_test_fraud = pickle.load(f)

# Load processed data for creditcard
with open("../data/processed/X_train_creditcard.pkl", "rb") as f: X_train_creditcard = pickle.load(f)
with open("../data/processed/X_test_creditcard.pkl", "rb") as f: X_test_creditcard = pickle.load(f)
with open("../data/processed/y_train_creditcard.pkl", "rb") as f: y_train_creditcard = pickle.load(f)
with open("../data/processed/y_test_creditcard.pkl", "rb") as f: y_test_creditcard = pickle.load(f)

# --- Logistic Regression for Fraud_Data.csv ---
print("\n--- Training Logistic Regression for Fraud_Data.csv ---")

# Initialize and train the Logistic Regression model
log_reg_fraud = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
log_reg_fraud.fit(X_train_fraud, y_train_fraud)

# Make predictions
y_pred_fraud = log_reg_fraud.predict(X_test_fraud)
y_proba_fraud = log_reg_fraud.predict_proba(X_test_fraud)[:, 1]

# Evaluate the model
print("\nClassification Report (Fraud_Data):")
print(classification_report(y_test_fraud, y_pred_fraud))

roc_auc_fraud = roc_auc_score(y_test_fraud, y_proba_fraud)
print(f"ROC AUC Score (Fraud_Data): {roc_auc_fraud:.4f}")

# Precision-Recall Curve and AUC-PR
precision_fraud, recall_fraud, _ = precision_recall_curve(y_test_fraud, y_proba_fraud)
auc_pr_fraud = auc(recall_fraud, precision_fraud)
print(f"AUC-PR Score (Fraud_Data): {auc_pr_fraud:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_fraud, precision_fraud, label=f'Logistic Regression (AUC-PR = {auc_pr_fraud:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Fraud_Data)')
plt.legend()
plt.grid(True)
plt.savefig("pr_curve_fraud_log_reg.png")
# plt.show()

# --- Logistic Regression for creditcard.csv ---
print("\n--- Training Logistic Regression for creditcard.csv ---")

# Initialize and train the Logistic Regression model
log_reg_creditcard = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
log_reg_creditcard.fit(X_train_creditcard, y_train_creditcard)

# Make predictions
y_pred_creditcard = log_reg_creditcard.predict(X_test_creditcard)
y_proba_creditcard = log_reg_creditcard.predict_proba(X_test_creditcard)[:, 1]

# Evaluate the model
print("\nClassification Report (creditcard):")
print(classification_report(y_test_creditcard, y_pred_creditcard))

roc_auc_creditcard = roc_auc_score(y_test_creditcard, y_proba_creditcard)
print(f"ROC AUC Score (creditcard): {roc_auc_creditcard:.4f}")

# Precision-Recall Curve and AUC-PR
precision_creditcard, recall_creditcard, _ = precision_recall_curve(y_test_creditcard, y_proba_creditcard)
auc_pr_creditcard = auc(recall_creditcard, precision_creditcard)
print(f"AUC-PR Score (creditcard): {auc_pr_creditcard:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_creditcard, precision_creditcard, label=f'Logistic Regression (AUC-PR = {auc_pr_creditcard:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (creditcard)')
plt.legend()
plt.grid(True)
plt.savefig("pr_curve_creditcard_log_reg.png")
# plt.show()

print("\nLogistic Regression model training and evaluation complete for both datasets.")

# Save the trained models
with open("log_reg_fraud_model.pkl", "wb") as f: pickle.dump(log_reg_fraud, f)
with open("log_reg_creditcard_model.pkl", "wb") as f: pickle.dump(log_reg_creditcard, f)

print("Trained Logistic Regression models saved to log_reg_fraud_model.pkl and log_reg_creditcard_model.pkl.")


