import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns

with open("../data/processed/X_train_fraud.pkl", "rb") as f: X_train_fraud = pickle.load(f)
with open("../data/processed/X_test_fraud.pkl", "rb") as f: X_test_fraud = pickle.load(f)
with open("../data/processed/y_train_fraud.pkl", "rb") as f: y_train_fraud = pickle.load(f)
with open("../data/processed/y_test_fraud.pkl", "rb") as f: y_test_fraud = pickle.load(f)

# Load processed data for creditcard
with open("../data/processed/X_train_creditcard.pkl", "rb") as f: X_train_creditcard = pickle.load(f)
with open("../data/processed/X_test_creditcard.pkl", "rb") as f: X_test_creditcard = pickle.load(f)
with open("../data/processed/y_train_creditcard.pkl", "rb") as f: y_train_creditcard = pickle.load(f)
with open("../data/processed/y_test_creditcard.pkl", "rb") as f: y_test_creditcard = pickle.load(f)

# --- Random Forest for Fraud_Data.csv ---
print("\n--- Training Random Forest for Fraud_Data.csv ---")

# Initialize and train the Random Forest model
rf_fraud = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_fraud.fit(X_train_fraud, y_train_fraud)

# Make predictions
y_pred_fraud_rf = rf_fraud.predict(X_test_fraud)
y_proba_fraud_rf = rf_fraud.predict_proba(X_test_fraud)[:, 1]

# Evaluate the model
print("\nClassification Report (Fraud_Data - Random Forest):")
print(classification_report(y_test_fraud, y_pred_fraud_rf))

roc_auc_fraud_rf = roc_auc_score(y_test_fraud, y_proba_fraud_rf)
print(f"ROC AUC Score (Fraud_Data - Random Forest): {roc_auc_fraud_rf:.4f}")

# Precision-Recall Curve and AUC-PR
precision_fraud_rf, recall_fraud_rf, _ = precision_recall_curve(y_test_fraud, y_proba_fraud_rf)
auc_pr_fraud_rf = auc(recall_fraud_rf, precision_fraud_rf)
print(f"AUC-PR Score (Fraud_Data - Random Forest): {auc_pr_fraud_rf:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_fraud_rf, precision_fraud_rf, label=f"Random Forest (AUC-PR = {auc_pr_fraud_rf:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Fraud_Data - Random Forest)")
plt.legend()
plt.grid(True)
plt.savefig("pr_curve_fraud_rf.png")
# plt.show()

# --- Random Forest for creditcard.csv ---
print("\n--- Training Random Forest for creditcard.csv ---")

# Initialize and train the Random Forest model
rf_creditcard = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_creditcard.fit(X_train_creditcard, y_train_creditcard)

# Make predictions
y_pred_creditcard_rf = rf_creditcard.predict(X_test_creditcard)
y_proba_creditcard_rf = rf_creditcard.predict_proba(X_test_creditcard)[:, 1]

# Evaluate the model
print("\nClassification Report (creditcard - Random Forest):")
print(classification_report(y_test_creditcard, y_pred_creditcard_rf))

roc_auc_creditcard_rf = roc_auc_score(y_test_creditcard, y_proba_creditcard_rf)
print(f"ROC AUC Score (creditcard - Random Forest): {roc_auc_creditcard_rf:.4f}")

# Precision-Recall Curve and AUC-PR
precision_creditcard_rf, recall_creditcard_rf, _ = precision_recall_curve(y_test_creditcard, y_proba_creditcard_rf)
auc_pr_creditcard_rf = auc(recall_creditcard_rf, precision_creditcard_rf)
print(f"AUC-PR Score (creditcard - Random Forest): {auc_pr_creditcard_rf:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_creditcard_rf, precision_creditcard_rf, label=f"Random Forest (AUC-PR = {auc_pr_creditcard_rf:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (creditcard - Random Forest)")
plt.legend()
plt.grid(True)
plt.savefig("pr_curve_creditcard_rf.png")
# plt.show()

print("\nRandom Forest model training and evaluation complete for both datasets.")

# Save the trained models
with open("../data/processed/rf_fraud_model.pkl", "wb") as f: pickle.dump(rf_fraud, f)
with open("../data/processed/rf_creditcard_model.pkl", "wb") as f: pickle.dump(rf_creditcard, f)

print("Trained Random Forest models saved to rf_fraud_model.pkl and rf_creditcard_model.pkl.")