import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle

# Load the engineered fraud data
fraud_data = pd.read_csv("/content/drive/MyDrive/10Acadamy/data/processed/fraud_data_engineered.csv")
creditcard_data = pd.read_csv("/content/drive/MyDrive/10Acadamy/data/processed/creditcard_data_cleaned.csv")

# --- Data Preparation for Fraud_Data.csv ---
print("\n--- Data Preparation for Fraud_Data.csv ---")

# Separate features and target
X_fraud = fraud_data.drop(columns=["class", "signup_time", "purchase_time", "ip_address", "ip_address_int", "device_id"])
y_fraud = fraud_data["class"]

# Identify categorical and numerical features
categorical_features_fraud = X_fraud.select_dtypes(include=["object"]).columns
numerical_features_fraud = X_fraud.select_dtypes(include=["int64", "float64", "int32"]).columns

# One-Hot Encode Categorical Features
if len(categorical_features_fraud) > 0:
    encoder_fraud = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_fraud_encoded = encoder_fraud.fit_transform(X_fraud[categorical_features_fraud])
    X_fraud_encoded_df = pd.DataFrame(X_fraud_encoded, columns=encoder_fraud.get_feature_names_out(categorical_features_fraud))
    X_fraud_processed = pd.concat([X_fraud[numerical_features_fraud].reset_index(drop=True), X_fraud_encoded_df], axis=1)
else:
    X_fraud_processed = X_fraud[numerical_features_fraud].reset_index(drop=True)

# Train-Test Split
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud_processed, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud)

print(f"Original Fraud_Data training set shape: {X_train_fraud.shape}, Target distribution: {Counter(y_train_fraud)}")

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_fraud_resampled, y_train_fraud_resampled = smote.fit_resample(X_train_fraud, y_train_fraud)

print(f"Resampled Fraud_Data training set shape: {X_train_fraud_resampled.shape}, Target distribution: {Counter(y_train_fraud_resampled)}")

# Normalization and Scaling
scaler_fraud = StandardScaler()
X_train_fraud_scaled = scaler_fraud.fit_transform(X_train_fraud_resampled)
X_test_fraud_scaled = scaler_fraud.transform(X_test_fraud)

# Save processed data for Fraud_Data
with open("../data/processed/X_train_fraud.pkl", "wb") as f: pickle.dump(X_train_fraud_scaled, f)
with open("../data/processed/X_test_fraud.pkl", "wb") as f: pickle.dump(X_test_fraud_scaled, f)
with open("../data/processed/y_train_fraud.pkl", "wb") as f: pickle.dump(y_train_fraud_resampled, f)
with open("../data/processed/y_test_fraud.pkl", "wb") as f: pickle.dump(y_test_fraud, f)

print("Fraud_Data preparation complete. Processed data saved to pickle files.")

# --- Data Preparation for creditcard.csv ---
print("\n--- Data Preparation for creditcard.csv ---")

# Separate features and target
X_creditcard = creditcard_data.drop(columns=["Class"])
y_creditcard = creditcard_data["Class"]

# Train-Test Split
X_train_creditcard, X_test_creditcard, y_train_creditcard, y_test_creditcard = train_test_split(X_creditcard, y_creditcard, test_size=0.3, random_state=42, stratify=y_creditcard)

print(f"Original creditcard training set shape: {X_train_creditcard.shape}, Target distribution: {Counter(y_train_creditcard)}")

# Handle Class Imbalance using SMOTE
smote_creditcard = SMOTE(random_state=42)
X_train_creditcard_resampled, y_train_creditcard_resampled = smote_creditcard.fit_resample(X_train_creditcard, y_train_creditcard)

print(f"Resampled creditcard training set shape: {X_train_creditcard_resampled.shape}, Target distribution: {Counter(y_train_creditcard_resampled)}")

# Normalization and Scaling
scaler_creditcard = StandardScaler()
X_train_creditcard_scaled = scaler_creditcard.fit_transform(X_train_creditcard_resampled)
X_test_creditcard_scaled = scaler_creditcard.transform(X_test_creditcard)

# Save processed data for creditcard
with open("../data/processed/X_train_creditcard.pkl", "wb") as f: pickle.dump(X_train_creditcard_scaled, f)
with open("../data/processed/X_test_creditcard.pkl", "wb") as f: pickle.dump(X_test_creditcard_scaled, f)
with open("../data/processed/y_train_creditcard.pkl", "wb") as f: pickle.dump(y_train_creditcard_resampled, f)
with open("../data/processed/y_test_creditcard.pkl", "wb") as f: pickle.dump(y_test_creditcard, f)

print("Creditcard data preparation complete. Processed data saved to pickle files.")

print("All data prepared and saved for model building.")
