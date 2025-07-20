import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter


fraud_data = pd.read_csv("../data/processed/fraud_data_engineered.csv")
creditcard_data = pd.read_csv("../data/processed/creditcard_data_cleaned.csv")


print("--- Data Transformation for fraud_data.csv ---")

X_fraud = fraud_data.drop(columns=['class', 'signup_time', 'purchase_time', 'ip_address', 'ip_address_int', 'device_id'])
y_fraud = fraud_data['class']

print(f"Fraud data shape after dropping columns: {X_fraud.shape}")

categorical_features_fraud = X_fraud.select_dtypes(include=['object']).columns
numerical_features_fraud = X_fraud.select_dtypes(include=['int64', 'float64', 'int32']).columns

print(f"Catagorical features: {list(categorical_features_fraud)}")
print(f"Numerical features: {list(numerical_features_fraud)}")


if len(categorical_features_fraud) > 0:
  encoder_fraud = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
  X_fraud_encoded = encoder_fraud.fit_transorm(X_fraud[categorical_features_fraud])
  X_fraud_encoded_df = pd.DataFrame(X_fraud_encoded, columns=encoder_fraud.get_feature_names_out(categorical_features_fraud))

  X_fraud_processed = pd.concat([X_fraud[numerical_features_fraud].reset_index(drop=True), X_fraud_encoded_df], axis=1)
else:
  X_fraud_processed = X_fraud[numerical_features_fraud].reset_index(drop=True)

print(f"Processed fraud data shape: {X_fraud_processed.shape}")

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

print("Fraud_Data transformation complete.")

print("\n--- Data Transformation for creditcard.csv ---")

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

print("Creditcard data transformation complete.")

print("\nData transformation for both datasets complete. Data is ready for model building.")
