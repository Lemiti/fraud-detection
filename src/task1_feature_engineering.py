import pandas as pd

# Load the merged fraud data
fraud_data = pd.read_csv("../data/processed/fraud_data_with_country.csv")

# Convert time columns to datetime objects
fraud_data["signup_time"] = pd.to_datetime(fraud_data["signup_time"])
fraud_data["purchase_time"] = pd.to_datetime(fraud_data["purchase_time"])

# 1. Time-Based features for Fraud_Data.csv
fraud_data["hour_of_day"] = fraud_data["purchase_time"].dt.hour
fraud_data["day_of_week"] = fraud_data["purchase_time"].dt.dayofweek # Monday=0, Sunday=6

# 2. time_since_signup: Calculate the duration between signup_time and purchase_time.
fraud_data["time_since_signup"] = (fraud_data["purchase_time"] - fraud_data["signup_time"]).dt.total_seconds()

# 3. Transaction frequency and velocity for Fraud_Data.csv
# This often requires a time window. Let's define a simple window, e.g., 24 hours.
# For simplicity, let's calculate frequency per user and device within a rolling window.

# Sort by user_id and purchase_time for correct rolling window calculation
fraud_data = fraud_data.sort_values(by=["user_id", "purchase_time"])

# Transaction frequency (e.g., number of transactions by user in last 24 hours)
# This is a more complex feature and might require a loop or specialized function for accurate rolling window.
# For now, let's calculate a simpler frequency: total transactions per user/device.

# Number of transactions per user
user_transaction_count = fraud_data.groupby("user_id")["purchase_time"].transform("count")
fraud_data["user_transaction_frequency"] = user_transaction_count

# Number of transactions per device
device_transaction_count = fraud_data.groupby("device_id")["purchase_time"].transform("count")
fraud_data["device_transaction_frequency"] = device_transaction_count

# Transaction velocity (e.g., sum of purchase values per user/device in last 24 hours)
# Similar to frequency, this is complex for rolling windows. Let's use total value for now.

# Total purchase value per user
user_purchase_value_sum = fraud_data.groupby("user_id")["purchase_value"].transform("sum")
fraud_data["user_purchase_velocity"] = user_purchase_value_sum

# Total purchase value per device
device_purchase_value_sum = fraud_data.groupby("device_id")["purchase_value"].transform("sum")
fraud_data["device_purchase_velocity"] = device_purchase_value_sum

print("Feature engineering complete for Fraud_Data.csv.")

# Display info of the updated dataframe
print("\nFraud_Data.csv after feature engineering:")
fraud_data.info()

# Save the dataframe with new features
fraud_data.to_csv("../data/processed/fraud_data_engineered.csv", index=False)

print("\nEngineered fraud data saved as fraud_data_engineered.csv")
