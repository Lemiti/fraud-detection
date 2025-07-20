import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fraud_data = pd.read_csv("../data/processed/fraud_data_cleaned.csv")
creditcard_data = pd.read_csv('../data/processed/creditcard_data_cleaned.csv')

# --- EDA for Fraud_Data.csv --- 

print("\n--- Exploratory Data Analysis for Fraud_Data.csv ---")

# Univariate Analysis
print('\nValue counts for catagorical features in Fraud_Data:')
for col in ["source", "browser", "sex", "class"]:
  print(f"\n{col}:")
  print(fraud_data[col].value_counts())


print("\nDescriptive statistics for numerical features in Fraud_Data:")
print(fraud_data[["purchase_value", "age", "ip_address"]].describe())

#Bivariate Analysis (example: fraud rate by source)
plt.figure(figsize=(10, 6))
sns.countplot(data=fraud_data, x="source", hue="class")
plt.title("Fraud Count by Source")
plt.savefig("notebooks/fraud_count_by_source.png")
#plt.show()


plt.figure(figsize=(10,6))
sns.countplot(data=fraud_data, x="browser", hue="class")
plt.title("Fraud Count by Browser")
plt.savefig("notebooks/fraud_count_by_browser.png")
#plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data="fraud_data", x="class", y="purchase_value")
plt.title("Purchase Value Distribution by class")
plt.savafig("notebooks/purchase_value_by_class.png")
#plt.show()

#convert time columns to datetime objects for time-based analysis
fraud_data["signup_time"] = pd.to_datetime(fraud_data["signup_time"])
fraud_data["purchase_time"] = pd.to_datetime(fraud_data["purchase_time"])

#calculate time difference between signup and purchase
fraud_data["time_to_purchase"] = (fraud_data["purchase_time"] - fraud_data["signup_time"]).dt.total_seconds()/3600


plt.figure(figsize=(10,6))
sns.histplot(data=fraud_data, x="time_to_purchase", hue="class", kde=True)
plt.title("Time to Purchase Distribution by Class")
plt.savefig("notebooks/time_to_purchase_distribution.png")
#plt.show()


# --- EDA for creditcard.csv ---
print("\n--- EDA for creditcard.csv ---")

#univariate Analysis
print('\nDescriptive statistics for numerical features in creditcard_data:')
print(creditcard_data[["Time", "Amount"]].describe())


print("\nValue counts for Class in creditcards_data:")
print(creditcard_data["Class"].value_counts())

#Bivariate Analysis
plt.figure(figsize=(10, 6))
plt.boxplot(data=creditcard_data, x="Class", y="Amount")
plt.title("Transaction Amount Distribution by Class")
plt.ylim(0, 2000) #Limit y-axis for better visualization of non-fraudent transactions
plt.savefig("notebooks/creditcard_amount_by_class.png")
#plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=creditcard_data, x="Time", hue="Class", bins=50, kde=True)
plt.title("Transaction Time Distribution by Class")
plt.savefig("notebooks/creditcard_time_distribution.png")

print("EDA complete. Plots saved in notebooks as PNG files.")
