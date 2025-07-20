import pandas as pd
import numpy as np

fraud_data = pd.read_csv("../data/processed/fraud_data_cleaned.csv")
ip_country_data = pd.read_csv("../data/prcessed/ip_country_data_cleaned.csv")

#Convert ip_address in fraud_data to integer format
#Ensure ip_address is treated as string before conversion to handle potential float representations
fraud_data["ip_address_int"] = fraud_data["ip_address"].apply(lambda x: int(x) if pd.notna(x) else None)

#convert Ip address range in ip_country_data to integer format
ip_country_data["lower_bound_ip_address_int"] = ip_country_data["lower_bound_ip_address"].apply(lambda x: int(x) if pd.notna(x) else None)
ip_country_data["upper_bount_ip_address_int"] = ip_country_data["upper_bount_ip_address"].apply(lambda x: int(x) if pd.notna(x) else None)


#Sort ip_country_data by lower_bound_ip_address_int for efficient merging
ip_country_data = ip_country_data.sort_values(by="lower_bound_ip_address_int")

fraud_data_temp = fraud_data[['ip_address_int']].copy()
fraud_data_temp = fraud_data_temp.sort_values(by='ip_address_int')

merged_data = pd.merge_asof(fraud_data_temp, ip_country_data, left_on='ip_address_int', right_on='lower_bound_ip_address_int', direction='forward')


merged_data = merged_data[merged_data['ip_address_int'] <= merged_data['upper_bound_ip_address_int']]

fraud_data = pd.merge(fraud_data, merged_data[['ip_address_int', 'country']], on='ip_address_int', how='left')

fraud_data['country'] = fraud_data['country'].fillna('Unknown')

print("IP to country mapping complete.")

# Check the results
print("\nCountry distribution in Fraud_Data:")
print(fraud_data["country"].value_counts().head(10))

# Save the merged dataset
fraud_data.to_csv("../data/processed/fraud_data_with_country.csv", index=False)

print("\nMerged dataset saved as fraud_data_with_country.csv")

fraud_data['country']