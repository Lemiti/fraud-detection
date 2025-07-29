import pandas as pd

fraud_data = pd.read_csv('../data/Fraud_Data.csv')
creditcard_data = pd.read_csv('../data/creditcard.csv')
ip_country_data = pd.read_csv('../data/IpAddress_to_Country.csv')

print('Fraude_Data.csv info:')
fraud_data.info()
print('\nMissing values in Fraud_Data.csv:')
print(fraud_data.isnull().sum())


print('\ncreditcard.csv info:')
creditcard_data.info()
print('\nMissing values in creditcard.csv')
print(creditcard_data.isnull().sum())


print('\nIpAddress_to_Country.csv info:')
ip_country_data.info()
print('\nMissing values in IpAddress_to_Country.csv:')
print(ip_country_data.isnull().sum())

fraud_data_cleaned = fraud_data.dropna()

fraud_data_cleaned = fraud_data.drop_duplicates()

print('\nFraud_Data.csv after cleaning')
fraud_data_cleaned.info()


creditcard_data_cleaned = creditcard_data.dropna()

creditcard_data_cleaned = creditcard_data.drop_duplicates()
print('\ncreditcard.csv after cleaning:')

ip_country_data_cleaned = ip_country_data.drop_duplicates()
print('\nIpAddress_to_Country.csv after cleaning:')
ip_country_data_cleaned.info()

fraud_data_cleaned.to_csv('../data/processed/fraud_data_cleand.csv', index=False)
creditcard_data_cleaned.to_csv('../data/processed/creditcard_data_cleaned.csv', index=False)
ip_country_data_cleaned.to_csv('../data/processed/ip_country_data_cleaned.csv', index=False)

print('\nData cleaning and missing value handling complete. Cleaned data saved to fraud_data_cleaned.csv, creditcard_data_cleaned.csv, and ip_country_data_cleaned.csv.')