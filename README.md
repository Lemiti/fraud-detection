# Fraud Detection for E-Commerce and Bank Transactions

![Fraud Detection](https://img.shields.io/badge/Status-Not_Completed-orange) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![License](https://img.shields.io/badge/License-MIT-green)

## **Project Overview**
This project aims to improve fraud detection for **Adey Innovations Inc.** using machine learning. It analyzes e-commerce (`Fraud_Data.csv`) and credit card (`creditcard.csv`) transactions, leveraging geolocation (`IpAddress_to_Country.csv`) and behavioral features to identify fraudulent activities.

**Key Goals:**
- Reduce false positives/negatives.
- Balance security and user experience.
- Explain model decisions using SHAP.

## **Datasets**
1. **Fraud_Data.csv**: E-commerce transactions (`user_id`, `purchase_value`, `ip_address`, etc.).
2. **creditcard.csv**: Anonymized credit card transactions (PCA features `V1-V28`).
3. **IpAddress_to_Country.csv**: Maps IP ranges to countries.

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/Lemiti/fraud-detection.git
   cd fraud-detection
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
   
