from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

with open("../data/processed/rf_fraud_model.pkl", "rb") as f: rf_fraud_model = pickle.load(f)
with open("../data/processed/rf_creditcard_model.pkl", "rb") as f: rf_creditcard_model = pickle.load(f)


fraud_data = pd.read_csv("../data/processed/fraud_data_engineered.csv")


@app.route('/', methods=['GET'])
def home():
  return jsonify({
    "nessage": "Fraud Detection API",
    "version": "1.0",
    "endpoints": {
      "/predict/fraud": "POST - Predict fraud for e-commerce transactions",
      "/predict/creditcard": "POST - Predict fraud for credit card transactions" 
    }
  })

@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
  try:
    #Get JSON data from request
    data = request.get_json()

    #Required fields for fraud prediction
    required_fields = ['user_id', 'purchase_value', 'source', 'browser', 'sex', 'age', 'country']

    #check if all required fields are present
    for fields in required_fields:
      if fields not in data:
        return jsonify({"error": f"Missing required field: {field}"}), 400

    #create a DataFrame from the input data
    input_df = pd.DataFrame([data])

    #select and prepare features (matching training data structure)
    feature_columns = ['user_id', 'purchase_value', 'age', 'source', 'browser', 'sex', 'country']
    input_features = input_df[feature_columns]


    categorical_features = ['source', 'browser', 'sex', 'country']
    numerical_features = ['user_id', 'purchase_value', 'age']

    #create dummy variables for categorical features
    input_encoded = pd.get_dummies(input_features, columns=categorical_features)

    expected_columns = 202 #Based on training data shape
    current_columns = input_encoded.shape[1]


    if current_columns < expected_columns:
      #pad with 0 for missing columns
      padding = pd.DataFrame(np.zeros((1, expected_columns - current_columns)))
      input_encoded = pd.concat([input_encoded, padding], axis=1)
    elif current_columns > expected_columns:
      input_encoded = input_encoded.iloc[:, :expected_columns]

    #make prediction
    prediction = rf_fraud_model.predict(input_encoded)[0]
    probability = rf_fraud_model.predict_proba(input_encoded)[0]

    return jsonify({
      "prediction": int(prediction), 
      "probability": {
        "non_fraud": float(probability[0]),
        "fraud": float(probability[1])
      },
      "risk_level": "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.3 else "Low"
    })

  except Exception as e:
    return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
  return jsonify({"status": "health", "message": "API is running"})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)






