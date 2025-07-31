import requests
import json

base_url = "http://127.0.0.1:5000"

def test_health_endpoint():
    response = requests.get(f"{base_url}/health")
    print("Health Endpoint Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_home_endpoint():
    response = requests.get(f"{base_url}/")
    print("Home Endpoint Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_fraud_prediction():
    fraud_data = {
        "user_id": 12345,
        "purchase_value": 150,
        "source": "SEO",
        "browser": "Chrome",
        "sex": "M",
        "age": 30,
        "country": "US"
    }

    response = requests.post(f"{base_url}/predict/fraud", json=fraud_data)
    print("Fraud Prediction Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Request Data: {fraud_data}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_creditcard_prediction():
    creditcard_data = {
        "Time": 0,
        "Amount": 149.62
    }

    for i in range(1, 29):
        creditcard_data[f"V{i}"] = 0.0

    response = requests.post(f"{base_url}/predict/creditcard", json=creditcard_data)
    print("Credit Card Prediction Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_missing_fields():
    incomplete_data = {
        "user_id": 12345,
        "purchase_value": 150
    }

    response = requests.post(f"{base_url}/predict/fraud", json=incomplete_data)
    print("Missing Fields Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

if __name__ == "__main__":
    print("Testing Fraud Detection API")
    print("=" * 50)

    try:
        test_health_endpoint()
        test_home_endpoint()
        test_fraud_prediction()
        test_creditcard_prediction()
        test_missing_fields()

        print("All tests completed!")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the Flask server is running.")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

