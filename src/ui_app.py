# Streamlit UI for Fraud Detection API
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("🔍 Fraud Detection Interface")

API_BASE_URL = "http://127.0.0.1:5000"

st.sidebar.header("Choose Prediction Type")
option = st.sidebar.radio("Select the transaction type:", ("E-Commerce Fraud", "Credit Card Fraud"))

if option == "E-Commerce Fraud":
    st.header("🛒 E-Commerce Fraud Prediction")

    user_id = st.number_input("User ID", min_value=0, value=12345)
    purchase_value = st.number_input("Purchase Value ($)", min_value=0.0, value=100.0)
    source = st.selectbox("Source", ["SEO", "Ads", "Direct"])
    browser = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Edge"])
    sex = st.selectbox("Gender", ["M", "F"])
    age = st.slider("Age", 18, 100, 30)
    country = st.text_input("Country", value="US")

    if st.button("Predict E-Commerce Fraud"):
        payload = {
            "user_id": user_id,
            "purchase_value": purchase_value,
            "source": source,
            "browser": browser,
            "sex": sex,
            "age": age,
            "country": country
        }
        response = requests.post(f"{API_BASE_URL}/predict/fraud", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {'Fraud' if result['prediction'] == 1 else 'Not Fraud'}")
            st.metric("Fraud Probability", f"{result['probability']['fraud']:.2%}")
            st.metric("Risk Level", result['risk_level'])
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

elif option == "Credit Card Fraud":
    st.header("💳 Credit Card Fraud Prediction")

    cc_data = {}
    cc_data['Time'] = st.number_input("Time", value=0)
    for i in range(1, 29):
        cc_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")
    cc_data['Amount'] = st.number_input("Amount", value=149.62)

    if st.button("Predict Credit Card Fraud"):
        response = requests.post(f"{API_BASE_URL}/predict/creditcard", json=cc_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {'Fraud' if result['prediction'] == 1 else 'Not Fraud'}")
            st.metric("Fraud Probability", f"{result['probability']['fraud']:.2%}")
            st.metric("Risk Level", result['risk_level'])
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

