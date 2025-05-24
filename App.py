import streamlit as st
import numpy as np
import joblib

# Set Streamlit page config
st.set_page_config(
    page_title="Binance Close Price Predictor",
    page_icon="📈",
    layout="centered"
)

# Load model
model = joblib.load("bitcoin_predictor.pkl")

# Title and description
st.title("📈 Binance Close Price Predictor")
st.caption("Estimate the closing price of Binance Coin (BNB) based on market data.")

# Layout: side-by-side inputs
with st.form("prediction_form"):
    st.subheader("🔢 Input Features")

    col1, col2 = st.columns(2)
    with col1:
        high = st.number_input("High", min_value=0.0, value=653.40, format="%.6f")
        low = st.number_input("Low", min_value=0.0, value=640.40, format="%.6f")
        open_price = st.number_input("Open", min_value=0.0, value=649.85, format="%.6f")
        Close = st.number_input('Previous Close price', min_value=0.0, value=740.80, format="%.6f")

    with col2:
        volume = st.number_input("Volume", min_value=0.0, value=8_280_000.0, format="%.2f")
        marketcap = st.number_input("Market Cap", min_value=0.0, value=107_300_000_000.0, format="%.2f")

    predict_button = st.form_submit_button("🔍 Predict")

# Prediction and result
if predict_button:
    input_data = np.array([[high, low, open_price,Close, volume, marketcap]])
    prediction = model.predict(input_data)

    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    st.metric(label="Predicted Close Price (USD)", value=f"${prediction[0]:,.2f}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Powered by scikit-learn")
