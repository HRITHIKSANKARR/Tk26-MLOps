import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Sentinel-1: Ethereum Fraud Detector", page_icon="🛡️")

st.title("🛡️ Sentinel-1: Ethereum Fraud Detector")
st.markdown("""
This app predicts whether an Ethereum wallet is fraudulent based on its behavioral transaction history.
**Model**: Random Forest Classifier (max_depth=5)
""")

# 1. Load the model artifact
@st.cache_resource
def load_model():
    try:
        return joblib.load('eth_fraud_model.pkl')
    except FileNotFoundError:
        st.error("Model artifact 'eth_fraud_model.pkl' not found. Please run the training pipeline first.")
        return None

model = load_model()

if model:
    # 2. User Input Sliders (4 Behavioral Pillars)
    st.sidebar.header("Wallet Behavioral Metrics")
    
    avg_min_sent = st.sidebar.slider("Velocity: Avg min between sent tnx", 0.0, 10000.0, 500.0)
    time_diff = st.sidebar.slider("Lifespan: Time Diff (Mins)", 0.0, 1000000.0, 10000.0)
    sent_tnx = st.sidebar.slider("Outflow: Sent tnx", 0, 10000, 10)
    received_tnx = st.sidebar.slider("Inflow: Received Tnx", 0, 10000, 20)
    
    # 3. Predict Button
    if st.button("Scan Wallet Strategy"):
        # Format input for prediction
        input_data = pd.DataFrame({
            'Avg min between sent tnx': [avg_min_sent],
            'Time Diff between first and last (Mins)': [time_diff],
            'Sent tnx': [sent_tnx],
            'Received Tnx': [received_tnx]
        })
        
        # Execute Prediction
        prediction = model.predict(input_data)[0]
        
        # Render Output
        st.subheader("Threat Assessment Result")
        if prediction == 0:
            st.success("✅ Safe Wallet: Behavior matches legitimate patterns.")
        else:
            st.error("🚨 Fraud Alert: High-risk behavioral patterns detected!")
            
    st.info("Note: This is a predictive assessment based on historical Kaggle data.")
else:
    st.warning("⚠️ Please ensure 'eth_fraud_model.pkl' is in the root directory.")
