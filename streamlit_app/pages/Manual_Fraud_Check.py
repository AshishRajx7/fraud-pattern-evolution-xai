import streamlit as st
import pandas as pd
import joblib

st.title('🖊️ Manual Fraud Check (All Features)')

model = joblib.load('models/fraud_detection_model.pkl')
scaler = joblib.load('models/scaler.pkl')

feature_list = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

input_values = {}

with st.expander("Enter Transaction Metadata (Time & Amount)", expanded=True):
    input_values['Time'] = st.number_input('Transaction Time (seconds)', min_value=0.0, value=10000.0)
    input_values['Amount'] = st.number_input('Transaction Amount ($)', min_value=0.01, value=50.0)

with st.expander("Enter PCA Transformed Features (V1 - V28)", expanded=False):
    for i in range(1, 29):
        input_values[f'V{i}'] = st.number_input(f'V{i}', value=0.0)

if st.button('Check for Fraud'):
    input_df = pd.DataFrame([input_values])

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f'🚨 Fraudulent Transaction Detected! Probability: {probability:.2f}')
    else:
        st.success(f'✅ Legitimate Transaction. Probability: {probability:.2f}')
