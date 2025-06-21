import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title('📂 File Upload and Predictions')

# Load Model and Scaler
model = joblib.load('models/fraud_detection_model.pkl')
scaler = joblib.load('models/scaler.pkl')

uploaded_file = st.file_uploader("Upload a Transaction File (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Sample:")
    st.write(data.head())

    if 'Class' in data.columns:
        data_features = data.drop('Class', axis=1)
    else:
        data_features = data

    X = scaler.transform(data_features)

    predictions = model.predict(X)
    prediction_probs = model.predict_proba(X)[:, 1]

    data['Fraud Probability'] = prediction_probs
    data['Prediction'] = predictions

    st.write("### Predictions:")
    st.write(data[['Fraud Probability', 'Prediction']])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.write("### SHAP Feature Importance (Bar Plot):")
    shap.summary_plot(shap_values[1], pd.DataFrame(X, columns=data_features.columns), plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
