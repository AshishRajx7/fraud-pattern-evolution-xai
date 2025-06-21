import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title('📊 Model Explainability with SHAP')

model = joblib.load('models/fraud_detection_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.write("""
Upload a CSV file to visualize SHAP feature importance and understand what drives the model's decisions.
""")

uploaded_file = st.file_uploader("Upload a Transaction File (CSV) for SHAP Analysis", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'Class' in data.columns:
        data_features = data.drop('Class', axis=1)
    else:
        data_features = data

    X = scaler.transform(data_features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.write("### SHAP Summary Plot:")
    shap.summary_plot(shap_values[1], pd.DataFrame(X, columns=data_features.columns), plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
