# streamlit_app/app.py

import streamlit as st

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Fraud Detection Dashboard")

st.write("""
Welcome to the **Fraud Detection System with Explainable AI (XAI)**.

👉 Use the sidebar to navigate between:
- 📂 File Upload & Predictions
- 🖊️ Manual Fraud Check
- 📊 Model Explainability
""")
