import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from data_loader import load_data, preprocess_data

def explain_model(file_path, model_path='models/fraud_detection_model.pkl'):
    # Load the trained model
    model = joblib.load(model_path)

    # Load and preprocess the data
    data = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    # Plot summary
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values[1], X_test, show=True)

if __name__ == "__main__":
    explain_model('data/raw/creditcard.csv')
