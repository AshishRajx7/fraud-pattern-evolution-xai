# Fraud Pattern Evolution with Explainable AI (XAI)

A real-time fraud detection system with concept drift detection and SHAP-based model explainability.  
This project solves not just fraud classification but also addresses the changing nature of fraud patterns over time — a critical challenge in the financial industry.

---
## Live Demo
[Click here to access the live Streamlit app](https://fraud-pattern-evolution-xai-ct9fm6wfekxsygbmz65zup.streamlit.app/)

---

## Project Overview

Traditional fraud detection models often fail to adapt to evolving fraud patterns.

This project provides an end-to-end solution that:
- Detects fraudulent financial transactions using machine learning
- Continuously monitors for concept drift (changes in fraud behavior over time)
- Delivers transparent, explainable predictions using SHAP (SHapley Additive exPlanations)
- Offers an interactive Streamlit dashboard for both file-based and manual fraud checks

This project demonstrates how to build AI systems that are not just predictive but also interpretable and adaptive.

---

## Problem Statement

The financial sector constantly battles fraud that evolves with time and bypasses static models.  
Static systems can’t detect shifts in fraud patterns (concept drift), and they typically act as "black boxes" without explainability.

This project solves these key issues:
- Detects fraud in transactional data
- Monitors for concept drift using ADWIN to track shifting fraud behaviors
- Uses SHAP to explain model decisions in human-interpretable ways
- Offers an interactive, easy-to-use multi-page dashboard

---

## Key Features

- Real-time fraud detection on uploaded transaction files (CSV)
- Manual fraud checking via user input
- Visual explanations using SHAP summary plots
- Concept drift monitoring over time
- Multi-page Streamlit dashboard with navigation
- User-friendly design with input validation and progress indicators

---

## Project Structure

```
fraud-pattern-evolution-xai/
├── models/                       # Trained Random Forest model and scaler
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
├── data/                         # Sample datasets
│   └── raw/
│       └── creditcard.csv
├── streamlit_app/
│   ├── app.py                    # Dashboard home page
│   └── pages/                    # Multi-page navigation files
│       ├── 1_Upload_and_Predict.py
│       ├── 2_Manual_Fraud_Check.py
│       └── 3_Model_Explainability.py
├── src/                          # Model training, preprocessing, drift detection scripts
├── requirements.txt              # All dependencies
└── README.md

```
## Tech Stack
```
| Technology   | Purpose                                 |
| ------------ | --------------------------------------- |
| Python       | Core programming language               |
| Scikit-Learn | Machine Learning model training         |
| SHAP         | Explainable AI (model interpretability) |
| SMOTE        | Handling class imbalance                |
| Streamlit    | Interactive dashboard development       |
| Pandas       | Data manipulation and analysis          |
| Matplotlib   | Data visualization                      |
```

## Dataset
```
| Attribute               | Details                            |
| ----------------------- | ---------------------------------- |
| Total Transactions      | 284,807                            |
| Fraudulent Transactions | 492                                |
| Fraud Class Imbalance   | Highly imbalanced dataset          |
| Features                | PCA transformed features V1 to V28 |
| Additional Features     | Time, Amount                       |
| Target                  | Class (0 = Legitimate, 1 = Fraud)  |
```
