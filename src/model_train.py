import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data, preprocess_data

def train_model(X_train, y_train):
    """Train a Random Forest model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print key metrics"""
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    data = load_data('data/raw/creditcard.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Save the model for later use
    joblib.dump(model, 'models/fraud_detection_model.pkl')
    print("Model saved to models/fraud_detection_model.pkl")
