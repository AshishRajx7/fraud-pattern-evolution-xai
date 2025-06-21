import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the dataset"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Test the preprocessing pipeline
    data = load_data('data/raw/creditcard.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved successfully!")    
