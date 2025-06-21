import pandas as pd
from river.drift import ADWIN
from data_loader import load_data

def detect_concept_drift(file_path):
    df = load_data(file_path)

    drift_detector = ADWIN()
    drifts = []

    print("Checking for concept drift over time...")

    for index, row in df.iterrows():
        value = row['Class']  # 0 = Non-Fraud, 1 = Fraud

        in_drift = drift_detector.update(value)

        if in_drift:
            print(f"Drift detected at index: {index}")
            drifts.append(index)

    if not drifts:
        print("No concept drift detected.")
    else:
        print(f"Concept drift detected at {len(drifts)} points.")

    return drifts

if __name__ == "__main__":
    detect_concept_drift('data/raw/creditcard.csv')
