import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    base = os.path.dirname(__file__)
    synthetic = os.path.join(base, "symptoms_synthetic.csv")
    original = os.path.join(base, "symptoms.csv")

    if os.path.exists(synthetic):
        path = synthetic
    elif os.path.exists(original):
        path = original
    else:
        raise FileNotFoundError("No dataset found in backend/. Expected symptoms_synthetic.csv or symptoms.csv")

    print(f"Loading data from {path}")
    return pd.read_csv(path)


def train_and_save(df, model_path=None):
    X = df[["fever", "cough", "headache", "fatigue"]]
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
    print("Classification report:")
    print(classification_report(y_test, preds, zero_division=0))

    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "disease_model.pkl")

    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    df = load_data()
    train_and_save(df)
