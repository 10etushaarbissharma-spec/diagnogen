import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = "disease_model.pkl"
DATA_PATH = "symptoms.csv"


def train_and_save_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    data = pd.read_csv(data_path)
    X = data[["fever", "cough", "headache", "fatigue"]]
    y = data["disease"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    print(f"Trained and saved model to {model_path}")
    return model


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        print("Model not found, training new model...")
        return train_and_save_model()


def run_demo():
    model = load_or_train_model()

    demo_cases = [
        {"name": "All symptoms", "fever": 1, "cough": 1, "headache": 1, "fatigue": 1},
        {"name": "Fever + Cough", "fever": 1, "cough": 1, "headache": 0, "fatigue": 1},
        {"name": "Headache only", "fever": 0, "cough": 0, "headache": 1, "fatigue": 0},
        {"name": "No symptoms", "fever": 0, "cough": 0, "headache": 0, "fatigue": 0},
    ]

    print("\nDiagnoGen MVP - Hardcoded demo cases\n")
    for case in demo_cases:
        features = [[case["fever"], case["cough"], case["headache"], case["fatigue"]]]
        pred = model.predict(features)[0]
        out = f"{case['name']}: predicted -> {pred}"
        # show simple probability when available
        try:
            probs = model.predict_proba(features)[0]
            top_idx = probs.argmax()
            top_prob = probs[top_idx]
            out += f" (confidence ~ {top_prob:.2f})"
        except Exception:
            pass
        print(out)


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: required data file '{DATA_PATH}' not found in backend/.")
        print("Please ensure `symptoms.csv` is present next to this demo script.")
    else:
        run_demo()
