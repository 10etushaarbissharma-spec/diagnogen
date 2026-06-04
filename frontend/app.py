import streamlit as st
import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "backend" / "disease_model.pkl"
SYN_PATH = ROOT / "backend" / "symptoms_synthetic.csv"
DATA_PATH = SYN_PATH if SYN_PATH.exists() else ROOT / "backend" / "symptoms.csv"


st.set_page_config(page_title="DiagnoGen", page_icon="🩺", layout="centered")

st.markdown("""
<style>
.main { background-color: #0E1117; }
h1 { color: #00D4FF; text-align: center; }
.stButton>button { background-color: #00D4FF; color: black; border-radius: 10px; height: 3em; width: 100%; font-size: 18px; font-weight: bold; }
.result-box { padding: 20px; border-radius: 10px; background-color: #1E1E1E; color: white; font-size: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 DiagnoGen (Local Demo)")
st.subheader("Local ML pipeline demo — educational only")

st.write("### Select Symptoms")

col1, col2 = st.columns(2)
with col1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
with col2:
    headache = st.checkbox("Headache")
    fatigue = st.checkbox("Fatigue")


def train_and_save_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(data_path)
    X = df[["fever", "cough", "headache", "fatigue"]]
    y = df["disease"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model


def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save_model()


st.write("")

predict_col, info_col = st.columns([2, 1])

with predict_col:
    if st.button("Predict Locally"):
        if not DATA_PATH.exists():
            st.error(f"Missing data file: {DATA_PATH}")
        else:
            model = load_or_train_model()
            features = [[int(fever), int(cough), int(headache), int(fatigue)]]
            try:
                pred = model.predict(features)[0]
                prob_text = ""
                try:
                    proba = model.predict_proba(features)[0]
                    prob_text = f" (confidence ~ {proba.max():.2f})"
                except Exception:
                    pass

                st.markdown(f"<div class=\"result-box\">Predicted Disease:<br><br><b>{pred}</b>{prob_text}</div>", unsafe_allow_html=True)
                st.info("This prediction is for educational support only and not medical advice.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with info_col:
    if st.button("Train / Retrain Model"):
        if not DATA_PATH.exists():
            st.error(f"Missing data file: {DATA_PATH}")
        else:
            with st.spinner("Training model…"):
                model = train_and_save_model()
            st.success(f"Model trained and saved to {MODEL_PATH}")


st.write("---")

if st.checkbox("Show training dataset"):
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        st.dataframe(df)
    else:
        st.error(f"Missing data file: {DATA_PATH}")

if st.checkbox("Show demo cases"):
    demo_cases = [
        {"name": "All symptoms", "fever": 1, "cough": 1, "headache": 1, "fatigue": 1},
        {"name": "Fever + Cough", "fever": 1, "cough": 1, "headache": 0, "fatigue": 1},
        {"name": "Headache only", "fever": 0, "cough": 0, "headache": 1, "fatigue": 0},
        {"name": "No symptoms", "fever": 0, "cough": 0, "headache": 0, "fatigue": 0},
    ]
    st.write(demo_cases)

st.write("")
st.caption("Note: This is a local demo. To re-enable the API-based flow, see backend/main.py")