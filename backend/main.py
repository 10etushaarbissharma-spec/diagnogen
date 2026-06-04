from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import shap
import pandas as pd

app = FastAPI()

# Locate model relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")

model = None
explainer = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
    except Exception:
        pass

class Symptoms(BaseModel):
    fever: int
    cough: int
    headache: int
    fatigue: int

@app.get("/")
def home():
    return {"message": "Backend Running"}

@app.post("/predict")
def predict(symptom: Symptoms):
    if model is None:
        return {"error": "Model not loaded"}

    data = pd.DataFrame([[
        symptom.fever,
        symptom.cough,
        symptom.headache,
        symptom.fatigue
    ]], columns=["fever", "cough", "headache", "fatigue"])

    prediction = model.predict(data)
    disease = prediction[0]

    # Calculate confidence and full probabilities breakdown
    confidence = 1.0
    probabilities = {}
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(data)[0]
            classes = model.classes_
            probabilities = {cls: float(prob) for cls, prob in zip(classes, probs)}
            confidence = float(probabilities.get(disease, 1.0))
    except Exception:
        pass

    # Calculate SHAP values
    shap_analysis = None
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(data)
            classes = list(model.classes_)
            pred_idx = classes.index(disease)
            
            feature_names = ["fever", "cough", "headache", "fatigue"]
            contributions = {
                name: float(shap_vals[0, idx, pred_idx])
                for idx, name in enumerate(feature_names)
            }
            
            base_value = float(explainer.expected_value[pred_idx])
            
            shap_analysis = {
                "base_value": base_value,
                "final_value": float(confidence),
                "contributions": contributions
            }
        except Exception:
            pass

    return {
        "predicted_disease": disease,
        "confidence": confidence,
        "probabilities": probabilities,
        "shap_analysis": shap_analysis
    }