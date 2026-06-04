from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("disease_model.pkl")

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

    data = [[
        symptom.fever,
        symptom.cough,
        symptom.headache,
        symptom.fatigue
    ]]

    prediction = model.predict(data)

    return {
        "predicted_disease": prediction[0]
    }