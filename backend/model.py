import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("symptoms.csv")

X = data[["fever", "cough", "headache", "fatigue"]]
y = data["disease"]

model = RandomForestClassifier()

model.fit(X, y)

joblib.dump(model, "disease_model.pkl")

print("Model trained successfully!")