from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Fraud"
    else:
        result = "Normal"

    return {"prediction": result}

