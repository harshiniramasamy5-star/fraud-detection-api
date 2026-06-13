from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Column order must match training: Time, V1..V28, Amount
COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Load model and scaler once at startup
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    features: list[float]  # 30 values in COLS order


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    df = pd.DataFrame([txn.features], columns=COLS)
    # Scale only Time and Amount, exactly as during training
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "fraud": bool(prediction),
        "fraud_probability": round(probability, 4),
    }