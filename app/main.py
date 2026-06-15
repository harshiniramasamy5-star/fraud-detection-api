from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Column order must match training: Time, V1..V28, Amount
COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Load model and scaler once at startup
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    features: list[float]  # 30 values in COLS order


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    if len(txn.features) != len(COLS):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(COLS)} features, got {len(txn.features)}",
        )
    df = pd.DataFrame([txn.features], columns=COLS)
    # Scale only Time and Amount, exactly as during training
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "fraud": bool(prediction),
        "fraud_probability": round(probability, 4),
    }