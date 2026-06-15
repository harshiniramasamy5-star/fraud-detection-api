from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Column order must match training: Time, V1..V28, Amount
COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Decision threshold. The default 0.5 assumes false positives and false
# negatives cost the same; in fraud they do not. This should be set to the
# value chosen during training (e.g. the recall-constrained optimum) so the
# model optimizes total cost rather than raw accuracy.
THRESHOLD = 0.5

# Load model and scaler once at startup
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

app = FastAPI(title="Fraud Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    features: list[float]  # 30 values in COLS order


class TransactionBatch(BaseModel):
    transactions: list[Transaction]


def _risk_level(prob: float) -> str:
    """Bucket a fraud probability into an operational risk level.

    Mirrors how a real system routes transactions to different actions:
    clear automatically, queue for review, hold, or block outright.
    """
    if prob >= 0.95:
        return "CRITICAL"   # block immediately
    if prob >= 0.80:
        return "HIGH"       # hold and review
    if prob >= THRESHOLD:
        return "MEDIUM"     # flag for review
    return "LOW"            # clears automatically


def _score(features: list[float]) -> dict:
    """Score a single feature vector and return the full decision payload."""
    if len(features) != len(COLS):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(COLS)} features, got {len(features)}",
        )
    df = pd.DataFrame([features], columns=COLS)
    # Scale only Time and Amount, exactly as during training
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "fraud": probability >= THRESHOLD,
        "fraud_probability": round(probability, 4),
        "risk_level": _risk_level(probability),
        "threshold_used": THRESHOLD,
    }


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    return _score(txn.features)


@app.post("/predict/batch")
def predict_batch(batch: TransactionBatch):
    results = [_score(t.features) for t in batch.transactions]
    flagged = sum(1 for r in results if r["fraud"])
    total = len(results)
    return {
        "predictions": results,
        "total_transactions": total,
        "flagged_as_fraud": flagged,
        "fraud_rate_in_batch": round(flagged / total, 4) if total else 0.0,
    }
