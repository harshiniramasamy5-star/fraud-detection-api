# Credit Card Fraud Detection — End-to-End ML Service

A production-style machine learning system that detects fraudulent credit card transactions in real time. The project covers the full lifecycle: data analysis, model training on severely imbalanced data, model explainability, and deployment as a containerized REST API.

**Live demo:** `https://fraud-detection-api-5eog.onrender.com/docs`
*(Hosted on Render's free tier — the first request after inactivity may take 30–60 seconds to cold-start.)*

---

## Problem

The dataset contains 284,807 transactions, of which only **492 (0.17%) are fraudulent**. This extreme class imbalance is the central challenge: a model that naively predicts "not fraud" for every transaction would score 99.83% accuracy while catching zero fraud. Accuracy is therefore useless here, and the project is evaluated on **precision, recall, and ROC-AUC** instead.

---

## Approach

Three models were trained and compared to demonstrate the impact of handling imbalance and of model choice:

| Model | Recall (fraud) | Precision (fraud) | False Positives | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.92 | 0.06 | 1,389 | 0.972 |
| **XGBoost (final model)** | **0.86** | **0.64** | **48** | **0.985** |

The balanced logistic regression achieved slightly higher recall but raised 1,389 false alarms (6% precision) — unusable in practice, as a bank would drown in false positives. **XGBoost caught 86% of fraud with only 48 false positives (29x fewer)** and the best F1, making it the chosen production model.

### Key decisions
- **Feature scaling:** Time and Amount were standardized (the V1-V28 features are already PCA-transformed). The fitted scaler is saved and applied identically at inference to prevent train/serve skew.
- **Class imbalance:** handled via XGBoost's scale_pos_weight (ratio of normal to fraud).
- **Explainability:** SHAP values identify which features drive each fraud prediction, making the model auditable rather than a black box.

---

## Tech Stack

- **ML:** scikit-learn, XGBoost, SHAP
- **API:** FastAPI + Uvicorn
- **Containerization:** Docker
- **Deployment:** Render
- **Data analysis:** pandas, NumPy, Matplotlib, Seaborn

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Liveness check; returns {"status": "ok"} |
| POST | /predict | Returns fraud prediction and probability for a transaction |
| GET | /docs | Interactive Swagger UI (auto-generated) |

### Example request

\`\`\`json
POST /predict
{
  "features": [0.0, -1.359, -0.072, 2.536, 1.378, -0.338, 0.462, 0.239,
               0.098, 0.363, 0.090, -0.551, -0.617, -0.991, -0.311, 1.468,
               -0.470, 0.207, 0.025, 0.403, 0.251, -0.018, 0.277, -0.110,
               0.066, 0.128, -0.189, 0.133, -0.021, 149.62]
}
\`\`\`

The features array must contain 30 values in the order: Time, V1-V28, Amount.

### Example response

\`\`\`json
{
  "fraud": false,
  "fraud_probability": 0.0021
}
\`\`\`

Input is validated automatically via Pydantic — malformed requests are rejected before reaching the model.

---

## Run Locally

### With Docker (recommended)

\`\`\`bash
git clone https://github.com/harshiniramasamy5-star/fraud-detection-api.git
cd fraud-detection-api
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
\`\`\`

Then open http://localhost:8000/docs.

### Without Docker

\`\`\`bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
\`\`\`

---

## Project Structure

\`\`\`
fraud-detection-api/
├── app/
│   └── main.py            # FastAPI application
├── model/
│   ├── fraud_model.pkl    # Trained XGBoost model
│   └── scaler.pkl         # Fitted StandardScaler
├── notebooks/
│   └── 01_eda.ipynb       # EDA, training, evaluation, SHAP
├── Dockerfile
├── requirements.txt
└── README.md
\`\`\`

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group) — 284,807 European cardholder transactions over two days. Features V1-V28 are anonymized via PCA; Time, Amount, and Class (0 = normal, 1 = fraud) are provided directly. The dataset is not included in this repository due to size; download it from the link and place creditcard.csv in a data/ folder to retrain.

---

## Possible Extensions

- CI/CD pipeline (GitHub Actions) to run tests and auto-deploy on push
- Request logging and input-drift monitoring
- A /retrain endpoint for model updates
- A lightweight frontend that consumes the API
