from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from api.model_loader import model, feature_name, decide_transaction
import pandas as pd

app=FastAPI(title="Fraud Detection API",description="API for predicting fraudulent transactions")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: str

@app.post("/predict",response_model=PredictionResponse)
def predict_transaction_request(request: TransactionRequest):
    transaction_dict = request.model_dump()
    transaction_dataframe = pd.DataFrame([transaction_dict])
    try:
        transaction_dataframe = transaction_dataframe[feature_name]
        probabilities = model.predict_proba(transaction_dataframe)
        fraud_probability = float(probabilities[0][1])
        decision = decide_transaction(fraud_probability)
        return {"fraud_probability": round(fraud_probability, 4),"decision": decision}

    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422,detail="Invalid transaction features") from exc

    except Exception as exc:raise HTTPException(
            status_code=500,
            detail="Prediction failed") from exc

@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost",
        "version": "1.0",
        "approval_threshold": 0.6,
        "block_threshold": 0.8,
    }

@app.get("/response")
def response(fraud_probability: float,decision: str):
    response_model=PredictionResponse
    assert isinstance(response_model["fraud_probability"],(int,float))
    assert response_model["decision"] in {"Approved", "Review", "Block"}