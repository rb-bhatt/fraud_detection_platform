from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from api.model_loader import model, feature_name, decide_transaction
from api.explainer import explain_transaction
import pandas as pd
from fastapi import Depends
from api.database import create_tables, get_db, Transaction, ReviewCase, Session
import json
import uuid
app=FastAPI(title="Fraud Detection API",description="API for predicting fraudulent transactions")

create_tables()

    
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

class FeatureExplanation(BaseModel):
    feature: str
    contribution: float

class DetailedPredictionResponse(BaseModel):
    prediction: PredictionResponse
    explanation: list[FeatureExplanation]

class TransactionResponse(BaseModel):
    id: int
    transaction_id: str
    amount: float
    fraud_probability: float
    decision: str
    original_features: str | None
    created_at: datetime


class ReviewCaseResponse(BaseModel):
    id: int
    transaction_id: str
    status: str
    analyst_decision: str | None
    notes: str | None
    created_at: datetime

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/review-cases",response_model=list[ReviewCaseResponse])
def get_review_cases(db: Session = Depends(get_db)):
    return db.query(ReviewCase).all()


@app.get("/transactions",response_model=list[TransactionResponse])
def get_transactions(db: Session = Depends(get_db)):
    return db.query(Transaction).all()


@app.post("/predict", response_model=DetailedPredictionResponse)
def predict_transaction_request(request: TransactionRequest,db: Session = Depends(get_db),):
    transaction_dict = request.model_dump()
    transaction_dataframe = pd.DataFrame([transaction_dict])
    try:
        transaction_dataframe = transaction_dataframe[feature_name]
        probabilities = model.predict_proba(transaction_dataframe)
        fraud_probability = float(probabilities[0][1])
        decision = decide_transaction(fraud_probability)
        explanation = explain_transaction(transaction_dataframe)
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            amount=float(request.Amount),
            fraud_probability=fraud_probability,
            decision=decision,
            original_features=json.dumps(transaction_dict),
        )
        db.add(transaction)
        db.commit()
        db.refresh(transaction)
        if decision in {"Review", "Block"}:
            review_case = ReviewCase(
                transaction_id=transaction.transaction_id,
                status="open",
            )
            db.add(review_case)
            db.commit()
        return {
            "prediction": {
                "fraud_probability": round(fraud_probability, 4),
                "decision": decision,
            },
            "explanation": explanation
        }
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