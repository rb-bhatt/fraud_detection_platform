from datetime import datetime
import os
import math
import time
import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from api.model_loader import model, feature_name, decide_transaction,autoencoder_model, scaler, calculate_anomaly_score
from api.explainer import explain_transaction
import pandas as pd
from fastapi import Depends
from api.database import create_tables, get_db, Transaction, ReviewCase, Session
from api.ai_assistant import build_investigation_summary
from api.agent_workflow import recommend_action
from api.agent_tools import get_transaction_history
import logging

logger = logging.getLogger(__name__)

app=FastAPI(title="Fraud Detection API",description="API for predicting fraudulent transactions")
dashboard_origin = os.getenv("DASHBOARD_ORIGIN")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        *([dashboard_origin] if dashboard_origin else []),
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    anomaly_score: float
    decision: str

class FeatureExplanation(BaseModel):
    feature: str
    contribution: float

class AgentRecommendation(BaseModel):
    recommended_action: str
    requires_human_approval:bool
    reason:str

class DetailedPredictionResponse(BaseModel):
    prediction: PredictionResponse
    explanation: list[FeatureExplanation]
    investigation_summary:str
    agent_recommendation: AgentRecommendation
    latency_ms: float
    
class TransactionResponse(BaseModel):
    id: int
    transaction_id: str
    amount: float
    fraud_probability: float
    anomaly_score: float | None
    decision: str
    original_features: str | None
    created_at: datetime
    latency_ms: float


class ReviewCaseResponse(BaseModel):
    id: int
    transaction_id: str
    status: str
    analyst_decision: str | None
    notes: str | None
    created_at: datetime

class CaseActionRequest(BaseModel):
    transaction_id: str
    notes: str | None = None

@app.get("/health")
def health_check():
    return {"status": "healthy",}

@app.get("/review-cases",response_model=list[ReviewCaseResponse])
def get_review_cases(db: Session = Depends(get_db)):
    return db.query(ReviewCase).all()


@app.get("/transactions", response_model=list[TransactionResponse])
def get_transactions(db: Session = Depends(get_db)):
    return db.query(Transaction).order_by(Transaction.id.desc()).all()


@app.post("/predict", response_model=DetailedPredictionResponse)
def predict_transaction_request(request: TransactionRequest,db: Session = Depends(get_db),):
    start_time = time.perf_counter()
    transaction_dict = request.model_dump()
    transaction_dataframe = pd.DataFrame([transaction_dict])
    try:
        transaction_dataframe = transaction_dataframe[feature_name]
        probabilities = model.predict_proba(transaction_dataframe)
        fraud_probability = float(probabilities[0][1])
        anomaly_score = calculate_anomaly_score(autoencoder_model, scaler, transaction_dataframe)
        decision = decide_transaction(fraud_probability)
        explanation = explain_transaction(transaction_dataframe)
        investigation_summary= build_investigation_summary(
            fraud_probability,
            decision,
            explanation,
            transaction_dict)
        agent_recommendation = recommend_action(fraud_probability, decision, investigation_summary)
        latency_ms = (time.perf_counter() - start_time) * 1000
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            amount=float(request.Amount),
            fraud_probability=fraud_probability,
            anomaly_score=anomaly_score,
            decision=decision,
            original_features=json.dumps(transaction_dict),
            latency_ms=round(latency_ms, 2),
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
            db.refresh(review_case)
        return {
            "prediction": {
                "fraud_probability": round(fraud_probability, 4),
                "anomaly_score": round(anomaly_score, 4),
                "decision": decision,
            },
            "explanation": explanation,
            "investigation_summary": investigation_summary,
            "agent_recommendation":agent_recommendation,
            "latency_ms": round(latency_ms, 2)
        }
    except (KeyError, ValueError) as exc:
        logger.exception("Invalid transaction features")
        raise HTTPException(
            status_code=422,
            detail="Invalid transaction features"
        ) from exc

    except Exception as exc:
        logger.exception("Prediction pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc
    

@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost",
        "secondary_model": "PyTorch Autoencoder",
        "secondary_signal": "anomaly_score",
        "version": "1.0",
        "approval_threshold": 0.6,
        "block_threshold": 0.8,
    }

@app.post("/approve-case")
def approve_case(payload: CaseActionRequest, db: Session = Depends(get_db)):
    case = db.query(ReviewCase).filter(ReviewCase.transaction_id == payload.transaction_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Review case not found")
    
    case.status = "approved"
    case.analyst_decision = "approved"
    case.notes = payload.notes
    db.commit()
    db.refresh(case)
    return {"status": "success", "message": f"Case {payload.transaction_id} approved", "case_id": case.id}

@app.post("/reject-case")
def reject_case(payload: CaseActionRequest, db: Session = Depends(get_db)):
    case = db.query(ReviewCase).filter(ReviewCase.transaction_id == payload.transaction_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Review case not found")
    
    case.status = "rejected"
    case.analyst_decision = "rejected"
    case.notes = payload.notes
    db.commit()
    db.refresh(case)
    return {"status": "success", "message": f"Case {payload.transaction_id} rejected", "case_id": case.id}

@app.get("/agent/history/{transaction_id}")
def transaction_history(transaction_id: str, db: Session = Depends(get_db)):
    return get_transaction_history(db, transaction_id)

@app.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    transactions = db.query(Transaction).all()
    total_transactions = len(transactions)
    approved = sum(t.decision == "Approved" for t in transactions)
    review = sum(t.decision == "Review" for t in transactions)
    blocked = sum(t.decision == "Block" for t in transactions)
    fraud_rate = ((review + blocked) / total_transactions if total_transactions
        else 0)
    average_latency_ms = (
        sum(t.latency_ms for t in transactions) / total_transactions if total_transactions
        else 0)
    anomaly_scores = [
        t.anomaly_score
        for t in transactions
        if t.anomaly_score is not None and math.isfinite(t.anomaly_score)
    ]
    open_review_cases = (db.query(ReviewCase).filter(ReviewCase.status == "open").count())

    return {
        "total_transactions": total_transactions,
        "approved": approved,
        "review": review,
        "blocked": blocked,
        "fraud_rate": round(fraud_rate, 4),
        "open_review_cases": open_review_cases,
        "average_latency_ms": round(average_latency_ms, 2),
        "average_anomaly_score": round(
            sum(anomaly_scores) / len(anomaly_scores), 4
        ) if anomaly_scores else 0,
    }