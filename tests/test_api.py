from fastapi.testclient import TestClient
import pytest
from api.main import app
import pandas as pd

client=TestClient(app)

def test_health_check():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()== {"status": "healthy"}

valid_transaction={
    "Time":90000,
    "V1":0.0,
    "V2":0.2,
    "V3":0.3,
    "V4":0.4,
    "V5":0.5,
    "V6":0.6,
    "V7":0.7,
    "V8":0.8,
    "V9":0.9,
    "V10":1.0,
    "V11":1.1,
    "V12":1.2,
    "V13":1.3,
    "V14":1.4,
    "V15":1.5,
    "V16":1.6,
    "V17":1.7,
    "V18":1.8,
    "V19":1.9,
    "V20":2.0,
    "V21":2.1,
    "V22":2.2,
    "V23":2.3,
    "V24":2.4,
    "V25":2.5,
    "V26":2.6,
    "V27":2.7,
    "V28":2.8   ,
    "Amount":100}

def response_validator(data):
    assert "prediction" in data
    assert "explanation" in data
    assert "investigation_summary" in data
    assert isinstance(data["investigation_summary"],str)
    assert data ["investigation_summary"]
    assert "agent_recommendation" in data
    assert "latency_ms" in data

    
    pred = data["prediction"]
    assert isinstance(pred["fraud_probability"], (int, float))
    assert 0 <= pred["fraud_probability"] <= 1
    assert "anomaly_score" in pred
    assert isinstance(pred["anomaly_score"], (int, float))
    assert "decision" in pred
    
    expl = data["explanation"]
    assert isinstance(expl, list)
    assert len(expl) <= 3
    for item in expl:
        assert "feature" in item
        assert isinstance(item["contribution"], (int, float))

    agent= data["agent_recommendation"]
    assert "recommended_action" in agent
    assert "requires_human_approval" in agent
    assert "reason" in agent
    assert isinstance(agent["requires_human_approval"], bool)

    agent = data["agent_recommendation"]
    assert agent["recommended_action"] in {
        "Monitor",
        "human_review",
        "escalate_for_manual_approval",
    }

    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0

def test_predict_transaction_request():
    response=client.post("/predict",json=valid_transaction)
    assert response.status_code==200
    assert "prediction" in response.json()
    assert "fraud_probability" in response.json()["prediction"]
    assert "decision" in response.json()["prediction"]
    assert response.json()["prediction"]["decision"] in {"Approved", "Review", "Block"}
    response_validator(response.json())

def test_post_predict():
    response=client.post("/predict", json={"Time":0,"Amount":100})
    assert response.status_code==422

def test_invalid_transaction_request():
    response = client.post("/predict",json={"Time": "invalid", "Amount": 100})
    assert response.status_code == 422

def test_extra_fields_in_transaction_request():
    response = client.post("/predict",json={**valid_transaction, "extra_field": 123})
    assert response.status_code == 422


def test_model_info():
    response=client.get("/model-info")
    assert response.status_code==200
    data = response.json()
    assert data["model"] == "XGBoost"
    assert data["approval_threshold"] == 0.6 
    assert data["block_threshold"] == 0.8

def test_missing_model_artifacts_error(monkeypatch):
    import api.model_loader
    with pytest.raises(Exception):
        api.model_loader.calculate_anomaly_score(None, api.model_loader.scaler, pd.DataFrame([valid_transaction]))

def test_incorrect_feature_columns():
    invalid_transaction = {
        **valid_transaction,
        "WrongFeature": 123,
    }
    response = client.post(
        "/predict",
        json=invalid_transaction,
    )
    assert response.status_code == 422

def test_transactions_endpoint():
    prediction_response = client.post("/predict", json=valid_transaction)
    assert prediction_response.status_code == 200

    response = client.get("/transactions")
    assert response.status_code == 200

    transactions = response.json()
    assert isinstance(transactions, list)
    assert len(transactions) > 0

    latest = transactions[-1]
    assert "transaction_id" in latest
    assert "amount" in latest
    assert "fraud_probability" in latest
    assert "decision" in latest

def test_review_cases_endpoint():
    response = client.get("/review-cases")

    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_approve_missing_case():
    response = client.post(
        "/approve-case",
        json={
            "transaction_id": "missing-case-id",
            "notes": "Test case",
        },
    )

    assert response.status_code == 404

def test_agent_history():
    prediction_response = client.post("/predict",json=valid_transaction)
    assert prediction_response.status_code == 200

    transactions_response = client.get("/transactions")
    assert transactions_response.status_code == 200

    transactions = transactions_response.json()
    transaction_id = transactions[-1]["transaction_id"]

    response = client.get(f"/agent/history/{transaction_id}")

    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] >= 0
    assert 0 <= data["fraud_rate"] <= 1
    assert data["approved"] >= 0
    assert data["review"] >= 0
    assert data["blocked"] >= 0
    assert "average_latency_ms" in data
    assert data["average_latency_ms"] >= 0