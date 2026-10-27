from fastapi.testclient import TestClient
from api.main import app
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
    
    pred = data["prediction"]
    assert isinstance(pred["fraud_probability"], (int, float))
    assert 0 <= pred["fraud_probability"] <= 1
    assert "decision" in pred
    
    expl = data["explanation"]
    assert isinstance(expl, list)
    assert len(expl) <= 3
    for item in expl:
        assert "feature" in item
        assert isinstance(item["contribution"], (int, float))

def test_predict_transaction_request():
    response=client.post("/predict",json=valid_transaction)
    assert response.status_code==200
    assert "prediction" in response.json()
    assert "fraud_probability" in response.json()["prediction"]
    assert "decision" in response.json()["prediction"]
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