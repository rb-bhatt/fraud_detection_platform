import json
import pickle
from pathlib import Path

from src.deep_learning.inference import (
    calculate_anomaly_score,
    load_autoencoder_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
with (PROJECT_ROOT / "models" / "xgboost_model.pkl").open("rb") as open_file:
    model = pickle.load(open_file)
with (PROJECT_ROOT / "models" / "feature_names.json").open(
    "r", encoding="utf-8"
) as open_file:
    feature_name = json.load(open_file)

autoencoder_model, scaler = load_autoencoder_artifacts()
def decide_transaction(fraud_prob: float) -> str:
    APPROVAL_THRESHOLD = 0.6
    BLOCK_THRESHOLD = 0.8
    if not 0 <= fraud_prob <= 1:
        raise ValueError("Fraud probability must be between 0 and 1.")
    
    if fraud_prob < APPROVAL_THRESHOLD:
        decision = "Approved"
    elif fraud_prob < BLOCK_THRESHOLD:
        decision = "Review"
    else:
        decision = "Block"
    print(f"Decision: {decision}, Probability: {fraud_prob:.4f}")
    return decision