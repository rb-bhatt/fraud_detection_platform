import pickle
import json
with open("models/xgboost_model.pkl","rb") as open_file:
    model=pickle.load(open_file)
with open("models/feature_names.json","rb") as open_file:
    feature_name=json.load(open_file)
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