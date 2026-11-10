import pickle
import json
import joblib
import torch
import numpy as np

with open("models/xgboost_model.pkl","rb") as open_file:
    model=pickle.load(open_file)
with open("models/feature_names.json","rb") as open_file:
    feature_name=json.load(open_file)
scaler = joblib.load("models/deep_learning/scaler.pkl")
autoencoder_model = torch.load("models/deep_learning/autoencoder_scaler")
def calculate_anomaly_score(model_ae, feature_scaler, features_df):
    scaled_features = feature_scaler.transform(features_df)
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
    model_ae.eval()
    with torch.no_grad():
        reconstructed = model_ae(tensor_features)
    reconstruction_error = torch.mean((tensor_features - reconstructed) ** 2, dim=1)
    return float(reconstruction_error.numpy()[0])
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