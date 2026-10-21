import pickle
import pandas as pd
import json

def decide_transaction(fraud_prob: float) -> str:
    if not 0 <= fraud_prob <= 1:
        raise ValueError("Fraud probability must be between 0 and 1.")
    if fraud_prob < 0.2:
        decision = "Approved"
    elif fraud_prob < 0.8:
        decision = "Review"
    else:
        decision = "Block"
    print(f"Decision: {decision}, Probability: {fraud_prob:.4f}")
    return decision

def predict_transaction(transaction_data: pd.DataFrame)->dict:
    transaction_scaled=scaler.transform(transaction_data)
    fraud_probability=model.predict_proba(transaction_scaled)[:,1]
    fraud_prob=float(fraud_probability[0])
    decision=decide_transaction(fraud_probability[0])
    return {"fraud_probability": fraud_prob, "decision": decision}

if __name__ == "__main__":
    with open("models/scale.pkl","rb") as open_file:
        scaler=pickle.load(open_file)
    with open("models/logistic_model.pkl","rb") as open_file:
        model=pickle.load(open_file)
    with open("models/feature_names.json","r") as open_file:
        feature_name=json.load(open_file)

    transaction_data = pd.DataFrame([{
        feature: 0
        for feature in feature_name}])
    
    transaction_data["Time"] = 12345
    transaction_data["Amount"] = 100.50
    print("Columns match:", transaction_data.columns.tolist() == feature_name)
    print("Number of features:", len(transaction_data.columns))
    print("DataFrame shape:", transaction_data.shape)

    transaction_scaled=scaler.transform(transaction_data)
    fraud_probability=model.predict_proba(transaction_scaled)[:,1]
    print("Fraud Probability:", fraud_probability)
    fraud_prob=float(fraud_probability[0])
    #fraud_probability(fraud_prob)

    model_name = "logistic_model"
    model_version = "1.0"       
    feature_count = len(feature_name)  
    threshold_value = 0.8
    training_dataset_name = "creditcard_clean.csv"
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "feature_count": feature_count,
        "threshold_value": threshold_value,
        "training_dataset_name": training_dataset_name
    }
    with open("models/model_metadata.json","w") as open_file:
        json.dump(metadata,open_file,indent=4)
    print("Model metadata successfully written to model_metadata.json")
    result = predict_transaction(transaction_data)
    print(result)