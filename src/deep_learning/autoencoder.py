import json
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score,confusion_matrix,f1_score,precision_recall_curve,precision_score,recall_score,)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = "data/raw/creditcard.csv"
MODEL_PATH = "models/deep_learning/autoencoder.pth"
SCALER_PATH = "models/deep_learning/scaler.pkl"
REPORT_PATH = "reports/autoencoder_metrics.json"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path):
    return pd.read_csv(path)


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16),nn.ReLU(),nn.Linear(16, 8),)
        self.decoder = nn.Sequential(nn.Linear(8, 16),nn.ReLU(),nn.Linear(16, input_dim),)
    
    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


def prepare_data(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["Class"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["Class"]
    )
    normal_train = train_df[train_df["Class"] == 0]
    feature_columns = [column for column in df.columns if column != "Class"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_train[feature_columns])
    X_val = scaler.transform(val_df[feature_columns])
    X_test = scaler.transform(test_df[feature_columns])

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        val_df["Class"].to_numpy(),
        test_df["Class"].to_numpy(),
        scaler,
    )


def train_autoencoder(
    model,
    X_train,
    X_val_normal,
    epochs=20,
    batch_size=64,
    learning_rate=0.001,
):
    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        TensorDataset(X_val_normal), batch_size=batch_size, shuffle=False
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (transactions,) in train_loader:
            reconstructed = model(transactions)
            loss = criterion(reconstructed, transactions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * transactions.size(0)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for (transactions,) in validation_loader:
                loss = criterion(model(transactions), transactions)
                validation_loss += loss.item() * transactions.size(0)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {train_loss / len(X_train):.6f}, "
            f"Validation Loss: {validation_loss / len(X_val_normal):.6f}"
        )

    return model


def calculate_reconstruction_error(model, data):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data)
        return torch.mean((data - reconstructed) ** 2, dim=1)


def find_best_threshold(labels, errors):
    precision, recall, thresholds = precision_recall_curve(labels, errors)
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index]), {
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index]),
    }


def evaluate_model(model, X_test, y_test, threshold):
    errors = calculate_reconstruction_error(model, X_test).numpy()
    predictions = (errors > threshold).astype(int)
    metrics = {
        "model": "Autoencoder",
        "threshold": float(threshold),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, errors)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def save_artifacts(model, scaler, metrics):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(metrics, report_file, indent=2)


def main():
    set_seed()
    df = load_data(DATA_PATH)
    X_train, X_val, X_test, y_val, y_test, scaler = prepare_data(df)
    X_val_normal = X_val[torch.tensor(y_val == 0)]

    model = Autoencoder(input_dim=X_train.shape[1])
    model = train_autoencoder(model, X_train, X_val_normal)

    validation_errors = calculate_reconstruction_error(model, X_val).numpy()
    threshold, validation_metrics = find_best_threshold(y_val, validation_errors)
    metrics = evaluate_model(model, X_test, y_test, threshold)
    metrics["validation"] = validation_metrics
    save_artifacts(model, scaler, metrics)
    print("Saved model, scaler, and metrics report.")

def calculate_anomaly_score(model,scaler,features):
    scaled_features= scaler.transform([features])
    tensor_features= torch.tensor(scaled_features,dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_features)
    reconstruction_error = torch.mean((tensor_features - reconstructed) ** 2, dim=1)
    return reconstruction_error.numpy()


if __name__ == "__main__":
    main()


