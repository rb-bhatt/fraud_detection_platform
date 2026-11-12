from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import torch

from .autoencoder import Autoencoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "deep_learning" / "autoencoder.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "deep_learning" / "scaler.pkl"


def load_autoencoder_artifacts() -> tuple[Autoencoder, object]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Autoencoder model not found: {MODEL_PATH}")
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Autoencoder scaler not found: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    model = Autoencoder(input_dim=len(scaler.feature_names_in_))
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


def calculate_anomaly_score(
    model: Autoencoder,
    scaler: object,
    features: pd.DataFrame | np.ndarray,
) -> float:
    if isinstance(features, pd.DataFrame):
        scaled_features = scaler.transform(features)
    else:
        feature_values = np.asarray(features, dtype=np.float32)
        scaled_features = scaler.transform(feature_values)
    if scaled_features.ndim != 2 or scaled_features.shape[0] != 1:
        raise ValueError("features must contain exactly one transaction row")

    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(tensor_features)
    error = torch.mean((tensor_features - reconstructed) ** 2, dim=1)
    score = float(error.item())
    if not math.isfinite(score):
        raise ValueError("Anomaly score is not finite for these features")
    return score
