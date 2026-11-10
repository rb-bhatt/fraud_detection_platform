# Fraud Detection Platform

This project demonstrates a production-oriented fraud detection workflow using
classical machine learning, PyTorch deep learning, explainability, GenAI-style
investigation summaries, and human-supervised agent recommendations.

## Architecture

1. FastAPI receives and validates a transaction.
2. XGBoost produces the production fraud probability.
3. A PyTorch autoencoder produces a complementary reconstruction-error
   `anomaly_score`.
4. XGBoost thresholds determine `Approved`, `Review`, or `Block`.
5. SHAP, the investigation summary, and the agent recommendation explain the
   result.
6. SQLite stores transactions and review cases for monitoring.

XGBoost remains the decision-maker. The autoencoder is an experimental
unsupervised signal and does not replace the supervised classifier.

## Run

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\uvicorn.exe api.main:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `GET /transactions`
- `GET /metrics`

## Model comparison

See [reports/model_comparison.md](reports/model_comparison.md) for the
evaluation and production-role decision.
