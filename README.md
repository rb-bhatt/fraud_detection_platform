# Fraud Detection Platform

This project demonstrates a production-oriented fraud detection workflow using
classical machine learning, PyTorch deep learning, explainability, GenAI-style
investigation summaries, and human-supervised agent recommendations.

Tests run automatically through GitHub Actions on pushes and pull requests.

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

## Dashboard

Start the API in one terminal:

```powershell
.\.venv\Scripts\uvicorn.exe api.main:app --reload
```

Start the dashboard in a second terminal:

```powershell
Set-Location dashboard
python -m http.server 5500
```

Open `http://127.0.0.1:5500` in a browser. The dashboard provides:

- Decision and model-health charts
- Transaction and review-case tables
- Approve/reject actions for open review cases
- XGBoost and PyTorch model metadata
- A quick prediction form using neutral feature defaults

## Docker

Docker Desktop is required. From the project root:

```powershell
docker compose up --build
```

Open the dashboard at `http://localhost:5500` and the API at
`http://localhost:8000/docs`. Stop the services with:

```powershell
docker compose down
```

The SQLite database is persisted through the `./data:/app/data` Compose volume.

## Model comparison

See [reports/model_comparison.md](reports/model_comparison.md) for the
evaluation and production-role decision.
