# Fraud Detection Platform

This project demonstrates a production-oriented fraud detection workflow using
classical machine learning, PyTorch deep learning, explainability, GenAI-style
investigation summaries, and human-supervised agent recommendations.

Tests run automatically through GitHub Actions on pushes and pull requests.

## Live demo

- Dashboard: https://fraud-detection-platform-tsqk.onrender.com
- API docs: https://fraud-detection-0c4l.onrender.com/docs

The Prediction view includes dataset-backed `Approved`, `Review`, and `Block`
scenarios. Clicking a scenario loads the complete feature vector and runs the
prediction automatically.

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

## GenAI and agentic workflow

The GenAI layer creates a grounded investigation summary from the model
decision, fraud probability, transaction amount, and SHAP signals. Gemini,
OpenRouter, or the deterministic mock provider can be selected through
environment variables. The LLM does not make or override the fraud decision.

The agent workflow is human-supervised:

```text
Approved -> Monitor
Review   -> Human review
Block    -> Escalate for manual approval
```

Analysts can approve or reject review cases from the dashboard. High-impact
actions are recommendations only until a human makes the final decision.

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
- Approved, Review, and Block buttons backed by real dataset examples

The quick form's neutral defaults are only a UI demonstration. For realistic
results, use the dataset-backed scenarios or send the complete `Time`, `V1`–
`V28`, and `Amount` feature vector to `POST /predict`.

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
The Docker stack has been verified locally with the API health endpoint,
dashboard page, proxied metrics endpoint, and quick prediction flow.

## Gemini investigation summaries

The default provider is the local mock provider, so tests and local development
do not require an API key. To enable Gemini, set these environment variables
without committing the key:

```powershell
$env:LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "your-key"
```

Gemini summarizes the existing fraud probability, decision, and SHAP
explanations. It does not make the fraud decision; XGBoost remains responsible
for `Approved`, `Review`, and `Block`.

OpenRouter can be enabled instead:

```powershell
$env:LLM_PROVIDER = "openrouter"
$env:OPENROUTER_API_KEY = "your-key"
$env:OPENROUTER_MODEL = "openrouter/free"
```

## Testing

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover API validation and persistence, model metadata, metrics,
GenAI fallback behavior, review workflows, and the three demonstration
scenarios.

## Render deployment

The [render.yaml](render.yaml) Blueprint defines an API service and dashboard
static site. In Render, create a new Blueprint from this repository, then
verify the generated service URLs and update `API_BASE_URL` and
`DASHBOARD_ORIGIN` if Render assigns different names. Add the
`GEMINI_API_KEY` secret to the API service. Never place the key in source code
or the dashboard.

## Model comparison

See [reports/model_comparison.md](reports/model_comparison.md) for the
evaluation and production-role decision.

## Limitations

- The dataset's `V1`–`V28` columns are anonymized, so explanations identify
  influential features but cannot describe their business meaning.
- The autoencoder is a complementary anomaly signal, not the production
  classifier.
- LLM summaries are optional and can fall back to deterministic text when a
  provider is unavailable.
- Fraud decisions are model predictions; real financial blocking should include
  analyst review, policy checks, monitoring, and threshold calibration.
