# Network Intrusion Detection API

A FastAPI service that scores network flow records for intrusion risk using a
two-stage model pipeline, with per-prediction SHAP explanations.

Trained on the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
dataset schema.

## How it works

1. **Stage 1 — anomaly filter**: an `IsolationForest` screens every flow. If
   it isn't flagged as anomalous, the API short-circuits and returns `Normal`
   immediately — the classifier never runs.
2. **Stage 2 — attack categorization**: flows flagged by stage 1 go to a
   `CatBoostClassifier` that predicts an attack category (e.g. `DoS`,
   `Exploits`, `Reconnaissance`) with a confidence score.
3. **Explanation**: a SHAP `TreeExplainer` (built once at startup) returns the
   top 5 features driving that specific prediction.

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /score`

Body: a single flow record — see `FlowInput` in [app.py](app.py) for the full
schema (44 fields matching the UNSW-NB15 columns after cleaning). All numeric
fields are validated as non-negative; `hour` must be `0-23`.

Normal traffic:

```json
{
  "stage1_flagged": false,
  "prediction": "Normal",
  "risk_score": "low",
  "top_contributing_features": null
}
```

Flagged traffic:

```json
{
  "stage1_flagged": true,
  "prediction": "Exploits",
  "confidence": 0.7138,
  "risk_score": "high",
  "top_contributing_features": [
    {"feature": "sttl", "shap_value": 1.108},
    {"feature": "smeansz", "shap_value": 0.664}
  ]
}
```

`risk_score` is `"high"` when `confidence > 0.7`, otherwise `"medium"`.

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Interactive docs at `http://127.0.0.1:8000/docs`.

> `--reload` is not recommended here: without the `watchfiles` package,
> uvicorn falls back to a slow polling reloader that recursively scans the
> working directory, including `.venv/` — which can hang for tens of minutes
> given how many files ship with numpy/scipy/pandas/catboost/shap. If you
> want `--reload`, `pip install watchfiles` first.

## Running with Docker

```bash
docker build -t intrusion-detection-api .
docker run -p 8000:8000 intrusion-detection-api
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest test_score.py -v
```

Tests use `fastapi.testclient.TestClient` against the app in-process — no
running server required.

## Project structure

```
fastapi_service/
├── app.py                  # FastAPI app: model loading, scoring, SHAP
├── test_score.py           # pytest suite
├── model_artifacts/
│   ├── isolation_forest.joblib
│   ├── catboost_stage2.cbm
│   ├── proto_freq.json     # protocol -> training-set frequency
│   └── feature_columns.json
├── requirements.txt
├── requirements-dev.txt
└── Dockerfile
```

## Known limitations

- `model_artifacts/isolation_forest.joblib` was pickled with a specific
  scikit-learn version; `requirements.txt` pins `scikit-learn==1.6.1` to
  match it. Bumping scikit-learn without re-pickling the model risks silent
  prediction drift.
- Attack-category confidence is a single-model softmax score, not a
  calibrated probability.
