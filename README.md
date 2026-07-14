# Network Intrusion Detection API

A two-stage machine learning pipeline that screens network flow records for
anomalies and classifies flagged ones into attack categories, served as a
FastAPI microservice with a SHAP explanation attached to every flagged
prediction.

## Background

The pipeline is built as a cascade rather than a single classifier: a cheap,
unsupervised `IsolationForest` screens every incoming flow first, and only
flows it flags as anomalous are passed to the more expensive `CatBoost`
multiclass classifier and SHAP explainer. This keeps the costly part of the
pipeline off the hot path for ordinary traffic. The feature schema (44
fields, including a stray-space column name `ct_src_ ltm` preserved from the
source data) matches the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
dataset after cleaning.

The model was built from the four raw UNSW-NB15 CSV files, not a pre-cleaned
version — cleaning included stripping whitespace from categorical columns
(the raw data has both `'Fuzzers'` and `' Fuzzers'` as distinct values),
normalizing the `Backdoors`/`Backdoor` label inconsistency, parsing
hex-encoded ports, and dropping `srcip`/`dstip` to prevent the model from
memorizing specific hosts instead of learning flow behavior. Full EDA,
cleaning, and modeling work is in the
[Kaggle notebook](https://www.kaggle.com/code/ovchalke/api-anomaly-detection).

## How it works

1. **Stage 1 — anomaly filter**: an `IsolationForest` screens every flow. If
   it isn't flagged as anomalous, the API short-circuits and returns `Normal`
   immediately — the classifier and explainer never run.
2. **Stage 2 — attack categorization**: flows flagged by stage 1 go to a
   `CatBoostClassifier` that predicts an attack category (e.g. `DoS`,
   `Exploits`, `Reconnaissance`) with a confidence score.
3. **Explanation**: a SHAP `TreeExplainer` (built once at startup, reused
   across requests) returns the top 5 features driving that specific
   prediction, ranked by absolute SHAP value.

## Results / Model performance

**Evaluation methodology:** time-aware split, not random. UNSW-NB15 ships as
four sequential capture files; the first three are used for training and the
fourth, held out entirely, is the test set. A random row-level split would
let temporally-correlated flows leak between train and test and overstate
performance — the time-aware split is a harder, more honest test of
generalization to traffic the model hasn't seen.

**Headline metric:**

| Metric | Value |
|---|---|
| End-to-end detection rate (attacks correctly flagged by Stage 1 *and* correctly categorized by Stage 2, as a share of all real attacks in the held-out test set) | 87.2% |
| Per-class precision/recall/F1 | *pending — the notebook's execution log wasn't retained by Kaggle's API; will be added from a fresh run's output* |
| Confusion matrix | *pending, same reason* |

**Threshold tuning:** the `IsolationForest` `contamination` parameter went
through two sweeps. A coarse sweep over `[0.03, 0.05, 0.08, 0.11, 0.15, 0.20]`
located the useful range; a fine sweep over `[0.08, 0.09, 0.10, 0.11]` then
tracked per-class recall and macro/weighted F1 at each value. The result:
attack recall was flat from `0.09` to `0.11`, so `0.09` was chosen over
`0.11` — it captures the same detection benefit at a lower false-alarm cost,
since a lower contamination value flags fewer flows for the (more expensive,
more error-prone) Stage 2 classifier to process. The exact recall/F1/false-alarm
numbers from that sweep are pending for the same reason as above.

**Models compared:**

| Stage | Tried | Shipped | Why |
|---|---|---|---|
| 1 (anomaly filter) | `IsolationForest` vs. a Keras autoencoder (32→16→32 encoder/decoder, reconstruction-error threshold) | `IsolationForest` | Both were evaluated via `classification_report` + PR-AUC on the held-out set; the autoencoder was not carried into the final pipeline. Comparative PR-AUC values pending. |
| 2 (attack classifier) | `CatBoostClassifier` with `auto_class_weights='Balanced'` vs. the same model with **sqrt-dampened** custom class weights vs. `LightGBM` | `CatBoostClassifier`, sqrt-dampened weights | The dampening (`sqrt` of the balanced class weights) exists specifically because straight `'balanced'` weighting over-inflates tiny classes like Worms; the dampened-weight CatBoost was the best-performing config of the three. Comparative F1 values pending. |

**Weakest classes:** DoS, Analysis, and Backdoor have the weakest precision
in this pipeline. Exact precision/recall numbers per class are pending — see
above.

*The three "pending" items above all come from the same root cause: Kaggle's
API returned this notebook's source code but not its saved execution
output/log, so the printed `classification_report`/confusion-matrix/sweep
tables aren't recoverable without re-running it. Re-running with "Save &
Run All" and re-pulling, or pasting the printed output directly, will
complete this section with real numbers rather than estimates.*

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /score`

Body: a single flow record — see `FlowInput` in [app.py](app.py) for the full
schema (44 fields matching the UNSW-NB15 columns after cleaning). All numeric
fields are validated as non-negative; `hour` must be `0-23`; the two binary
flag fields (`is_sm_ips_ports`, `is_ftp_login`) must be `0` or `1`.

Normal traffic — stage 1 doesn't flag it, so stage 2 never runs:

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
    {"feature": "smeansz", "shap_value": 0.664},
    {"feature": "sbytes", "shap_value": 0.596},
    {"feature": "ct_srv_dst", "shap_value": 0.372},
    {"feature": "service", "shap_value": 0.236}
  ]
}
```

`risk_score` is `"high"` when `confidence > 0.7`, otherwise `"medium"`.
Invalid input (e.g. negative `dur`, `hour` outside `0-23`) returns `422` with
a Pydantic validation error body, not a `500`.

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
> working directory, including `.venv/` — which hung for 20+ minutes in
> testing given how many files ship with numpy/scipy/pandas/catboost/shap.
> If you want `--reload`, `pip install watchfiles` first.

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

22 tests covering: the stage-1 short-circuit on normal traffic, stage-2
classification + SHAP explanation shape on flows crafted to trigger the
anomaly filter, and input validation (negative values, out-of-range `hour`,
out-of-range binary flags, missing required/aliased fields). Uses
`fastapi.testclient.TestClient` against the app in-process — no running
server required. Runs in CI on every push via `.github/workflows/tests.yml`.

## Project structure

```
fastapi_service/
├── app.py                     # FastAPI app: model loading, scoring endpoint, SHAP explanation
├── test_score.py               # pytest suite (22 tests, in-process via TestClient)
├── model_artifacts/
│   ├── isolation_forest.joblib # stage 1: unsupervised anomaly filter
│   ├── catboost_stage2.cbm     # stage 2: multiclass attack classifier
│   ├── proto_freq.json         # protocol -> training-set frequency, for the proto_freq feature
│   └── feature_columns.json    # canonical feature order + which columns are numeric/categorical
├── requirements.txt            # runtime dependencies (scikit-learn pinned to match the pickled model)
├── requirements-dev.txt        # requirements.txt + pytest/httpx for testing
├── Dockerfile                  # builds a container that serves the API on port 8000
└── .github/workflows/tests.yml # CI: runs the pytest suite on push/PR
```

## Known limitations

- `model_artifacts/isolation_forest.joblib` was pickled with
  scikit-learn 1.6.1; `requirements.txt` pins that exact version.
  Unpickling it under a newer scikit-learn (e.g. 1.9.0) still loads but
  raises `InconsistentVersionWarning`, and prediction behavior isn't
  guaranteed to match — this was observed directly, not theoretical.
- Attack-category confidence is a raw softmax score from `CatBoost`, not a
  calibrated probability — it doesn't reflect true likelihood, and is
  least trustworthy for the classes with weak precision (see Results).
  `CalibratedClassifierCV` would address this but needs the original
  training/validation split, which isn't part of this repo.
- `proto_freq.json` maps protocols to their training-set frequency; a
  protocol never seen during training gets frequency `0.0` at inference,
  which is a reasonable default but not validated against real unseen-protocol
  traffic.
- DoS, Analysis, and Backdoor have the weakest precision of the attack
  categories (see Results). Exact numbers aren't published yet — Kaggle's
  API didn't retain this notebook's execution output, only its source.
- Stage 2's training data is drawn from Stage 1's flagged set, so its
  performance is coupled to the `contamination=0.09` choice; retraining
  Stage 1 at a different threshold without also re-evaluating Stage 2
  invalidates the reported end-to-end numbers.

## License

MIT — see [LICENSE](LICENSE).
