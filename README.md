# FinShield — Explainable Fraud Detection

A portfolio project demonstrating transaction processing, rule-based decisions,
XGBoost inference with SHAP explanations, and reproducible ML evaluation.
Python 3.11 · PostgreSQL 14+ · FastAPI · React · Node 24

**Scope:** the live dashboard uses synthetic transactions and policy-generated
labels. Its metrics measure policy imitation, not proven fraud detection.
The separate benchmark accepts independently labeled CSV data. No independently
labeled public-data benchmark result is claimed in this repository yet.

## Two workflows

| Workflow | Purpose | Entry point |
|---|---|---|
| Live demo | Explore synthetic transactions, hybrid decisions and SHAP explanations | `finshield/processor.py`, `finshield/api.py`, `dashboard/` |
| Offline benchmark | Compare models using chronological splits and validation-selected thresholds | `finshield/benchmark.py` |

Read the [engineering and interview guide](docs/ENGINEERING_GUIDE.md) for architecture,
ML reasoning, alternatives, validation evidence, limitations, and interview answers.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env  # configure your own local PostgreSQL connection
createdb -U postgres finshield
psql -U postgres -d finshield -v ON_ERROR_STOP=1 -f schema.sql -f finshield/schema_patch_v2.sql
cd finshield
python seed_data_v2.py
python ml_model.py
```

**Apply `schema.sql` only to a new/disposable database: it drops existing tables.**
Seed once; the seed script is not an idempotent migration tool.

Start these in separate terminals with the virtual environment active:

```bash
# From finshield/ (the inner Python directory)
python processor.py
uvicorn api:app --reload
# From dashboard/
npm ci
npm run dev
```

The development dashboard calls the API on localhost port 8000. This is a local
demo, not a configured public deployment. No actual payments are processed.

## Run an independently labeled benchmark

Obtain a legally usable labeled dataset yourself and record its source, version,
license, label definition and download hash. A compatible example schema is the
ULB credit-card dataset's `Time`, `Class`, `Amount`, and `V1`–`V28` columns; its
availability, license and suitability must be checked before use. Do not invent
transaction times for datasets without event timestamps.

```bash
python finshield/benchmark.py \
  --csv data/creditcard.csv --output models/experiments/run-001 \
  --features Amount V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 V11 V12 V13 V14 V15 V16 V17 V18 V19 V20 V21 V22 V23 V24 V25 V26 V27 V28 \
  --provenance "Record actual source URL, dataset version, license and label origin here" \
  --fp-cost 1 --fn-cost 25
```

The script trains prior, logistic-regression and XGBoost baselines, selects thresholds
and a candidate on validation cost, then reports held-out test results. Outputs:
`report.json`, `model.joblib`, and `requirements-resolved.txt`. Use a fresh output
directory for every run. Costs are illustrative units, not measured financial losses.
Explicit features must be audited for leakage; the program cannot infer their meaning.
These artifacts **do not replace** the live demo's model or feature contract.

For a dataset-free wiring check:

```bash
python tests/benchmark_smoke.py --output /tmp/finshield-smoke-001
```

This generates synthetic Gaussian data; its metrics are not fraud benchmarks.

## Prediction API

`POST /api/predict` returns the **ML-only** demo score, decision, model version,
SHAP impacts in log-odds, and a synthetic-evaluation disclaimer. It is distinct
from the processor's hybrid policy. Swagger documentation is available at `/docs`
on the running API. `GET /api/health` checks database connectivity.

```json
{
  "amount": 45000, "hour_of_day": 3, "day_of_week": 5,
  "device_type": "unknown", "account_age_days": 5,
  "avg_txn_amount": 800, "std_txn_amount": 300, "total_txns": 2,
  "risk_tier": "MEDIUM", "fraud_rate": 0.08,
  "is_blacklisted": false, "merchant_age_days": 20
}
```

Models are loaded once per API worker, on the first prediction. Missing artifacts
return 503; invalid inputs return 422. Only load trusted local joblib artifacts.

## Validation

```bash
python -m unittest discover -s tests -v
# Requires seeded database and trained demo model:
python tests/smoke_inference.py
# Frontend:
cd dashboard
npm ci
npm run lint
npm run build
```

GitHub Actions defines unit checks, a disposable PostgreSQL seed/train/API
integration job, and frontend install/lint/build checks. A workflow definition is
not evidence that hosted Actions have run; local validation is documented separately.

## Limitations

No independently labeled benchmark completed yet; no calibration, historical
feature snapshots, drift monitoring, authentication, public deployment, or measured
production latency/availability. The hybrid weights and decision thresholds are
heuristics. See the guide before discussing performance in interviews.
