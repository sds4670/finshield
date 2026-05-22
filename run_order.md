# FinShield — Run Order

Run these steps in order after cloning the repo.

---

## Step 0 — Prerequisites

- Python 3.10+
- PostgreSQL 14+ (running locally)
- Node.js 18+

---

## Step 1 — Environment Setup

```bash
cp .env.example .env
# Edit .env and set your PostgreSQL password:
#   DB_PASSWORD=your_postgres_password_here
```

---

## Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Database Setup (run once)

```bash
# Create the database
psql -U postgres -c "CREATE DATABASE finshield;"

# Apply base schema
psql -U postgres -d finshield -f schema.sql

# Apply v2 patch (adds std_txn_amount column)
psql -U postgres -d finshield -f finshield/schema_patch_v2.sql
```

---

## Step 4 — Seed Historical Data (run once)

```bash
cd finshield
python seed_data_v2.py
```

Expected: ~200 users, 50 merchants, 2000 transactions seeded.

---

## Step 5 — Train the ML Model (run once)

```bash
# Still inside finshield/ subfolder
python ml_model.py
```

Expected: Model saved to `finshield/models/`. Prints ROC-AUC ~0.94.

---

## Step 6 — Start the Live Pipeline

Open three separate terminals:

### Terminal 1 — Transaction Processor

```bash
cd finshield
python processor.py
```

### Terminal 2 — FastAPI Backend

```bash
cd finshield
uvicorn api:app --reload
# Runs on http://localhost:8000
```

### Terminal 3 — React Dashboard

```bash
cd dashboard
npm install   # first time only
npm run dev
# Opens http://localhost:5173
```

---

## Summary Table

| Step | Command | Run |
|---|---|---|
| Install deps | `pip install -r requirements.txt` | Once |
| DB schema | `psql ... -f schema.sql` | Once |
| Seed data | `python finshield/seed_data_v2.py` | Once |
| Train model | `python finshield/ml_model.py` | Once |
| Pipeline | `python finshield/processor.py` | Every session |
| API | `uvicorn api:app --reload` | Every session |
| Dashboard | `npm run dev` (in dashboard/) | Every session |

---

**Note:** `transaction_generator.py` is the Day 2 standalone generator —
it is superseded by `processor.py` which runs the full pipeline.
You do not need to run both.
