"""
api.py — FastAPI Backend for FinShield Dashboard
=================================================
Exposes PostgreSQL data as JSON endpoints for the React frontend.

Run:
    pip install fastapi uvicorn psycopg2-binary
    uvicorn api:app --reload --port 8000

Endpoints:
    GET /api/live-feed          → last 50 transactions
    GET /api/stats              → summary counters
    GET /api/merchant-insights  → fraud by merchant/category
    GET /api/risk-analytics     → hourly heatmap, score distribution
    GET /api/model-insights     → SHAP top features, flagged explanations
    GET /api/rule-breakdown     → rule trigger counts
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import psycopg2.extras
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FinShield API")

# Allow React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     5432,
    "database": os.getenv("DB_NAME", "finshield"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def query(sql, params=None):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def query_one(sql, params=None):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params)
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


# ── Live Feed ─────────────────────────────────────────────────

@app.get("/api/live-feed")
def live_feed():
    rows = query("""
        SELECT
            t.transaction_id,
            u.name as user_name,
            m.name as merchant_name,
            m.category,
            t.amount,
            t.status,
            t.device_type,
            t.hour_of_day,
            t.created_at,
            rs.score,
            ff.rule_triggered,
            ff.notes
        FROM transactions t
        JOIN users u ON t.user_id = u.user_id
        JOIN merchants m ON t.merchant_id = m.merchant_id
        LEFT JOIN risk_scores rs ON t.transaction_id = rs.transaction_id
        LEFT JOIN fraud_flags ff ON t.transaction_id = ff.transaction_id
        ORDER BY t.created_at DESC
        LIMIT 50
    """)
    # Convert datetime to string for JSON serialisation
    for r in rows:
        if r.get('created_at'):
            r['created_at'] = r['created_at'].strftime('%H:%M:%S')
        if r.get('amount'):
            r['amount'] = float(r['amount'])
        if r.get('score'):
            r['score'] = float(r['score'])
    return rows


# ── Summary Stats ─────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    totals = query_one("""
        SELECT
            COUNT(*)                                                      as total,
            SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END)        as approved,
            SUM(CASE WHEN status = 'FLAGGED'  THEN 1 ELSE 0 END)        as flagged,
            SUM(CASE WHEN status = 'BLOCKED'  THEN 1 ELSE 0 END)        as blocked,
            ROUND(AVG(CASE WHEN status != 'APPROVED' THEN 1.0 ELSE 0 END) * 100, 2) as fraud_rate,
            ROUND(AVG(rs.score)::numeric, 4)                             as avg_score,
            SUM(t.amount)                                                 as total_volume,
            SUM(CASE WHEN status != 'APPROVED' THEN t.amount ELSE 0 END) as fraud_volume
        FROM transactions t
        LEFT JOIN risk_scores rs ON t.transaction_id = rs.transaction_id
    """)

    recent = query_one("""
        SELECT COUNT(*) as recent_count
        FROM transactions
        WHERE created_at > NOW() - INTERVAL '5 minutes'
    """)

    result = totals or {}
    result['recent_5min'] = recent.get('recent_count', 0) if recent else 0
    for k in ['total_volume', 'fraud_volume', 'avg_score', 'fraud_rate']:
        if result.get(k):
            result[k] = float(result[k])
    return result


# ── Merchant Insights ─────────────────────────────────────────

@app.get("/api/merchant-insights")
def merchant_insights():
    by_category = query("""
        SELECT
            m.category,
            COUNT(*)                                                        as total_txns,
            SUM(CASE WHEN t.status != 'APPROVED' THEN 1 ELSE 0 END)       as fraud_count,
            ROUND(AVG(CASE WHEN t.status != 'APPROVED' THEN 1.0 ELSE 0 END) * 100, 2) as fraud_rate,
            ROUND(AVG(t.amount)::numeric, 2)                               as avg_amount
        FROM transactions t
        JOIN merchants m ON t.merchant_id = m.merchant_id
        GROUP BY m.category
        ORDER BY fraud_rate DESC
    """)

    top_risky = query("""
        SELECT
            m.name,
            m.category,
            m.city,
            COUNT(*)                                                        as total_txns,
            SUM(CASE WHEN t.status != 'APPROVED' THEN 1 ELSE 0 END)       as fraud_count,
            ROUND(m.fraud_rate * 100, 2)                                   as fraud_rate_pct,
            m.is_blacklisted
        FROM merchants m
        JOIN transactions t ON m.merchant_id = t.merchant_id
        GROUP BY m.merchant_id, m.name, m.category, m.city, m.fraud_rate, m.is_blacklisted
        ORDER BY m.fraud_rate DESC
        LIMIT 10
    """)

    volume_vs_fraud = query("""
        SELECT
            m.name,
            m.category,
            COUNT(*) as txn_count,
            ROUND(m.fraud_rate * 100, 2) as fraud_rate_pct,
            ROUND(AVG(t.amount)::numeric, 0) as avg_amount
        FROM merchants m
        JOIN transactions t ON m.merchant_id = t.merchant_id
        GROUP BY m.merchant_id, m.name, m.category, m.fraud_rate
        HAVING COUNT(*) > 5
        ORDER BY txn_count DESC
        LIMIT 30
    """)

    for r in by_category + top_risky + volume_vs_fraud:
        for k in ['fraud_rate', 'avg_amount', 'fraud_rate_pct']:
            if r.get(k) is not None:
                r[k] = float(r[k])

    return {
        "by_category": by_category,
        "top_risky_merchants": top_risky,
        "volume_vs_fraud": volume_vs_fraud
    }


# ── Risk Analytics ────────────────────────────────────────────

@app.get("/api/risk-analytics")
def risk_analytics():
    hourly = query("""
        SELECT
            hour_of_day,
            COUNT(*) as total,
            SUM(CASE WHEN status != 'APPROVED' THEN 1 ELSE 0 END) as fraud_count,
            ROUND(AVG(CASE WHEN status != 'APPROVED' THEN 1.0 ELSE 0 END) * 100, 2) as fraud_rate
        FROM transactions
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """)

    score_dist = query("""
        SELECT
            CASE
                WHEN score < 0.2  THEN '0.0-0.2'
                WHEN score < 0.4  THEN '0.2-0.4'
                WHEN score < 0.6  THEN '0.4-0.6'
                WHEN score < 0.8  THEN '0.6-0.8'
                ELSE '0.8-1.0'
            END as bucket,
            COUNT(*) as count
        FROM risk_scores
        GROUP BY bucket
        ORDER BY bucket
    """)

    by_device = query("""
        SELECT
            device_type,
            COUNT(*) as total,
            SUM(CASE WHEN status != 'APPROVED' THEN 1 ELSE 0 END) as fraud_count,
            ROUND(AVG(CASE WHEN status != 'APPROVED' THEN 1.0 ELSE 0 END) * 100, 2) as fraud_rate
        FROM transactions
        GROUP BY device_type
        ORDER BY fraud_rate DESC
    """)

    trend = query("""
        SELECT
            DATE_TRUNC('minute', created_at) as minute,
            COUNT(*) as total,
            SUM(CASE WHEN status != 'APPROVED' THEN 1 ELSE 0 END) as fraud_count
        FROM transactions
        WHERE created_at > NOW() - INTERVAL '30 minutes'
        GROUP BY minute
        ORDER BY minute
    """)

    for r in hourly + score_dist + by_device:
        for k in ['fraud_rate']:
            if r.get(k) is not None:
                r[k] = float(r[k])
    for r in trend:
        if r.get('minute'):
            r['minute'] = r['minute'].strftime('%H:%M')

    return {
        "hourly_heatmap": hourly,
        "score_distribution": score_dist,
        "by_device": by_device,
        "fraud_trend": trend
    }


# ── Model Insights ────────────────────────────────────────────

@app.get("/api/model-insights")
def model_insights():
    rule_breakdown = query("""
        SELECT
            rule_triggered,
            COUNT(*) as count,
            ROUND(AVG(ml_score)::numeric, 4) as avg_ml_score,
            SUM(CASE WHEN final_decision = 'BLOCKED' THEN 1 ELSE 0 END) as blocked_count
        FROM fraud_flags
        WHERE rule_triggered IS NOT NULL
        GROUP BY rule_triggered
        ORDER BY count DESC
    """)

    recent_flagged = query("""
        SELECT
            t.transaction_id,
            u.name as user_name,
            m.name as merchant_name,
            t.amount,
            t.status,
            t.hour_of_day,
            t.device_type,
            ff.rule_triggered,
            ff.ml_score,
            ff.notes,
            ff.final_decision,
            t.created_at
        FROM fraud_flags ff
        JOIN transactions t  ON ff.transaction_id = t.transaction_id
        JOIN users u         ON t.user_id = u.user_id
        JOIN merchants m     ON t.merchant_id = m.merchant_id
        ORDER BY t.created_at DESC
        LIMIT 20
    """)

    score_by_rule = query("""
        SELECT
            rule_triggered as rule,
            ROUND(AVG(ml_score)::numeric, 3) as avg_score,
            COUNT(*) as count
        FROM fraud_flags
        GROUP BY rule_triggered
        ORDER BY avg_score DESC
    """)

    for r in rule_breakdown + recent_flagged + score_by_rule:
        for k in ['avg_ml_score', 'ml_score', 'avg_score', 'amount']:
            if r.get(k) is not None:
                r[k] = float(r[k])
        if r.get('created_at'):
            r['created_at'] = r['created_at'].strftime('%H:%M:%S')

    return {
        "rule_breakdown": rule_breakdown,
        "recent_flagged": recent_flagged,
        "score_by_rule": score_by_rule
    }


# Local demo inference. Public deployment requires authentication/rate limiting.
from functools import lru_cache
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from fastapi import HTTPException


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    amount: float = Field(gt=0, le=1e9)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    device_type: Literal['mobile', 'web', 'POS', 'unknown']
    account_age_days: int = Field(ge=0)
    avg_txn_amount: float = Field(ge=0)
    std_txn_amount: float = Field(ge=0)
    total_txns: int = Field(ge=0)
    risk_tier: Literal['LOW', 'MEDIUM', 'HIGH']
    fraud_rate: float = Field(ge=0, le=1)
    is_blacklisted: bool
    merchant_age_days: int = Field(ge=0)


@lru_cache(maxsize=1)
def get_scorer():
    from ml_model import FraudModelScorer
    return FraudModelScorer()


@app.post('/api/predict')
def predict(transaction: TransactionInput):
    """ML-only synthetic-demo score and SHAP impacts, not a hybrid decision."""
    try:
        scorer = get_scorer()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail='Model unavailable; run training first')
    return scorer.score(transaction.model_dump())


@app.get('/api/health')
def health():
    try:
        query_one('SELECT 1 AS ready')
    except psycopg2.Error:
        raise HTTPException(status_code=503, detail='Database unavailable')
    return {'status': 'ok', 'database': 'ready'}
