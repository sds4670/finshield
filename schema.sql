-- ============================================================
-- FinShield: Real-Time Fraud Detection Pipeline
-- schema.sql — Run this first to set up your database
-- ============================================================

-- Drop tables if re-running (safe reset)
DROP TABLE IF EXISTS risk_scores CASCADE;
DROP TABLE IF EXISTS fraud_flags CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS merchants CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- ============================================================
-- TABLE 1: users
-- Stores registered users of the payment platform
-- risk_tier is pre-computed from account history
-- ============================================================
CREATE TABLE users (
    user_id         SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    email           VARCHAR(150) UNIQUE NOT NULL,
    phone           VARCHAR(15),
    account_age_days INT DEFAULT 0,          -- how long they've been on platform
    avg_txn_amount  NUMERIC(10, 2) DEFAULT 0, -- rolling average transaction amount
    total_txns      INT DEFAULT 0,            -- total transactions made
    risk_tier       VARCHAR(10) DEFAULT 'MEDIUM' CHECK (risk_tier IN ('LOW', 'MEDIUM', 'HIGH')),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- WHY risk_tier here?
-- Instead of computing this every time a transaction comes in,
-- we pre-label users. New users = MEDIUM (unknown), old users with
-- clean history = LOW, users with past flags = HIGH.


-- ============================================================
-- TABLE 2: merchants
-- Stores merchant/vendor information
-- fraud_rate is updated after every flagged transaction
-- ============================================================
CREATE TABLE merchants (
    merchant_id         SERIAL PRIMARY KEY,
    name                VARCHAR(150) NOT NULL,
    category            VARCHAR(50),           -- e.g. Electronics, Food, Travel
    city                VARCHAR(80),
    registered_date     DATE DEFAULT CURRENT_DATE,
    total_transactions  INT DEFAULT 0,
    fraud_count         INT DEFAULT 0,
    fraud_rate          NUMERIC(5, 4) DEFAULT 0.0000,  -- e.g. 0.0312 = 3.12%
    is_blacklisted      BOOLEAN DEFAULT FALSE
);

-- WHY fraud_rate stored here?
-- During ML feature engineering, merchant_fraud_rate is one of the
-- strongest signals. Storing it avoids expensive JOIN + COUNT every
-- time a new transaction arrives.


-- ============================================================
-- TABLE 3: transactions
-- Core table — every payment event lands here first
-- status tracks the lifecycle: PENDING → APPROVED/FLAGGED/BLOCKED
-- ============================================================
CREATE TABLE transactions (
    transaction_id  VARCHAR(36) PRIMARY KEY,   -- UUID format
    user_id         INT REFERENCES users(user_id),
    merchant_id     INT REFERENCES merchants(merchant_id),
    amount          NUMERIC(12, 2) NOT NULL,
    currency        VARCHAR(5) DEFAULT 'INR',
    device_type     VARCHAR(20) CHECK (device_type IN ('mobile', 'web', 'POS', 'unknown')),
    ip_address      VARCHAR(45),               -- supports IPv6 too
    hour_of_day     INT,                       -- extracted at insert time (0-23)
    day_of_week     INT,                       -- 0=Monday, 6=Sunday
    status          VARCHAR(10) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'APPROVED', 'FLAGGED', 'BLOCKED')),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- WHY pre-extract hour_of_day and day_of_week?
-- These are critical ML features. Extracting them at write time
-- means your model pipeline doesn't have to compute them on every read.
-- This is called "feature pre-computation" — standard in production ML.

-- Index on user_id for fast velocity checks (how many txns in last 1hr?)
CREATE INDEX idx_txn_user_id ON transactions(user_id);
CREATE INDEX idx_txn_created_at ON transactions(created_at);
CREATE INDEX idx_txn_status ON transactions(status);


-- ============================================================
-- TABLE 4: fraud_flags
-- Only created when a transaction is suspicious
-- Separating this keeps transactions table clean and fast
-- ============================================================
CREATE TABLE fraud_flags (
    flag_id         SERIAL PRIMARY KEY,
    transaction_id  VARCHAR(36) REFERENCES transactions(transaction_id),
    rule_triggered  VARCHAR(100),   -- e.g. "VELOCITY_BREACH" or "HIGH_AMOUNT_NEW_MERCHANT"
    ml_score        NUMERIC(5, 4),  -- XGBoost probability output e.g. 0.8732
    final_decision  VARCHAR(10) CHECK (final_decision IN ('FLAGGED', 'BLOCKED', 'CLEARED')),
    reviewed_by     VARCHAR(50) DEFAULT 'SYSTEM',  -- SYSTEM or analyst name
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- WHY separate from transactions?
-- In a real system, 97%+ of transactions are clean.
-- If you stored fraud fields in transactions, you'd have NULL columns
-- on almost every row — wasteful and slow.
-- Separate table = only created when needed = efficient.


-- ============================================================
-- TABLE 5: risk_scores
-- Stores the ML model's output for every scored transaction
-- model_version lets you track which model made the call
-- ============================================================
CREATE TABLE risk_scores (
    score_id        SERIAL PRIMARY KEY,
    transaction_id  VARCHAR(36) REFERENCES transactions(transaction_id),
    model_version   VARCHAR(20) DEFAULT 'xgb_v1.0',
    score           NUMERIC(5, 4) NOT NULL,    -- 0.0000 to 1.0000
    threshold_used  NUMERIC(5, 4) DEFAULT 0.5, -- what cutoff was used
    decision        VARCHAR(10),               -- FRAUD or LEGIT based on threshold
    scored_at       TIMESTAMP DEFAULT NOW()
);

-- WHY store model_version and threshold_used?
-- In production, models are retrained. If your v1 model flagged something
-- but v2 cleared it, you need to know which version was responsible.
-- threshold_used matters because fraud teams adjust thresholds seasonally
-- (e.g. stricter during festival season when fraud spikes).
