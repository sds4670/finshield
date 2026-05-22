# -*- coding: utf-8 -*-
"""
transaction_generator.py — Day 2
=================================
Simulates a live payment stream by generating one transaction
every 1-3 seconds and writing it to the database.

Run this in a separate terminal while your dashboard is open:
    python transaction_generator.py

Stop it with Ctrl+C anytime.

NOTE: This file is superseded by processor.py (Day 5) which runs
the full pipeline including the ML model. Use processor.py instead.
"""

import sys
import io
import os
from dotenv import load_dotenv

# Force UTF-8 output so emojis don't crash on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Load .env from project root (one level up from this file)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

import psycopg2
import psycopg2.extras
import random
import uuid
import time
from datetime import datetime, timedelta
from collections import defaultdict
from faker import Faker

from fraud_scoring import (
    UserContext, MerchantContext, TransactionContext,
    compute_fraud_score
)

fake = Faker('en_IN')
random.seed()   # No fixed seed here — we want different data every run

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     5432,
    "database": os.getenv("DB_NAME", "finshield"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}


# ============================================================
# CONCEPT 1 — HOURLY TRAFFIC WEIGHTS
# Transaction probability by hour of day
# Based on real Indian digital payment patterns
# ============================================================
HOURLY_WEIGHTS = {
    0:  0.2,   # midnight        — very low
    1:  0.1,   # 1am             — near zero
    2:  0.1,   # 2am             — near zero
    3:  0.1,   # 3am             — near zero (fraud peak though)
    4:  0.2,   # 4am             — very low
    5:  0.3,   # 5am             — early risers
    6:  0.6,   # 6am             — morning starts
    7:  0.8,   # 7am             — commute begins
    8:  1.0,   # 8am             — morning spike
    9:  1.2,   # 9am             — peak morning
    10: 1.1,   # 10am            — high
    11: 1.0,   # 11am            — steady
    12: 1.3,   # noon            — lunch peak
    13: 1.2,   # 1pm             — post lunch
    14: 0.9,   # 2pm             — afternoon dip
    15: 0.9,   # 3pm             — steady
    16: 1.0,   # 4pm             — picks up
    17: 1.1,   # 5pm             — evening commute
    18: 1.3,   # 6pm             — evening peak
    19: 1.4,   # 7pm             — highest traffic
    20: 1.4,   # 8pm             — online shopping peak
    21: 1.2,   # 9pm             — still high
    22: 0.8,   # 10pm            — winding down
    23: 0.4,   # 11pm            — late night
}

def get_sleep_duration() -> float:
    """
    Returns how long to sleep before next transaction.
    
    During peak hours (weight=1.4) → shorter sleep = more transactions
    During off hours (weight=0.1) → longer sleep = fewer transactions
    
    Why this matters:
    Your fraud rate by hour chart in the dashboard will show realistic
    patterns — not flat uniform traffic. This is what separates a
    portfolio project from a toy demo.
    """
    hour = datetime.now().hour
    weight = HOURLY_WEIGHTS.get(hour, 1.0)
    
    # Base sleep is 2 seconds. Weight adjusts it.
    # weight=1.4 → sleep = 2/1.4 = 1.4 seconds (faster)
    # weight=0.1 → sleep = 2/0.1 = 20 seconds (slower)
    # Clamped to 0.5-8 seconds for demo purposes
    sleep = 2.0 / weight
    sleep = max(0.5, min(8.0, sleep))
    
    # Add jitter — real traffic is never perfectly regular
    sleep += random.uniform(-0.3, 0.3)
    return max(0.3, sleep)


# ============================================================
# CONCEPT 2 — IN-MEMORY VELOCITY TRACKER
# Tracks recent transactions per user without querying DB
# ============================================================
class VelocityTracker:
    """
    Maintains a rolling window of recent transactions per user.
    
    Why in-memory instead of querying the DB?
    - DB query for velocity: ~50-200ms (too slow for real-time)
    - In-memory lookup: <1ms (instant)
    
    This pattern is called a "hot cache" — fast local state that
    mirrors a subset of what's in the database.
    
    Tradeoff: if you restart the generator, velocity history resets.
    In production, this would be Redis. For your project, this is fine.
    """
    def __init__(self):
        # user_id → list of (timestamp, amount, merchant_id)
        self.history = defaultdict(list)
    
    def record(self, user_id: int, amount: float, merchant_id: int):
        now = datetime.now()
        self.history[user_id].append((now, amount, merchant_id))
        # Clean up entries older than 24 hours to prevent memory bloat
        cutoff = now - timedelta(hours=24)
        self.history[user_id] = [
            h for h in self.history[user_id] if h[0] > cutoff
        ]
    
    def count_in_window(self, user_id: int, seconds: int) -> int:
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return sum(1 for h in self.history[user_id] if h[0] > cutoff)
    
    def amount_in_window(self, user_id: int, seconds: int) -> float:
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return sum(h[1] for h in self.history[user_id] if h[0] > cutoff)
    
    def seen_merchant(self, user_id: int, merchant_id: int) -> bool:
        return any(h[2] == merchant_id for h in self.history[user_id])


# ============================================================
# CONCEPT 3 — FRAUD SCENARIO INJECTION
# Occasionally inject realistic fraud attack patterns
# This ensures your live stream has interesting fraud events
# ============================================================
class FraudScenarioInjector:
    """
    Every N transactions, inject a realistic fraud scenario.
    
    Why inject scenarios separately from normal scoring?
    Because purely random generation might go 10 minutes without
    any fraud — boring for a demo. Scenarios guarantee interesting
    events appear at a controllable rate.
    
    Think of this as a "chaos monkey" for fraud.
    """
    
    def __init__(self, injection_rate=0.08):
        # 8% of transactions will be from a fraud scenario
        self.injection_rate = injection_rate
    
    def should_inject(self) -> bool:
        return random.random() < self.injection_rate
    
    def get_scenario(self, user, merchants) -> dict:
        """
        Returns transaction overrides for a specific fraud scenario.
        Each scenario mimics a real-world attack pattern.
        """
        scenarios = [
            self._card_testing_scenario,
            self._account_takeover_scenario,
            self._bust_out_scenario,
            self._odd_hour_scenario,
        ]
        scenario_fn = random.choice(scenarios)
        return scenario_fn(user, merchants)
    
    def _card_testing_scenario(self, user, merchants) -> dict:
        """
        Card Testing: attacker makes many tiny transactions quickly
        to verify stolen card works before making large purchase.
        Signal: very small amounts, high velocity
        """
        return {
            'amount': round(random.uniform(1, 50), 2),
            'device_type': 'unknown',
            'hour_of_day': random.choice([2, 3, 4]),
            'scenario_name': 'CARD_TESTING'
        }
    
    def _account_takeover_scenario(self, user, merchants) -> dict:
        """
        Account Takeover: attacker logs into victim's account,
        uses new device, makes large purchase immediately.
        Signal: new device + high amount + odd hour
        """
        return {
            'amount': round(user['avg_txn_amount'] * random.uniform(5, 10), 2),
            'device_type': 'unknown',
            'hour_of_day': random.choice([1, 2, 3, 4]),
            'scenario_name': 'ACCOUNT_TAKEOVER'
        }
    
    def _bust_out_scenario(self, user, merchants) -> dict:
        """
        Bust-Out Fraud: fraudster opens account, builds credit history,
        then makes maximum purchase and disappears.
        Signal: large amount at new merchant
        """
        return {
            'amount': round(random.uniform(80000, 200000), 2),
            'device_type': 'web',
            'hour_of_day': datetime.now().hour,
            'scenario_name': 'BUST_OUT'
        }
    
    def _odd_hour_scenario(self, user, merchants) -> dict:
        """
        Odd Hour Transaction: normal-looking transaction but at 3am.
        Moderate risk — could be legitimate night-shift worker.
        Signal: odd hour only
        """
        return {
            'amount': round(user['avg_txn_amount'] * random.uniform(0.8, 2.0), 2),
            'device_type': random.choice(['mobile', 'web']),
            'hour_of_day': random.choice([1, 2, 3, 4]),
            'scenario_name': 'ODD_HOUR'
        }


# ============================================================
# DATABASE HELPERS
# ============================================================
def load_users(conn) -> list:
    """Load all users from DB once at startup — not on every transaction.
    
    Tries to fetch std_txn_amount (added by schema_patch_v2.sql).
    Falls back to computing it as 35% of avg if the column doesn't exist yet.
    """
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cursor.execute("""
            SELECT user_id, account_age_days, avg_txn_amount,
                   std_txn_amount, total_txns, risk_tier
            FROM users
        """)
    except Exception:
        # schema_patch_v2.sql not applied yet — compute std from avg
        conn.rollback()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""
            SELECT user_id, account_age_days, avg_txn_amount,
                   (avg_txn_amount * 0.35) AS std_txn_amount,
                   total_txns, risk_tier
            FROM users
        """)
    return cursor.fetchall()


def load_merchants(conn) -> list:
    """Load all merchants from DB once at startup."""
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
        SELECT merchant_id, fraud_rate, is_blacklisted,
               EXTRACT(DAY FROM NOW() - registered_date)::INT as reg_days,
               category
        FROM merchants
    """)
    return cursor.fetchall()


def write_transaction(conn, txn_id, user_id, merchant_id, amount,
                       device_type, hour, day, status, result, scenario=None):
    """
    Writes one complete transaction + fraud data to DB.
    
    Uses a single connection transaction (BEGIN/COMMIT) so either
    ALL three inserts succeed or NONE do. This is called atomicity —
    you never want a transaction row without its risk_score row.
    """
    cursor = conn.cursor()
    
    try:
        # Insert transaction
        cursor.execute("""
            INSERT INTO transactions 
            (transaction_id, user_id, merchant_id, amount, currency,
             device_type, ip_address, hour_of_day, day_of_week, status, created_at)
            VALUES (%s, %s, %s, %s, 'INR', %s, %s, %s, %s, %s, NOW())
        """, (
            txn_id, user_id, merchant_id, amount,
            device_type, fake.ipv4(), hour, day, status
        ))

        # Insert risk score (always)
        cursor.execute("""
            INSERT INTO risk_scores
            (transaction_id, model_version, score, threshold_used, decision, scored_at)
            VALUES (%s, 'xgb_v1.0', %s, %s, %s, NOW())
        """, (
            txn_id,
            result['score'],
            result['threshold'],
            'FRAUD' if status != 'APPROVED' else 'LEGIT'
        ))

        # Insert fraud flag (only if flagged/blocked)
        if status in ('FLAGGED', 'BLOCKED'):
            reason = scenario or result.get('reason') or 'ML_SCORE_THRESHOLD'
            cursor.execute("""
                INSERT INTO fraud_flags
                (transaction_id, rule_triggered, ml_score, final_decision, reviewed_by, created_at)
                VALUES (%s, %s, %s, %s, 'SYSTEM', NOW())
            """, (txn_id, reason, result['score'], status))

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"  ❌ DB write failed: {e}")
        return False


# ============================================================
# MAIN GENERATOR LOOP
# ============================================================
def run_generator():
    print("[START] FinShield Transaction Generator Starting...\n")
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Load reference data ONCE at startup
    # Why once? Users and merchants don't change during a streaming session.
    # Loading them per-transaction would be wasteful.
    print("Loading users and merchants from DB...")
    users = load_users(conn)
    merchants = load_merchants(conn)
    print(f"[OK] Loaded {len(users)} users, {len(merchants)} merchants\n")
    
    if not users or not merchants:
        print("[ERR] No users or merchants found. Run seed_data_v2.py first.")
        return

    # Initialise helpers
    velocity = VelocityTracker()
    injector = FraudScenarioInjector(injection_rate=0.08)
    
    # Stats tracking
    total = 0
    fraud_count = 0
    start_time = datetime.now()
    
    print("Streaming transactions... (Ctrl+C to stop)\n")
    print(f"{'#':<6} {'Amount':>10} {'Status':<10} {'Score':>6} {'Reason':<30} {'Time'}")
    print("-" * 80)

    try:
        while True:
            # Pick random user and merchant
            db_user = random.choice(users)
            db_merchant = random.choice(merchants)
            
            user_id = db_user['user_id']
            merchant_id = db_merchant['merchant_id']
            now = datetime.now()
            
            # Default transaction properties
            avg_amt = float(db_user['avg_txn_amount'] or 1000)
            std_amt = float(db_user['std_txn_amount'] or avg_amt * 0.4)
            
            amount = abs(round(random.gauss(avg_amt, std_amt), 2))
            amount = max(10, amount)
            device_type = random.choice(['mobile', 'web', 'POS'])
            hour = now.hour
            scenario_name = None

            # Fraud scenario injection
            if injector.should_inject():
                overrides = injector.get_scenario(
                    {'avg_txn_amount': avg_amt},
                    merchants
                )
                amount = overrides.get('amount', amount)
                device_type = overrides.get('device_type', device_type)
                hour = overrides.get('hour_of_day', hour)
                scenario_name = overrides.get('scenario_name')

            # Build context objects for scoring
            user_ctx = UserContext(
                user_id=user_id,
                account_age_days=db_user['account_age_days'] or 1,
                avg_txn_amount=avg_amt,
                std_txn_amount=std_amt,
                total_txns=db_user['total_txns'] or 0,
                risk_tier=db_user['risk_tier'] or 'MEDIUM',
                known_device_types=['mobile', 'web']
            )

            merchant_ctx = MerchantContext(
                merchant_id=merchant_id,
                fraud_rate=float(db_merchant['fraud_rate'] or 0),
                is_blacklisted=db_merchant['is_blacklisted'] or False,
                registered_days_ago=int(db_merchant['reg_days'] or 30),
                category=db_merchant['category'] or 'Other'
            )

            txn_ctx = TransactionContext(
                amount=amount,
                hour_of_day=hour,
                day_of_week=now.weekday(),
                device_type=device_type,
                velocity_1min=velocity.count_in_window(user_id, 60),
                velocity_5min=velocity.count_in_window(user_id, 300),
                velocity_1hr=velocity.count_in_window(user_id, 3600),
                amount_sum_1hr=velocity.amount_in_window(user_id, 3600),
                is_first_txn_at_merchant=not velocity.seen_merchant(user_id, merchant_id)
            )

            # Score the transaction
            result = compute_fraud_score(txn_ctx, user_ctx, merchant_ctx, add_noise=False)
            status = result['decision']
            
            # Override status if scenario forces it
            if scenario_name in ('CARD_TESTING', 'ACCOUNT_TAKEOVER', 'BUST_OUT'):
                if result['score'] < 0.5:
                    result['score'] = round(random.uniform(0.55, 0.85), 4)
                    status = 'FLAGGED' if result['score'] < 0.75 else 'BLOCKED'

            # Write to DB
            txn_id = str(uuid.uuid4())
            success = write_transaction(
                conn, txn_id, user_id, merchant_id, amount,
                device_type, hour, now.weekday(), status, result, scenario_name
            )

            if success:
                # Update velocity tracker
                velocity.record(user_id, amount, merchant_id)
                
                total += 1
                if status != 'APPROVED':
                    fraud_count += 1

                # Console output
                status_icon = '🟢' if status == 'APPROVED' else ('🟡' if status == 'FLAGGED' else '🔴')
                reason = scenario_name or result.get('reason') or '-'
                print(
                    f"{total:<6} "
                    f"₹{amount:>9,.0f} "
                    f"{status_icon} {status:<8} "
                    f"{result['score']:>5.3f}  "
                    f"{str(reason):<30} "
                    f"{now.strftime('%H:%M:%S')}"
                )

                # Summary every 50 transactions
                if total % 50 == 0:
                    elapsed = (datetime.now() - start_time).seconds
                    rate = round(total / elapsed * 60, 1) if elapsed > 0 else 0
                    print(f"\n  📊 {total} txns | {fraud_count} fraud ({round(fraud_count/total*100,1)}%) | {rate} txns/min\n")

            # Sleep based on realistic hourly traffic pattern
            sleep_time = get_sleep_duration()
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        elapsed = (datetime.now() - start_time).seconds
        print(f"\n\n⛔ Generator stopped.")
        print(f"   Total transactions: {total}")
        print(f"   Fraud detected:     {fraud_count} ({round(fraud_count/total*100,1) if total else 0}%)")
        print(f"   Runtime:            {elapsed}s")
        conn.close()


if __name__ == "__main__":
    run_generator()
