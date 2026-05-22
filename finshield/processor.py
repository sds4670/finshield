"""
processor.py — Day 5
=====================
The central pipeline that wires all FinShield components together.

Architecture:
    Generator Thread  →  Queue  →  Processor Thread  →  Database

Run this instead of transaction_generator.py from Day 5 onwards.
It starts the generator internally and processes each transaction
through the full pipeline.

Usage:
    python processor.py

Stop with Ctrl+C.
"""

import sys
import io
import os
from dotenv import load_dotenv

# Force UTF-8 output to prevent emoji crash on Windows terminals
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Load .env from project root (one level up from this file)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

import time
import uuid
import queue
import threading
import psycopg2
import psycopg2.extras
import random
from datetime import datetime, timedelta
from collections import defaultdict, deque
from faker import Faker

from fraud_scoring import UserContext, MerchantContext, TransactionContext
from rule_engine import RuleEngine
from ml_model import FraudModelScorer, make_final_decision, engineer_features
import pandas as pd

fake = Faker('en_IN')

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     5432,
    "database": os.getenv("DB_NAME", "finshield"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

# ============================================================
# HOURLY TRAFFIC WEIGHTS (same as generator)
# ============================================================
HOURLY_WEIGHTS = {
    0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.3,
    6: 0.6, 7: 0.8, 8: 1.0, 9: 1.2, 10: 1.1, 11: 1.0,
    12: 1.3, 13: 1.2, 14: 0.9, 15: 0.9, 16: 1.0, 17: 1.1,
    18: 1.3, 19: 1.4, 20: 1.4, 21: 1.2, 22: 0.8, 23: 0.4,
}

def get_sleep_duration() -> float:
    hour = datetime.now().hour
    weight = HOURLY_WEIGHTS.get(hour, 1.0)
    sleep = max(0.5, min(8.0, 2.0 / weight))
    return sleep + random.uniform(-0.2, 0.2)


# ============================================================
# VELOCITY TRACKER (same as generator, shared across threads)
# ============================================================
class VelocityTracker:
    """
    Thread-safe in-memory velocity tracker.
    Uses a lock to prevent race conditions between generator
    and processor threads reading/writing simultaneously.

    Why thread safety matters here:
    Without a lock, the generator thread could be writing to
    history while the processor thread is reading it — causing
    corrupted velocity counts or crashes.
    """
    def __init__(self):
        self.history = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, user_id, amount, merchant_id):
        with self._lock:
            now = datetime.now()
            self.history[user_id].append((now, amount, merchant_id))
            cutoff = now - timedelta(hours=24)
            self.history[user_id] = [
                h for h in self.history[user_id] if h[0] > cutoff
            ]

    def count_in_window(self, user_id, seconds):
        with self._lock:
            cutoff = datetime.now() - timedelta(seconds=seconds)
            return sum(1 for h in self.history[user_id] if h[0] > cutoff)

    def amount_in_window(self, user_id, seconds):
        with self._lock:
            cutoff = datetime.now() - timedelta(seconds=seconds)
            return sum(h[1] for h in self.history[user_id] if h[0] > cutoff)

    def seen_merchant(self, user_id, merchant_id):
        with self._lock:
            return any(h[2] == merchant_id for h in self.history[user_id])


# ============================================================
# PIPELINE STATS TRACKER
# Tracks live metrics for the dashboard and console output
# ============================================================
class PipelineStats:
    """
    Tracks rolling statistics across the last N transactions.
    Used by the dashboard to show live metrics without
    querying the database on every refresh.

    Uses deque (double-ended queue) with maxlen — automatically
    drops old entries when full. This is an O(1) rolling window.
    """
    def __init__(self, window=100):
        self.window = window
        self.recent_scores   = deque(maxlen=window)
        self.recent_decisions = deque(maxlen=window)
        self.recent_amounts  = deque(maxlen=window)
        self.total_processed = 0
        self.total_fraud     = 0
        self.total_blocked   = 0
        self.start_time      = datetime.now()
        self._lock = threading.Lock()

    def record(self, score, decision, amount):
        with self._lock:
            self.recent_scores.append(score)
            self.recent_decisions.append(decision)
            self.recent_amounts.append(amount)
            self.total_processed += 1
            if decision in ('FLAGGED', 'BLOCKED'):
                self.total_fraud += 1
            if decision == 'BLOCKED':
                self.total_blocked += 1

    def get_summary(self) -> dict:
        with self._lock:
            elapsed = max((datetime.now() - self.start_time).seconds, 1)
            decisions = list(self.recent_decisions)
            return {
                'total_processed':  self.total_processed,
                'total_fraud':      self.total_fraud,
                'total_blocked':    self.total_blocked,
                'fraud_rate':       round(self.total_fraud / max(self.total_processed, 1) * 100, 2),
                'txns_per_minute':  round(self.total_processed / elapsed * 60, 1),
                'avg_score_recent': round(sum(self.recent_scores) / max(len(self.recent_scores), 1), 4),
                'recent_flagged':   decisions.count('FLAGGED'),
                'recent_blocked':   decisions.count('BLOCKED'),
                'recent_approved':  decisions.count('APPROVED'),
            }


# ============================================================
# FRAUD SCENARIO INJECTOR (same as generator)
# ============================================================
class FraudScenarioInjector:
    def __init__(self, injection_rate=0.08):
        self.injection_rate = injection_rate

    def should_inject(self):
        return random.random() < self.injection_rate

    def get_scenario(self, user_avg):
        scenarios = [
            {'amount': round(random.uniform(1, 50), 2),
             'device_type': 'unknown', 'hour_of_day': random.choice([2, 3, 4]),
             'scenario_name': 'CARD_TESTING'},
            {'amount': round(user_avg * random.uniform(5, 10), 2),
             'device_type': 'unknown', 'hour_of_day': random.choice([1, 2, 3, 4]),
             'scenario_name': 'ACCOUNT_TAKEOVER'},
            {'amount': round(random.uniform(80000, 200000), 2),
             'device_type': 'web', 'hour_of_day': datetime.now().hour,
             'scenario_name': 'BUST_OUT'},
            {'amount': round(user_avg * random.uniform(0.8, 2.0), 2),
             'device_type': random.choice(['mobile', 'web']),
             'hour_of_day': random.choice([1, 2, 3, 4]),
             'scenario_name': 'ODD_HOUR'},
        ]
        return random.choice(scenarios)


# ============================================================
# DATABASE WRITER
# Handles all DB writes for the pipeline
# ============================================================
class DatabaseWriter:
    """
    Manages database writes for processed transactions.

    Uses a separate connection from the generator — each thread
    should have its own DB connection to avoid conflicts.

    Why not share one connection?
    psycopg2 connections are not thread-safe. Two threads writing
    on the same connection can corrupt each other's transactions.
    """
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.conn.autocommit = False   # we manage commits manually

    def write_processed_transaction(self,
                                     txn_id, user_id, merchant_id,
                                     amount, device_type, hour, day,
                                     final_result, rule_result, ml_result):
        """
        Atomically writes transaction + risk score + fraud flag.
        All three succeed or all three roll back.
        """
        cursor = self.conn.cursor()
        try:
            decision = final_result['final_decision']
            final_score = final_result['final_score']

            # 1. Insert transaction with final status
            cursor.execute("""
                INSERT INTO transactions
                (transaction_id, user_id, merchant_id, amount, currency,
                 device_type, ip_address, hour_of_day, day_of_week, status, created_at)
                VALUES (%s, %s, %s, %s, 'INR', %s, %s, %s, %s, %s, NOW())
            """, (
                txn_id, user_id, merchant_id, amount,
                device_type, fake.ipv4(), hour, day, decision
            ))

            # 2. Insert risk score with combined info
            cursor.execute("""
                INSERT INTO risk_scores
                (transaction_id, model_version, score, threshold_used, decision, scored_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                txn_id,
                final_result.get('model_version', 'xgb_v1.0'),
                final_score,
                0.45,
                'FRAUD' if decision != 'APPROVED' else 'LEGIT'
            ))

            # 3. Insert fraud flag if needed
            if decision in ('FLAGGED', 'BLOCKED'):
                # Determine the best reason to log
                rule_factors = rule_result.get('risk_factors', [])
                ml_decision  = ml_result.get('ml_decision', '')
                decided_by   = final_result.get('decided_by', 'COMBINED')

                if decided_by == 'RULE_ENGINE_HARD_BLOCK':
                    reason = rule_result.get('block_reason', 'HARD_BLOCK')
                elif rule_factors:
                    reason = rule_factors[0]   # primary rule that fired
                else:
                    reason = 'ML_SCORE_THRESHOLD'

                # Build notes from SHAP explanation
                explanations = final_result.get('explanations', [])
                notes = None
                if explanations:
                    top = explanations[0]
                    notes = (
                        f"Top driver: {top['feature']} "
                        f"(impact: {top['impact']:+.3f}, "
                        f"value: {top['value']})"
                    )

                cursor.execute("""
                    INSERT INTO fraud_flags
                    (transaction_id, rule_triggered, ml_score,
                     final_decision, reviewed_by, notes, created_at)
                    VALUES (%s, %s, %s, %s, 'SYSTEM', %s, NOW())
                """, (
                    txn_id, reason,
                    ml_result.get('ml_score', final_score),
                    decision, notes
                ))

            self.conn.commit()
            return True

        except Exception as e:
            self.conn.rollback()
            print(f"  ❌ DB write failed: {e}")
            return False

    def close(self):
        self.conn.close()


# ============================================================
# REFERENCE DATA LOADER
# Loads users and merchants once at startup
# ============================================================
def load_reference_data():
    """
    Loads all users and merchants from DB into memory at startup.

    Why load everything upfront?
    The pipeline processes one transaction every 1-3 seconds.
    Querying the DB for user/merchant data on every transaction
    would add 50-200ms latency per transaction.
    Loading upfront = sub-millisecond lookup.

    Tradeoff: if new users/merchants are added during runtime,
    the pipeline won't see them until restarted. Acceptable for
    a portfolio project — in production you'd use a cache with TTL.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""
        SELECT user_id, account_age_days, avg_txn_amount,
               std_txn_amount, total_txns, risk_tier
        FROM users
    """)
    users = cursor.fetchall()

    cursor.execute("""
        SELECT merchant_id, fraud_rate, is_blacklisted,
               EXTRACT(DAY FROM NOW() - registered_date)::INT as reg_days,
               category
        FROM merchants
    """)
    merchants = cursor.fetchall()

    conn.close()
    print(f"✅ Loaded {len(users)} users, {len(merchants)} merchants")
    return list(users), list(merchants)


# ============================================================
# GENERATOR THREAD
# Produces transactions and puts them in the queue
# ============================================================
def generator_thread(txn_queue, users, merchants, velocity, injector, stop_event):
    """
    Runs in a separate thread — continuously generates transactions
    and puts them into the shared queue.

    stop_event: threading.Event — set this to gracefully stop the thread
    """
    while not stop_event.is_set():
        db_user     = random.choice(users)
        db_merchant = random.choice(merchants)

        user_id     = db_user['user_id']
        merchant_id = db_merchant['merchant_id']
        now         = datetime.now()

        avg_amt = float(db_user['avg_txn_amount'] or 1000)
        std_amt = float(db_user['std_txn_amount'] or avg_amt * 0.4)

        amount      = max(10, abs(round(random.gauss(avg_amt, std_amt), 2)))
        device_type = random.choice(['mobile', 'web', 'POS'])
        hour        = now.hour
        scenario    = None

        # Inject fraud scenario occasionally
        if injector.should_inject():
            ov = injector.get_scenario(avg_amt)
            amount      = ov.get('amount', amount)
            device_type = ov.get('device_type', device_type)
            hour        = ov.get('hour_of_day', hour)
            scenario    = ov.get('scenario_name')

        # Build context objects
        user_ctx = UserContext(
            user_id=user_id,
            account_age_days=int(db_user['account_age_days'] or 1),
            avg_txn_amount=avg_amt,
            std_txn_amount=std_amt,
            total_txns=int(db_user['total_txns'] or 0),
            risk_tier=db_user['risk_tier'] or 'MEDIUM',
            known_device_types=['mobile', 'web']
        )
        merchant_ctx = MerchantContext(
            merchant_id=merchant_id,
            fraud_rate=float(db_merchant['fraud_rate'] or 0),
            is_blacklisted=bool(db_merchant['is_blacklisted']),
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

        # Build raw dict for ML feature engineering
        raw_row = {
            'amount':           amount,
            'hour_of_day':      hour,
            'day_of_week':      now.weekday(),
            'device_type':      device_type,
            'account_age_days': user_ctx.account_age_days,
            'avg_txn_amount':   avg_amt,
            'std_txn_amount':   std_amt,
            'total_txns':       user_ctx.total_txns,
            'risk_tier':        user_ctx.risk_tier,
            'fraud_rate':       merchant_ctx.fraud_rate,
            'is_blacklisted':   merchant_ctx.is_blacklisted,
            'merchant_age_days': merchant_ctx.registered_days_ago,
            'category':         merchant_ctx.category,
            'rule_score':       0.0   # placeholder, updated by rule engine
        }

        # Put everything the processor needs into the queue
        txn_queue.put({
            'txn_id':       str(uuid.uuid4()),
            'user_id':      user_id,
            'merchant_id':  merchant_id,
            'day':          now.weekday(),
            'user_ctx':     user_ctx,
            'merchant_ctx': merchant_ctx,
            'txn_ctx':      txn_ctx,
            'raw_row':      raw_row,
            'scenario':     scenario,
            'timestamp':    now
        })

        # Record in velocity tracker
        velocity.record(user_id, amount, merchant_id)

        time.sleep(get_sleep_duration())


# ============================================================
# PROCESSOR THREAD
# Consumes transactions from queue, runs pipeline, writes to DB
# ============================================================
def processor_thread(txn_queue, rule_engine, scorer,
                      db_writer, stats, stop_event):
    """
    Runs in a separate thread — reads from queue, processes each
    transaction through the full pipeline, writes result to DB.
    """
    while not stop_event.is_set():
        try:
            # Block for max 1 second waiting for a transaction
            # timeout prevents the thread from hanging if queue is empty
            item = txn_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        txn_id      = item['txn_id']
        user_ctx    = item['user_ctx']
        merchant_ctx = item['merchant_ctx']
        txn_ctx     = item['txn_ctx']
        raw_row     = item['raw_row']
        scenario    = item['scenario']

        # ---- Step 1: Rule Engine ----
        rule_result_obj = rule_engine.evaluate(txn_ctx, user_ctx, merchant_ctx)
        rule_result = rule_result_obj.to_dict()

        # Update raw_row with rule score for ML
        raw_row['rule_score'] = rule_result['score']

        # ---- Step 2: ML Scorer (if not hard blocked) ----
        ml_result = {}
        if rule_result_obj.send_to_ml:
            try:
                ml_result = scorer.score(raw_row)
            except Exception as e:
                # ML failure shouldn't crash the pipeline
                # Fall back to rule engine decision
                print(f"  ⚠️  ML scoring failed: {e}. Using rule engine only.")
                ml_result = {
                    'ml_score':    rule_result['score'],
                    'ml_decision': rule_result['decision'],
                    'top_explanations': []
                }

        # ---- Step 3: Final Decision ----
        final_result = make_final_decision(rule_result, ml_result)

        # ---- Step 4: Write to DB ----
        db_writer.write_processed_transaction(
            txn_id=txn_id,
            user_id=item['user_id'],
            merchant_id=item['merchant_id'],
            amount=txn_ctx.amount,
            device_type=txn_ctx.device_type,
            hour=txn_ctx.hour_of_day,
            day=item['day'],
            final_result=final_result,
            rule_result=rule_result,
            ml_result=ml_result
        )

        # ---- Step 5: Update stats ----
        stats.record(
            score=final_result['final_score'],
            decision=final_result['final_decision'],
            amount=txn_ctx.amount
        )

        # ---- Console Output ----
        decision = final_result['final_decision']
        icon = {'APPROVED': '🟢', 'FLAGGED': '🟡', 'BLOCKED': '🔴'}.get(decision, '⚪')

        top_reason = (
            scenario or
            (rule_result['risk_factors'][0] if rule_result['risk_factors'] else None) or
            (ml_result.get('top_explanations', [{}])[0].get('feature', '-') if ml_result else '-')
        )

        summary = stats.get_summary()
        print(
            f"[{item['timestamp'].strftime('%H:%M:%S')}] "
            f"{icon} {decision:<8} "
            f"₹{txn_ctx.amount:>9,.0f}  "
            f"score: {final_result['final_score']:.3f}  "
            f"rule: {final_result.get('rule_score', 0):.3f}  "
            f"ml: {ml_result.get('ml_score', 0):.3f}  "
            f"| {str(top_reason):<28} "
            f"fraud%: {summary['fraud_rate']}%"
        )

        txn_queue.task_done()


# ============================================================
# MAIN — Starts both threads and runs the pipeline
# ============================================================
def run_pipeline():
    print("=" * 70)
    print("🚀 FinShield Pipeline Starting — Day 5")
    print("=" * 70)

    # Load reference data
    print("\n📦 Loading reference data...")
    users, merchants = load_reference_data()
    if not users or not merchants:
        print("❌ No data found. Run seed_data_v2.py first.")
        return

    # Initialise components
    print("⚙️  Initialising components...")
    txn_queue   = queue.Queue(maxsize=50)
    # maxsize=50: if processor falls behind, generator slows down
    # This is called "backpressure" — prevents queue from growing forever

    velocity    = VelocityTracker()
    injector    = FraudScenarioInjector(injection_rate=0.08)
    rule_engine = RuleEngine()
    stats       = PipelineStats(window=100)

    print("🤖 Loading ML model...")
    try:
        scorer = FraudModelScorer()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    print("💾 Connecting to database...")
    db_writer = DatabaseWriter()

    # Stop event — set this to gracefully stop both threads
    stop_event = threading.Event()

    # Start threads
    gen_thread = threading.Thread(
        target=generator_thread,
        args=(txn_queue, users, merchants, velocity, injector, stop_event),
        daemon=True,
        name="Generator"
    )
    proc_thread = threading.Thread(
        target=processor_thread,
        args=(txn_queue, rule_engine, scorer, db_writer, stats, stop_event),
        daemon=True,
        name="Processor"
    )

    gen_thread.start()
    proc_thread.start()

    print("\n✅ Pipeline running. Ctrl+C to stop.\n")
    print(
        f"{'Time':<10} {'Decision':<10} {'Amount':>12}  "
        f"{'Score':>7}  {'Rule':>7}  {'ML':>7}  "
        f"{'Reason':<30} {'Fraud%'}"
    )
    print("-" * 100)

    try:
        while True:
            time.sleep(30)
            # Print summary every 30 seconds
            s = stats.get_summary()
            print(f"\n{'='*70}")
            print(f"  📊 Pipeline Summary")
            print(f"  Total processed:  {s['total_processed']}")
            print(f"  Fraud detected:   {s['total_fraud']} ({s['fraud_rate']}%)")
            print(f"  Blocked:          {s['total_blocked']}")
            print(f"  Throughput:       {s['txns_per_minute']} txns/min")
            print(f"  Avg score (last {stats.window}): {s['avg_score_recent']}")
            print(f"{'='*70}\n")

    except KeyboardInterrupt:
        print("\n\n⛔ Stopping pipeline...")
        stop_event.set()
        gen_thread.join(timeout=3)
        proc_thread.join(timeout=3)
        db_writer.close()

        s = stats.get_summary()
        print(f"\n📊 Final Summary:")
        print(f"  Total processed: {s['total_processed']}")
        print(f"  Fraud detected:  {s['total_fraud']} ({s['fraud_rate']}%)")
        print(f"  Blocked:         {s['total_blocked']}")
        print(f"  Throughput:      {s['txns_per_minute']} txns/min")
        print("\n✅ Pipeline stopped cleanly.")


if __name__ == "__main__":
    run_pipeline()
