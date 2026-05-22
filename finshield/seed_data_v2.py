"""
seed_data.py (v2 — Improved Fraud Injection)
=============================================
Uses fraud_scoring.py instead of hardcoded additive weights.

Key improvement: fraud labels are now statistically grounded,
personalised per user, and use interaction effects.
This means your ML model will learn real patterns — not your own rules.
"""

import sys
import os
from dotenv import load_dotenv

# Ensure fraud_scoring.py is found regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root (one level up from this file)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

import psycopg2
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker

from fraud_scoring import (
    UserContext, MerchantContext, TransactionContext,
    compute_fraud_score
)

fake = Faker('en_IN')
random.seed(42)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     5432,
    "database": os.getenv("DB_NAME", "finshield"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# SEED USERS
# Now includes std_txn_amount — needed for z-score computation
# ============================================================
def seed_users(conn, n=200):
    cursor = conn.cursor()
    print(f"Seeding {n} users...")

    users = []
    for _ in range(n):
        account_age = random.randint(1, 730)

        if account_age < 30:
            risk_tier = 'MEDIUM'
        elif random.random() < 0.05:
            risk_tier = 'HIGH'
        else:
            risk_tier = 'LOW'

        avg_amount = round(random.uniform(200, 15000), 2)
        # std is roughly 30-50% of average for realistic spending patterns
        std_amount = round(avg_amount * random.uniform(0.3, 0.5), 2)
        total_txns = random.randint(0, 500)

        users.append((
            fake.name(), fake.email(), fake.phone_number()[:15],
            account_age, avg_amount, std_amount, total_txns, risk_tier
        ))

    cursor.executemany("""
        INSERT INTO users 
        (name, email, phone, account_age_days, avg_txn_amount, std_txn_amount, total_txns, risk_tier)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, users)

    conn.commit()
    print("✅ Users seeded")
    return users


# ============================================================
# SEED MERCHANTS (same as before)
# ============================================================
MERCHANT_CATEGORIES = ['Electronics', 'Food & Dining', 'Travel', 'Fashion',
                        'Groceries', 'Gaming', 'Healthcare', 'Fuel']
INDIAN_CITIES = ['Bengaluru', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai',
                  'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Kochi']

def seed_merchants(conn, n=50):
    cursor = conn.cursor()
    print(f"Seeding {n} merchants...")

    merchants = []
    for _ in range(n):
        total_txns = random.randint(100, 10000)
        if random.random() < 0.1:
            fraud_count = random.randint(int(total_txns * 0.1), int(total_txns * 0.3))
        else:
            fraud_count = random.randint(0, int(total_txns * 0.02))

        fraud_rate = round(fraud_count / total_txns, 4)
        reg_date = fake.date_between(start_date='-3y', end_date='today')

        merchants.append((
            fake.company(), random.choice(MERCHANT_CATEGORIES),
            random.choice(INDIAN_CITIES), reg_date,
            total_txns, fraud_count, fraud_rate,
            fraud_rate > 0.15
        ))

    cursor.executemany("""
        INSERT INTO merchants 
        (name, category, city, registered_date, total_transactions, 
         fraud_count, fraud_rate, is_blacklisted)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, merchants)

    conn.commit()
    print("✅ Merchants seeded")


# ============================================================
# SEED TRANSACTIONS — Now uses compute_fraud_score()
# ============================================================
def seed_transactions(conn, n=2000):
    cursor = conn.cursor()
    print(f"Seeding {n} historical transactions...")

    # Fetch all users and merchants from DB (with their IDs)
    cursor.execute("""
        SELECT user_id, account_age_days, avg_txn_amount, 
               std_txn_amount, total_txns, risk_tier 
        FROM users
    """)
    db_users = cursor.fetchall()

    cursor.execute("""
        SELECT merchant_id, fraud_rate, is_blacklisted, 
               EXTRACT(DAY FROM NOW() - registered_date)::INT, category 
        FROM merchants
    """)
    db_merchants = cursor.fetchall()

    # Track transactions per user for velocity simulation
    user_txn_history = {u[0]: [] for u in db_users}

    transactions = []
    fraud_flags_data = []
    risk_scores_data = []

    now = datetime.now()

    for i in range(n):
        # Pick random user and merchant
        db_user = random.choice(db_users)
        db_merchant = random.choice(db_merchants)

        user_id = db_user[0]

        # Build context objects
        user_ctx = UserContext(
            user_id=user_id,
            account_age_days=db_user[1],
            avg_txn_amount=db_user[2],
            std_txn_amount=db_user[3],
            total_txns=db_user[4],
            risk_tier=db_user[5],
            known_device_types=random.sample(
                ['mobile', 'web', 'POS'],
                k=random.randint(1, 2)
            )
        )

        merchant_ctx = MerchantContext(
            merchant_id=db_merchant[0],
            fraud_rate=float(db_merchant[1]),
            is_blacklisted=db_merchant[2],
            registered_days_ago=db_merchant[3] or 30,
            category=db_merchant[4]
        )

        # Generate timestamp
        created_at = now - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        # Simulate amount
        if random.random() < 0.05:
            amount = round(random.uniform(
                user_ctx.avg_txn_amount * 3,
                user_ctx.avg_txn_amount * 10
            ), 2)
        else:
            amount = abs(round(
                random.gauss(user_ctx.avg_txn_amount, user_ctx.std_txn_amount),
                2
            ))
            amount = max(10, amount)

        device_type = random.choice(['mobile', 'web', 'POS', 'unknown'])
        hour_of_day = created_at.hour
        day_of_week = created_at.weekday()

        # Compute velocity from simulated history
        recent_1min = sum(
            1 for t in user_txn_history[user_id]
            if (created_at - t['ts']).seconds < 60
        )
        recent_5min = sum(
            1 for t in user_txn_history[user_id]
            if (created_at - t['ts']).seconds < 300
        )
        recent_1hr = sum(
            1 for t in user_txn_history[user_id]
            if (created_at - t['ts']).seconds < 3600
        )
        amount_1hr = sum(
            t['amount'] for t in user_txn_history[user_id]
            if (created_at - t['ts']).seconds < 3600
        )
        seen_merchants = {t['merchant_id'] for t in user_txn_history[user_id]}
        is_first_txn = db_merchant[0] not in seen_merchants

        txn_ctx = TransactionContext(
            amount=amount,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            device_type=device_type,
            velocity_1min=recent_1min,
            velocity_5min=recent_5min,
            velocity_1hr=recent_1hr,
            amount_sum_1hr=amount_1hr,
            is_first_txn_at_merchant=is_first_txn
        )

        # ---- THE IMPROVED SCORING ----
        result = compute_fraud_score(txn_ctx, user_ctx, merchant_ctx, add_noise=True)

        score = result['score']
        decision = result['decision']
        reason = result['reason']

        # Map decision to DB status
        status_map = {'APPROVED': 'APPROVED', 'FLAGGED': 'FLAGGED', 'BLOCKED': 'BLOCKED'}
        status = status_map[decision]

        txn_id = str(uuid.uuid4())

        transactions.append((
            txn_id, user_id, db_merchant[0], amount, 'INR',
            device_type, fake.ipv4(),
            hour_of_day, day_of_week, status, created_at
        ))

        if decision in ('FLAGGED', 'BLOCKED'):
            fraud_flags_data.append((
                txn_id, reason, round(score, 4),
                decision, 'SYSTEM', created_at
            ))

        risk_scores_data.append((
            txn_id, 'xgb_v1.0', round(score, 4), result['threshold'],
            'FRAUD' if decision != 'APPROVED' else 'LEGIT',
            created_at
        ))

        # Update simulated history
        user_txn_history[user_id].append({
            'ts': created_at,
            'amount': amount,
            'merchant_id': db_merchant[0]
        })

        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n} transactions processed...")

    # Batch inserts
    cursor.executemany("""
        INSERT INTO transactions 
        (transaction_id, user_id, merchant_id, amount, currency, device_type,
         ip_address, hour_of_day, day_of_week, status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, transactions)

    cursor.executemany("""
        INSERT INTO fraud_flags 
        (transaction_id, rule_triggered, ml_score, final_decision, reviewed_by, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, fraud_flags_data)

    cursor.executemany("""
        INSERT INTO risk_scores 
        (transaction_id, model_version, score, threshold_used, decision, scored_at)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, risk_scores_data)

    conn.commit()

    fraud_count = len(fraud_flags_data)
    print(f"✅ Transactions seeded — {fraud_count} fraud cases ({round(fraud_count/n*100,1)}%)")


def verify(conn):
    cursor = conn.cursor()
    print("\n📊 Database Summary:")

    for table in ['users', 'merchants', 'transactions', 'fraud_flags', 'risk_scores']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cursor.fetchone()[0]} rows")

    cursor.execute("SELECT status, COUNT(*) FROM transactions GROUP BY status ORDER BY COUNT(*) DESC")
    print("\n📊 Transaction Status Breakdown:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cursor.execute("SELECT rule_triggered, COUNT(*) FROM fraud_flags GROUP BY rule_triggered ORDER BY COUNT(*) DESC")
    print("\n🚨 Fraud Rules Triggered:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")


if __name__ == "__main__":
    print("🚀 FinShield v2 — Seeding Database\n")

    # First run the test to confirm scoring works
    print("Running scoring sanity check...")
    import subprocess
    subprocess.run(["python", "fraud_scoring.py"])
    print("\n" + "="*50 + "\n")

    conn = get_connection()
    seed_users(conn)
    seed_merchants(conn)
    seed_transactions(conn)
    verify(conn)
    conn.close()

    print("\n✅ Day 1 v2 Complete.")
    print("Your fraud labels are now statistically grounded.")
    print("Next: transaction_generator.py (Day 2)")
