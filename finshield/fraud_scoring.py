"""
fraud_scoring.py
================
Improved fraud injection logic for FinShield seed data.

This module is used in TWO places:
1. seed_data.py  — to label historical transactions (fraud injection)
2. rule_engine.py (Day 3) — to score live incoming transactions

By keeping scoring logic in one place, both your training data
and your live system use identical logic. This is called
"training-serving consistency" — a common source of bugs when violated.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional


# ============================================================
# DATA CLASSES
# Clean way to pass transaction + context around
# Without these you'd be passing 15 function arguments everywhere
# ============================================================

@dataclass
class UserContext:
    user_id: int
    account_age_days: int
    avg_txn_amount: float
    std_txn_amount: float       # standard deviation of their transactions
    total_txns: int
    risk_tier: str              # LOW, MEDIUM, HIGH
    known_device_types: list    # devices this user has used before


@dataclass
class MerchantContext:
    merchant_id: int
    fraud_rate: float
    is_blacklisted: bool
    registered_days_ago: int
    category: str


@dataclass
class TransactionContext:
    amount: float
    hour_of_day: int
    day_of_week: int
    device_type: str
    velocity_1min: int          # txns by this user in last 1 min
    velocity_5min: int          # txns by this user in last 5 mins
    velocity_1hr: int           # txns by this user in last 1 hr
    amount_sum_1hr: float       # total amount spent by user in last 1 hr
    is_first_txn_at_merchant: bool


# ============================================================
# CONCEPT 1 — Z-SCORE DEVIATION
# How far is this transaction from the user's normal behaviour?
# ============================================================

def compute_amount_zscore(amount: float, user: UserContext) -> float:
    """
    Z-score = (value - mean) / std_deviation

    Returns how many standard deviations above/below normal this amount is.
    We clamp it to [0, 5] — anything beyond 5 std deviations is treated
    the same (extremely anomalous).

    Why clamp at 5?
    Because z=100 and z=5 are both "extremely anomalous" — there's no
    practical difference in fraud risk between them. Clamping prevents
    a single outlier from dominating the score.
    """
    if user.std_txn_amount == 0:
        # New user with no history — can't compute std, assume moderate risk
        return 1.5

    z = (amount - user.avg_txn_amount) / user.std_txn_amount
    z = max(0, z)       # negative z means below average — not suspicious
    return min(z, 5.0)  # clamp at 5


def zscore_to_risk(z: float) -> float:
    """
    Convert z-score to a 0-1 risk signal using sigmoid-like scaling.

    Why not just use z directly?
    Because z is unbounded and we need a 0-1 probability-like value
    to combine with other signals.

    Mapping:
    z=0   → 0.00  (normal)
    z=1   → 0.18  (slightly above average)
    z=2   → 0.33  (moderately suspicious)
    z=3   → 0.50  (significantly suspicious)
    z=4   → 0.67  (highly suspicious)
    z=5   → 0.83  (extremely suspicious)
    """
    return 1 - (1 / (1 + z * 0.25))


# ============================================================
# CONCEPT 2 — VELOCITY RISK
# Multiple time windows, each capturing different attack patterns
# ============================================================

def compute_velocity_risk(txn: TransactionContext) -> float:
    """
    Combines multiple velocity windows into one risk signal.

    Each window catches a different fraud type:
    - 1min window: card testing (rapid small transactions to check if card works)
    - 5min window: automated fraud bots
    - 1hr window: account takeover (attacker drains account quickly)

    Why not just use the 1hr window?
    Because card testing (1min pattern) would be invisible in a 1hr window
    if only a few transactions happened. Each window has different sensitivity.
    """
    signals = []

    # 1-minute velocity: > 3 transactions = card testing
    if txn.velocity_1min >= 3:
        signals.append(min(txn.velocity_1min / 5.0, 1.0))   # normalise to 0-1
    elif txn.velocity_1min >= 2:
        signals.append(0.3)
    else:
        signals.append(0.0)

    # 5-minute velocity: > 5 transactions = bot/automated fraud
    if txn.velocity_5min >= 5:
        signals.append(min(txn.velocity_5min / 8.0, 1.0))
    elif txn.velocity_5min >= 3:
        signals.append(0.25)
    else:
        signals.append(0.0)

    # 1-hour velocity: > 10 transactions = account takeover pattern
    if txn.velocity_1hr >= 10:
        signals.append(min(txn.velocity_1hr / 15.0, 1.0))
    elif txn.velocity_1hr >= 6:
        signals.append(0.3)
    else:
        signals.append(0.0)

    # Amount sum in last hour: spending too much too fast
    # Threshold = 3x user's average transaction amount
    # This catches attackers who make large but not individually suspicious txns
    hourly_limit = max(txn.amount_sum_1hr / 3, 1)   # avoid div by zero
    if txn.amount_sum_1hr > hourly_limit * 3:
        signals.append(0.6)
    elif txn.amount_sum_1hr > hourly_limit * 1.5:
        signals.append(0.3)
    else:
        signals.append(0.0)

    # Return max of all velocity signals (not average)
    # Why max? Because any ONE velocity pattern firing is enough to be suspicious.
    # Averaging would dilute a strong 1min signal with three 0.0 signals.
    return max(signals)


# ============================================================
# CONCEPT 3 — CONTEXTUAL RISK SIGNALS
# Individual features that contribute to risk
# ============================================================

def compute_contextual_signals(txn: TransactionContext,
                                user: UserContext,
                                merchant: MerchantContext) -> dict:
    """
    Returns individual risk signals as a dictionary.
    Each is a float between 0 and 1.

    Keeping them separate (not combined yet) lets you:
    1. Log each signal for explainability
    2. Use them as individual ML features
    3. Apply interaction effects in the next step
    """
    signals = {}

    # Time-based risk
    # Fraud peaks between 1am-5am (users asleep, fraud goes unnoticed)
    if txn.hour_of_day in range(1, 5):
        signals['odd_hour'] = 0.4
    elif txn.hour_of_day in range(22, 24) or txn.hour_of_day == 0:
        signals['late_night'] = 0.2
    else:
        signals['odd_hour'] = 0.0

    # Device risk
    # New device type = possible account takeover
    if txn.device_type == 'unknown':
        signals['unknown_device'] = 0.5
    elif txn.device_type not in user.known_device_types:
        signals['new_device'] = 0.3
    else:
        signals['new_device'] = 0.0

    # Merchant risk
    signals['merchant_fraud_rate'] = min(merchant.fraud_rate * 3, 1.0)
    # Why multiply by 3? A 10% merchant fraud rate is already very high.
    # Without scaling, a 10% rate would only contribute 0.1 to the signal.
    # Multiplying makes the signal more sensitive to merchant risk.

    if merchant.is_blacklisted:
        signals['blacklisted_merchant'] = 1.0   # maximum risk

    # New merchant for this user
    if txn.is_first_txn_at_merchant:
        signals['new_merchant'] = 0.15
    else:
        signals['new_merchant'] = 0.0

    # New user risk (account age)
    if user.account_age_days < 7:
        signals['new_account'] = 0.5
    elif user.account_age_days < 30:
        signals['new_account'] = 0.25
    else:
        signals['new_account'] = 0.0

    # User risk tier
    tier_risk = {'LOW': 0.0, 'MEDIUM': 0.1, 'HIGH': 0.4}
    signals['user_tier'] = tier_risk.get(user.risk_tier, 0.1)

    return signals


# ============================================================
# CONCEPT 4 — MULTIPLICATIVE COMBINATION
# Combines all signals without additive accumulation problem
# ============================================================

def combine_signals_multiplicative(signals: list) -> float:
    """
    Combines multiple 0-1 risk signals using the complement product formula.

    Formula: combined = 1 - product(1 - signal_i)

    Why this works:
    - If all signals are 0 → combined = 1 - 1 = 0 (no risk)
    - If any signal is 1 → combined = 1 (certain fraud)
    - Multiple medium signals combine naturally without over-triggering

    Example:
    signals = [0.3, 0.2, 0.4]
    combined = 1 - (0.7 * 0.8 * 0.6) = 1 - 0.336 = 0.664

    Additive would give: 0.3 + 0.2 + 0.4 = 0.9 (too high)
    """
    if not signals:
        return 0.0

    complement_product = 1.0
    for s in signals:
        complement_product *= (1 - s)

    return 1 - complement_product


# ============================================================
# CONCEPT 5 — INTERACTION EFFECTS
# Certain combinations of signals are exponentially more suspicious
# ============================================================

def apply_interaction_effects(base_score: float,
                               signals: dict,
                               txn: TransactionContext,
                               user: UserContext) -> float:
    """
    Boosts score when dangerous signal combinations appear together.

    Think of this like poker hands:
    - Three of a kind is not just "3x a single card"
    - It's a qualitatively different (and rarer) situation

    Same here — certain combinations are qualitatively more dangerous.
    """
    boosted_score = base_score
    boost_log = []

    # Interaction 1: New account + high amount + new merchant
    # Classic "bust-out fraud" — open account, spend heavily, disappear
    if (user.account_age_days < 30 and
        signals.get('new_merchant', 0) > 0 and
        txn.amount > user.avg_txn_amount * 3):
        boost = 0.20
        boosted_score = min(1.0, boosted_score + boost)
        boost_log.append(f"INTERACTION: new_account+high_amount+new_merchant (+{boost})")

    # Interaction 2: Odd hour + new device + velocity
    # Likely account takeover — attacker operates at night, uses own device, moves fast
    if (signals.get('odd_hour', 0) > 0.3 and
        signals.get('new_device', 0) > 0.2 and
        txn.velocity_1hr >= 3):
        boost = 0.25
        boosted_score = min(1.0, boosted_score + boost)
        boost_log.append(f"INTERACTION: odd_hour+new_device+velocity (+{boost})")

    # Interaction 3: Blacklisted merchant + high amount
    # Obvious: if merchant is known bad AND amount is large = definite block
    if signals.get('blacklisted_merchant', 0) == 1.0 and txn.amount > 10000:
        boosted_score = 1.0
        boost_log.append("INTERACTION: blacklisted_merchant+high_amount (→ 1.0)")

    return boosted_score, boost_log


# ============================================================
# CONCEPT 6 — DYNAMIC THRESHOLD
# One threshold for all users is too blunt
# ============================================================

def get_dynamic_threshold(user: UserContext,
                           txn: TransactionContext) -> float:
    """
    Returns the fraud threshold for THIS specific transaction context.

    Base threshold = 0.55 (slightly more than half)
    Adjustments make it stricter or more relaxed based on context.
    """
    threshold = 0.55

    # Stricter for risky users
    if user.risk_tier == 'HIGH':    threshold -= 0.15
    if user.risk_tier == 'LOW':     threshold += 0.10

    # Stricter for new accounts (can't trust history)
    if user.account_age_days < 7:   threshold -= 0.15
    elif user.account_age_days < 30: threshold -= 0.08

    # Stricter for large amounts (more damage if fraud)
    if txn.amount > 100000:         threshold -= 0.12
    elif txn.amount > 50000:        threshold -= 0.06

    # Stricter at odd hours
    if txn.hour_of_day in range(1, 5): threshold -= 0.05

    # Clamp — never go below 0.25 (too aggressive) or above 0.75 (too relaxed)
    return max(0.25, min(0.75, threshold))


# ============================================================
# MASTER SCORING FUNCTION
# Calls everything above in the right order
# ============================================================

def compute_fraud_score(txn: TransactionContext,
                         user: UserContext,
                         merchant: MerchantContext,
                         add_noise: bool = True) -> dict:
    """
    Main function — takes transaction + context, returns fraud assessment.

    Returns a dict (not just a score) so you can log every component.
    This is called "explainability" — you can tell exactly WHY a transaction
    was flagged, which is required for compliance in real fintech systems.

    add_noise=True when used for seed data injection (simulates real-world
    unpredictability). Set to False in production scoring.
    """

    # ---- Step 1: Hard block check ----
    # These skip scoring entirely
    if merchant.is_blacklisted and txn.amount > 50000:
        return {
            'score': 1.0,
            'decision': 'BLOCKED',
            'reason': 'HARD_BLOCK: blacklisted_merchant_high_amount',
            'threshold': 0.0,
            'signals': {},
            'interactions': []
        }

    if txn.velocity_1min >= 5:
        return {
            'score': 1.0,
            'decision': 'BLOCKED',
            'reason': 'HARD_BLOCK: velocity_card_testing',
            'threshold': 0.0,
            'signals': {},
            'interactions': []
        }

    # ---- Step 2: Compute individual signals ----
    z = compute_amount_zscore(txn.amount, user)
    amount_risk = zscore_to_risk(z)

    velocity_risk = compute_velocity_risk(txn)

    contextual = compute_contextual_signals(txn, user, merchant)

    # ---- Step 3: Combine all signals multiplicatively ----
    all_signals = [amount_risk, velocity_risk] + list(contextual.values())
    base_score = combine_signals_multiplicative(all_signals)

    # ---- Step 4: Apply interaction boosts ----
    final_score, interactions = apply_interaction_effects(
        base_score, contextual, txn, user
    )

    # ---- Step 5: Add noise (only for seed data) ----
    # Real fraud data has noise — not every high-risk transaction is fraud,
    # not every low-risk transaction is clean. Adding noise makes your
    # training data more realistic and prevents overfitting.
    if add_noise:
        noise = random.gauss(0, 0.05)   # small gaussian noise
        final_score = max(0.0, min(1.0, final_score + noise))

    # ---- Step 6: Apply dynamic threshold ----
    threshold = get_dynamic_threshold(user, txn)

    if final_score >= threshold + 0.20:
        decision = 'BLOCKED'
    elif final_score >= threshold:
        decision = 'FLAGGED'
    else:
        decision = 'APPROVED'

    # Determine primary reason for flagging
    reason = None
    if decision != 'APPROVED':
        top_signal = max(contextual, key=contextual.get) if contextual else 'amount_deviation'
        if velocity_risk > 0.5:
            reason = 'VELOCITY_BREACH'
        elif amount_risk > 0.5:
            reason = 'AMOUNT_DEVIATION'
        elif contextual.get('blacklisted_merchant', 0) > 0:
            reason = 'BLACKLISTED_MERCHANT'
        elif contextual.get('odd_hour', 0) > 0.3:
            reason = 'ODD_HOUR_TRANSACTION'
        elif contextual.get('new_device', 0) > 0.2:
            reason = 'NEW_DEVICE_DETECTED'
        else:
            reason = 'ML_SCORE_THRESHOLD'

    return {
        'score':        round(final_score, 4),
        'decision':     decision,
        'reason':       reason,
        'threshold':    round(threshold, 4),
        'z_score':      round(z, 4),
        'amount_risk':  round(amount_risk, 4),
        'velocity_risk': round(velocity_risk, 4),
        'signals':      {k: round(v, 4) for k, v in contextual.items()},
        'interactions': interactions
    }


# ============================================================
# QUICK TEST — Run this file directly to see it working
# python fraud_scoring.py
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FinShield Fraud Scoring — Test Cases")
    print("=" * 60)

    # Simulate a normal user
    normal_user = UserContext(
        user_id=1,
        account_age_days=365,
        avg_txn_amount=2000,
        std_txn_amount=800,
        total_txns=200,
        risk_tier='LOW',
        known_device_types=['mobile', 'web']
    )

    # Simulate a new suspicious user
    new_user = UserContext(
        user_id=2,
        account_age_days=3,
        avg_txn_amount=500,
        std_txn_amount=200,
        total_txns=2,
        risk_tier='MEDIUM',
        known_device_types=['mobile']
    )

    normal_merchant = MerchantContext(
        merchant_id=1,
        fraud_rate=0.01,
        is_blacklisted=False,
        registered_days_ago=500,
        category='Groceries'
    )

    risky_merchant = MerchantContext(
        merchant_id=2,
        fraud_rate=0.18,
        is_blacklisted=True,
        registered_days_ago=10,
        category='Electronics'
    )

    # Test Case 1: Completely normal transaction
    txn1 = TransactionContext(
        amount=1800,
        hour_of_day=14,
        day_of_week=1,
        device_type='mobile',
        velocity_1min=0,
        velocity_5min=1,
        velocity_1hr=2,
        amount_sum_1hr=3000,
        is_first_txn_at_merchant=False
    )

    # Test Case 2: Card testing attack
    txn2 = TransactionContext(
        amount=1,                   # tiny amount = card testing
        hour_of_day=3,              # 3am
        day_of_week=6,
        device_type='unknown',
        velocity_1min=6,            # 6 transactions in 1 minute!
        velocity_5min=8,
        velocity_1hr=10,
        amount_sum_1hr=10,
        is_first_txn_at_merchant=True
    )

    # Test Case 3: Bust-out fraud (new account, large amount, new merchant)
    txn3 = TransactionContext(
        amount=45000,               # 90x their average
        hour_of_day=2,
        day_of_week=5,
        device_type='web',          # haven't used web before
        velocity_1min=0,
        velocity_5min=1,
        velocity_1hr=1,
        amount_sum_1hr=45000,
        is_first_txn_at_merchant=True
    )

    tests = [
        ("Normal transaction, trusted user", txn1, normal_user, normal_merchant),
        ("Card testing attack",              txn2, new_user,    normal_merchant),
        ("Bust-out fraud",                   txn3, new_user,    risky_merchant),
    ]

    for name, txn, user, merchant in tests:
        result = compute_fraud_score(txn, user, merchant, add_noise=False)
        print(f"\n📌 {name}")
        print(f"   Score:     {result['score']} (threshold: {result['threshold']})")
        print(f"   Decision:  {result['decision']}")
        print(f"   Reason:    {result['reason']}")
        print(f"   Z-score:   {result.get('z_score', 'N/A (hard block)')} → amount_risk: {result.get('amount_risk', 'N/A')}")
        print(f"   Velocity:  {result.get('velocity_risk', 'N/A')}")
        if result.get('interactions'):
            print(f"   Boosts:    {result['interactions']}")
