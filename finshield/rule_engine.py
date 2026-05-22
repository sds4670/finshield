"""
rule_engine.py — Day 3
=======================
First line of defence in FinShield's fraud detection pipeline.

Sits between the transaction generator and the ML model.
Handles hard blocks instantly and generates rich context
for the ML scorer on soft flags.

Usage:
    from rule_engine import RuleEngine
    engine = RuleEngine()
    result = engine.evaluate(txn_ctx, user_ctx, merchant_ctx)
"""

import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Force UTF-8 output — prevents emoji crash on Windows cp1252 terminals
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from dataclasses import dataclass, field
from typing import Optional
from fraud_scoring import UserContext, MerchantContext, TransactionContext


# ============================================================
# RESULT OBJECTS
# Clean data structures — no messy dicts passed around
# ============================================================

@dataclass
class RuleFinding:
    """
    Records exactly what one rule found and why it fired.
    
    Why store evidence separately from the decision?
    Because your dashboard needs to show a human-readable explanation:
    "Blocked because: 6 transactions in 1 minute (threshold: 5)"
    You can't reconstruct that from just a score.
    """
    rule_name:    str
    fired:        bool
    severity:     str           # HARD_BLOCK, SOFT_FLAG, TRUST_BOOST
    contribution: float         # how much this adds/subtracts from score
    evidence:     dict          # raw values that triggered the rule
    message:      str           # human-readable explanation


@dataclass
class RuleEngineResult:
    """
    Final output of the rule engine for one transaction.
    Passed to the ML scorer if decision is PENDING.
    """
    decision:       str                     # BLOCKED, FLAGGED, APPROVED, PENDING
    score:          float                   # combined rule score 0-1
    hard_blocked:   bool                    # True if a hard block rule fired
    block_reason:   Optional[str]           # which hard block rule fired
    findings:       list = field(default_factory=list)   # all RuleFinding objects
    risk_factors:   list = field(default_factory=list)   # list of fired rule names
    trust_factors:  list = field(default_factory=list)   # list of trust boost names
    send_to_ml:     bool = True             # should ML scorer see this?

    def add_finding(self, finding: RuleFinding):
        self.findings.append(finding)
        if finding.fired:
            if finding.severity == 'TRUST_BOOST':
                self.trust_factors.append(finding.rule_name)
            elif finding.severity in ('HARD_BLOCK', 'SOFT_FLAG'):
                self.risk_factors.append(finding.rule_name)

    def to_dict(self) -> dict:
        return {
            'decision':     self.decision,
            'score':        round(self.score, 4),
            'hard_blocked': self.hard_blocked,
            'block_reason': self.block_reason,
            'risk_factors': self.risk_factors,
            'trust_factors': self.trust_factors,
            'findings': [
                {
                    'rule':         f.rule_name,
                    'fired':        f.fired,
                    'severity':     f.severity,
                    'contribution': f.contribution,
                    'evidence':     f.evidence,
                    'message':      f.message
                }
                for f in self.findings if f.fired
            ]
        }


# ============================================================
# RULE ENGINE CLASS
# Each rule is a separate method — easy to add/remove/modify
# ============================================================

class RuleEngine:
    """
    Evaluates a transaction against a set of deterministic rules.
    
    Design principles:
    1. Each rule is isolated — one method, one responsibility
    2. Rules chain — hard blocks stop evaluation immediately
    3. Every decision is explained — evidence stored per rule
    4. Trust signals reduce score — not everything is about adding risk
    """

    def __init__(self):
        # Configurable thresholds — change these without touching rule logic
        self.VELOCITY_1MIN_HARD_BLOCK   = 5       # card testing threshold
        self.VELOCITY_5MIN_SOFT_FLAG    = 4
        self.VELOCITY_1HR_SOFT_FLAG     = 8
        self.AMOUNT_HARD_BLOCK          = 1000000  # ₹10 lakhs
        self.AMOUNT_LARGE_SOFT_FLAG     = 100000   # ₹1 lakh
        self.AMOUNT_MEDIUM_SOFT_FLAG    = 50000    # ₹50k
        self.MERCHANT_FRAUD_HARD_BLOCK  = 0.30     # 30% fraud rate → hard block
        self.MERCHANT_FRAUD_SOFT_FLAG   = 0.10     # 10% fraud rate → soft flag
        self.NEW_ACCOUNT_DAYS_HIGH      = 7
        self.NEW_ACCOUNT_DAYS_MEDIUM    = 30
        self.ODD_HOUR_RANGE             = list(range(1, 5))  # 1am-4am

    # ----------------------------------------------------------
    # MASTER EVALUATE — calls all rules in order
    # ----------------------------------------------------------
    def evaluate(self,
                 txn: TransactionContext,
                 user: UserContext,
                 merchant: MerchantContext) -> RuleEngineResult:
        """
        Runs all rules against this transaction.
        
        Order matters:
        1. Hard block rules first — if one fires, stop immediately
        2. Soft flag rules — build up risk score
        3. Trust boost rules — reduce score for trusted signals
        4. Final decision based on combined score
        """

        result = RuleEngineResult(
            decision='PENDING',
            score=0.0,
            hard_blocked=False,
            block_reason=None,
            send_to_ml=True
        )

        # ---- LAYER 1: Hard Block Rules ----
        # Run these first. If any fires, return immediately.
        hard_block_rules = [
            self._rule_velocity_card_testing,
            self._rule_amount_hard_limit,
            self._rule_blacklisted_merchant,
            self._rule_merchant_extreme_fraud_rate,
        ]

        for rule_fn in hard_block_rules:
            finding = rule_fn(txn, user, merchant)
            result.add_finding(finding)

            if finding.fired and finding.severity == 'HARD_BLOCK':
                result.hard_blocked = True
                result.block_reason = finding.rule_name
                result.decision = 'BLOCKED'
                result.score = 1.0
                result.send_to_ml = False
                # Stop here — no point evaluating more rules
                return result

        # ---- LAYER 2: Soft Flag Rules ----
        # These build up a risk score using multiplicative combination
        soft_flag_rules = [
            self._rule_velocity_moderate,
            self._rule_amount_large,
            self._rule_odd_hour,
            self._rule_new_device,
            self._rule_new_account,
            self._rule_merchant_elevated_fraud_rate,
            self._rule_new_merchant_high_amount,
            self._rule_user_risk_tier,
            self._rule_amount_z_score,
        ]

        soft_contributions = []
        for rule_fn in soft_flag_rules:
            finding = rule_fn(txn, user, merchant)
            result.add_finding(finding)
            if finding.fired:
                soft_contributions.append(finding.contribution)

        # ---- LAYER 3: Trust Boost Rules ----
        # These REDUCE the score — trusted users get benefit of the doubt
        trust_rules = [
            self._rule_trusted_user,
            self._rule_known_merchant,
            self._rule_daytime_normal_amount,
        ]

        trust_reductions = []
        for rule_fn in trust_rules:
            finding = rule_fn(txn, user, merchant)
            result.add_finding(finding)
            if finding.fired:
                trust_reductions.append(finding.contribution)  # negative values

        # ---- Combine scores multiplicatively ----
        # Risk signals combine: 1 - product(1 - signal)
        if soft_contributions:
            complement = 1.0
            for c in soft_contributions:
                complement *= (1 - c)
            base_score = 1 - complement
        else:
            base_score = 0.0

        # Trust reductions subtract from base score
        total_trust = sum(trust_reductions)   # trust contributions are negative
        final_score = max(0.0, min(1.0, base_score + total_trust))

        result.score = round(final_score, 4)

        # ---- Final decision ----
        if final_score >= 0.75:
            result.decision = 'BLOCKED'
            result.send_to_ml = False   # high confidence, no need for ML
        elif final_score >= 0.45:
            result.decision = 'FLAGGED'
            result.send_to_ml = True    # uncertain — let ML decide
        else:
            result.decision = 'APPROVED'
            result.send_to_ml = final_score > 0.25  # borderline gets ML check

        return result


    # ============================================================
    # HARD BLOCK RULES
    # ============================================================

    def _rule_velocity_card_testing(self, txn, user, merchant) -> RuleFinding:
        """
        Card Testing Attack:
        Attackers verify stolen cards by making many tiny transactions
        rapidly. Pattern: 5+ transactions in under 1 minute.
        
        Why 5 as threshold?
        A real user rarely makes more than 2-3 purchases per minute.
        At 5+, the probability of legitimate use drops below 1%.
        """
        fired = txn.velocity_1min >= self.VELOCITY_1MIN_HARD_BLOCK
        return RuleFinding(
            rule_name='VELOCITY_CARD_TESTING',
            fired=fired,
            severity='HARD_BLOCK',
            contribution=1.0 if fired else 0.0,
            evidence={
                'velocity_1min': txn.velocity_1min,
                'threshold': self.VELOCITY_1MIN_HARD_BLOCK,
                'amount': txn.amount
            },
            message=(
                f"Card testing detected: {txn.velocity_1min} transactions "
                f"in 1 minute (threshold: {self.VELOCITY_1MIN_HARD_BLOCK})"
                if fired else "Velocity normal"
            )
        )

    def _rule_amount_hard_limit(self, txn, user, merchant) -> RuleFinding:
        """
        Absolute Amount Limit:
        Transactions above ₹10 lakhs are always blocked and escalated
        to human review. No ML model should auto-approve these.
        
        Why a hard limit?
        Regulatory requirement in India — RBI guidelines require
        additional verification for high-value transactions.
        """
        fired = txn.amount >= self.AMOUNT_HARD_BLOCK
        return RuleFinding(
            rule_name='AMOUNT_HARD_LIMIT',
            fired=fired,
            severity='HARD_BLOCK',
            contribution=1.0 if fired else 0.0,
            evidence={
                'amount': txn.amount,
                'threshold': self.AMOUNT_HARD_BLOCK
            },
            message=(
                f"Amount ₹{txn.amount:,.0f} exceeds hard limit ₹{self.AMOUNT_HARD_BLOCK:,.0f}"
                if fired else "Amount within hard limit"
            )
        )

    def _rule_blacklisted_merchant(self, txn, user, merchant) -> RuleFinding:
        """
        Blacklisted Merchant:
        Merchant has been confirmed fraudulent by the fraud team.
        No transaction should go through — regardless of amount or user.
        """
        fired = merchant.is_blacklisted
        return RuleFinding(
            rule_name='BLACKLISTED_MERCHANT',
            fired=fired,
            severity='HARD_BLOCK',
            contribution=1.0 if fired else 0.0,
            evidence={
                'merchant_id': merchant.merchant_id,
                'is_blacklisted': merchant.is_blacklisted
            },
            message=(
                f"Merchant {merchant.merchant_id} is blacklisted"
                if fired else "Merchant not blacklisted"
            )
        )

    def _rule_merchant_extreme_fraud_rate(self, txn, user, merchant) -> RuleFinding:
        """
        Extreme Merchant Fraud Rate:
        Merchant has > 30% historical fraud rate — almost certainly
        a compromised or fake merchant. Hard block all transactions.
        
        Why 30%?
        Even the worst legitimate merchants rarely exceed 5-8%.
        At 30%, the merchant is almost certainly complicit in fraud.
        """
        fired = merchant.fraud_rate >= self.MERCHANT_FRAUD_HARD_BLOCK
        return RuleFinding(
            rule_name='MERCHANT_EXTREME_FRAUD_RATE',
            fired=fired,
            severity='HARD_BLOCK',
            contribution=1.0 if fired else 0.0,
            evidence={
                'merchant_fraud_rate': merchant.fraud_rate,
                'threshold': self.MERCHANT_FRAUD_HARD_BLOCK
            },
            message=(
                f"Merchant fraud rate {merchant.fraud_rate:.1%} exceeds hard block threshold"
                if fired else "Merchant fraud rate acceptable"
            )
        )


    # ============================================================
    # SOFT FLAG RULES
    # ============================================================

    def _rule_velocity_moderate(self, txn, user, merchant) -> RuleFinding:
        """
        Moderate Velocity:
        Not enough for hard block but suspicious.
        3-4 transactions in 5 minutes or 6-9 in 1 hour.
        """
        fired_5min = txn.velocity_5min >= self.VELOCITY_5MIN_SOFT_FLAG
        fired_1hr  = txn.velocity_1hr  >= self.VELOCITY_1HR_SOFT_FLAG
        fired = fired_5min or fired_1hr

        # Higher contribution for faster velocity
        contribution = 0.0
        if fired_5min: contribution = max(contribution, 0.4)
        if fired_1hr:  contribution = max(contribution, 0.3)

        return RuleFinding(
            rule_name='VELOCITY_MODERATE',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=contribution,
            evidence={
                'velocity_5min': txn.velocity_5min,
                'velocity_1hr':  txn.velocity_1hr,
                'threshold_5min': self.VELOCITY_5MIN_SOFT_FLAG,
                'threshold_1hr':  self.VELOCITY_1HR_SOFT_FLAG
            },
            message=(
                f"Elevated velocity: {txn.velocity_5min} txns/5min, "
                f"{txn.velocity_1hr} txns/hr"
                if fired else "Velocity normal"
            )
        )

    def _rule_amount_large(self, txn, user, merchant) -> RuleFinding:
        """
        Large Amount:
        Not hard-blocked but warrants ML review.
        Contribution scales with how large the amount is.
        """
        if txn.amount >= self.AMOUNT_LARGE_SOFT_FLAG:
            contribution = 0.5
            msg = f"Very large amount ₹{txn.amount:,.0f}"
        elif txn.amount >= self.AMOUNT_MEDIUM_SOFT_FLAG:
            contribution = 0.3
            msg = f"Large amount ₹{txn.amount:,.0f}"
        else:
            contribution = 0.0
            msg = "Amount within normal range"

        fired = contribution > 0
        return RuleFinding(
            rule_name='AMOUNT_LARGE',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=contribution,
            evidence={
                'amount': txn.amount,
                'threshold_high':   self.AMOUNT_LARGE_SOFT_FLAG,
                'threshold_medium': self.AMOUNT_MEDIUM_SOFT_FLAG
            },
            message=msg
        )

    def _rule_odd_hour(self, txn, user, merchant) -> RuleFinding:
        """
        Odd Hour Transaction:
        1am-4am transactions are statistically more likely to be fraud.
        Not conclusive alone — combined with other signals it matters.
        """
        fired = txn.hour_of_day in self.ODD_HOUR_RANGE
        return RuleFinding(
            rule_name='ODD_HOUR',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=0.25 if fired else 0.0,
            evidence={
                'hour_of_day': txn.hour_of_day,
                'odd_hour_range': f"{self.ODD_HOUR_RANGE[0]}am-{self.ODD_HOUR_RANGE[-1]}am"
            },
            message=(
                f"Transaction at {txn.hour_of_day}:00 (odd hour window)"
                if fired else f"Transaction at normal hour ({txn.hour_of_day}:00)"
            )
        )

    def _rule_new_device(self, txn, user, merchant) -> RuleFinding:
        """
        New or Unknown Device:
        Unknown device type = possible account takeover.
        Device not in user's history = possible compromised account.
        """
        if txn.device_type == 'unknown':
            fired, contribution = True, 0.45
            msg = "Unknown device type detected"
        elif txn.device_type not in user.known_device_types:
            fired, contribution = True, 0.25
            msg = f"New device type '{txn.device_type}' not in user history"
        else:
            fired, contribution = False, 0.0
            msg = "Known device type"

        return RuleFinding(
            rule_name='NEW_DEVICE',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=contribution,
            evidence={
                'device_type': txn.device_type,
                'known_devices': user.known_device_types
            },
            message=msg
        )

    def _rule_new_account(self, txn, user, merchant) -> RuleFinding:
        """
        New Account Risk:
        New accounts have no trust history — higher baseline risk.
        Very new accounts (< 7 days) are highest risk.
        """
        if user.account_age_days < self.NEW_ACCOUNT_DAYS_HIGH:
            fired, contribution = True, 0.45
            msg = f"Very new account ({user.account_age_days} days old)"
        elif user.account_age_days < self.NEW_ACCOUNT_DAYS_MEDIUM:
            fired, contribution = True, 0.20
            msg = f"Relatively new account ({user.account_age_days} days old)"
        else:
            fired, contribution = False, 0.0
            msg = f"Established account ({user.account_age_days} days old)"

        return RuleFinding(
            rule_name='NEW_ACCOUNT',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=contribution,
            evidence={
                'account_age_days': user.account_age_days,
                'threshold_high':   self.NEW_ACCOUNT_DAYS_HIGH,
                'threshold_medium': self.NEW_ACCOUNT_DAYS_MEDIUM
            },
            message=msg
        )

    def _rule_merchant_elevated_fraud_rate(self, txn, user, merchant) -> RuleFinding:
        """
        Elevated Merchant Fraud Rate:
        Merchant has higher-than-normal fraud rate but not enough
        for a hard block. Warrants extra scrutiny.
        """
        fired = (self.MERCHANT_FRAUD_SOFT_FLAG
                 <= merchant.fraud_rate
                 < self.MERCHANT_FRAUD_HARD_BLOCK)

        contribution = min(merchant.fraud_rate * 2, 0.5) if fired else 0.0

        return RuleFinding(
            rule_name='MERCHANT_ELEVATED_FRAUD_RATE',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=contribution,
            evidence={
                'merchant_fraud_rate': merchant.fraud_rate,
                'soft_flag_threshold': self.MERCHANT_FRAUD_SOFT_FLAG,
                'hard_block_threshold': self.MERCHANT_FRAUD_HARD_BLOCK
            },
            message=(
                f"Merchant fraud rate {merchant.fraud_rate:.1%} is elevated"
                if fired else "Merchant fraud rate normal"
            )
        )

    def _rule_new_merchant_high_amount(self, txn, user, merchant) -> RuleFinding:
        """
        Interaction Rule — New Merchant + High Amount:
        First transaction at a merchant is normal.
        First transaction at a merchant with a large amount is not.
        
        Why an interaction rule?
        Neither signal alone justifies flagging. Together they do.
        This is a rule-level version of the interaction effects
        from fraud_scoring.py.
        """
        fired = (txn.is_first_txn_at_merchant and
                 txn.amount > user.avg_txn_amount * 2.5)

        contribution = 0.35 if fired else 0.0

        return RuleFinding(
            rule_name='NEW_MERCHANT_HIGH_AMOUNT',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=contribution,
            evidence={
                'is_first_txn_at_merchant': txn.is_first_txn_at_merchant,
                'amount': txn.amount,
                'user_avg_amount': user.avg_txn_amount,
                'ratio': round(txn.amount / max(user.avg_txn_amount, 1), 2)
            },
            message=(
                f"First transaction at merchant, amount ₹{txn.amount:,.0f} "
                f"is {txn.amount/max(user.avg_txn_amount,1):.1f}x user average"
                if fired else "New merchant or amount within normal range"
            )
        )

    def _rule_user_risk_tier(self, txn, user, merchant) -> RuleFinding:
        """
        User Risk Tier:
        Pre-computed trust level from user's full history.
        HIGH tier users get stricter treatment on all transactions.
        """
        tier_contributions = {'HIGH': 0.35, 'MEDIUM': 0.05, 'LOW': 0.0}
        contribution = tier_contributions.get(user.risk_tier, 0.05)
        fired = contribution > 0

        return RuleFinding(
            rule_name='USER_RISK_TIER',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=contribution,
            evidence={'risk_tier': user.risk_tier},
            message=f"User risk tier: {user.risk_tier}"
        )

    def _rule_amount_z_score(self, txn, user, merchant) -> RuleFinding:
        """
        Amount Z-Score:
        How many standard deviations above this user's average is this amount?
        Personalised — ₹50,000 might be normal for one user, extreme for another.
        """
        if user.std_txn_amount <= 0:
            return RuleFinding(
                rule_name='AMOUNT_Z_SCORE',
                fired=False,
                severity='SOFT_FLAG',
                contribution=0.0,
                evidence={'reason': 'insufficient_history'},
                message="Cannot compute z-score — insufficient transaction history"
            )

        z = (txn.amount - user.avg_txn_amount) / user.std_txn_amount
        z = max(0, z)

        if z >= 4.0:
            fired, contribution = True, 0.60
            msg = f"Amount is {z:.1f} std deviations above user average (extreme)"
        elif z >= 2.5:
            fired, contribution = True, 0.40
            msg = f"Amount is {z:.1f} std deviations above user average (high)"
        elif z >= 1.5:
            fired, contribution = True, 0.20
            msg = f"Amount is {z:.1f} std deviations above user average (moderate)"
        else:
            fired, contribution = False, 0.0
            msg = f"Amount within normal range (z={z:.1f})"

        return RuleFinding(
            rule_name='AMOUNT_Z_SCORE',
            fired=fired,
            severity='SOFT_FLAG',
            contribution=contribution,
            evidence={
                'amount':          txn.amount,
                'user_avg':        user.avg_txn_amount,
                'user_std':        user.std_txn_amount,
                'z_score':         round(z, 2)
            },
            message=msg
        )


    # ============================================================
    # TRUST BOOST RULES — reduce score for trusted signals
    # ============================================================

    def _rule_trusted_user(self, txn, user, merchant) -> RuleFinding:
        """
        Trusted User:
        Long-established account + LOW risk tier + many clean transactions.
        Reduce score — these users deserve benefit of the doubt.
        """
        is_trusted = (
            user.account_age_days > 365 and
            user.risk_tier == 'LOW' and
            user.total_txns > 50
        )
        contribution = -0.20 if is_trusted else 0.0   # negative = reduces score

        return RuleFinding(
            rule_name='TRUSTED_USER',
            fired=is_trusted,
            severity='TRUST_BOOST',
            contribution=contribution,
            evidence={
                'account_age_days': user.account_age_days,
                'risk_tier':        user.risk_tier,
                'total_txns':       user.total_txns
            },
            message=(
                "Trusted user — score reduced"
                if is_trusted else "User does not meet trust criteria"
            )
        )

    def _rule_known_merchant(self, txn, user, merchant) -> RuleFinding:
        """
        Known Merchant:
        User has transacted at this merchant before without issues.
        Familiarity reduces risk — this is expected behaviour.
        """
        is_known = (
            not txn.is_first_txn_at_merchant and
            merchant.fraud_rate < 0.05
        )
        contribution = -0.10 if is_known else 0.0

        return RuleFinding(
            rule_name='KNOWN_MERCHANT',
            fired=is_known,
            severity='TRUST_BOOST',
            contribution=contribution,
            evidence={
                'is_first_txn': txn.is_first_txn_at_merchant,
                'merchant_fraud_rate': merchant.fraud_rate
            },
            message=(
                "Known low-risk merchant — score reduced"
                if is_known else "Merchant not in user's history or has elevated risk"
            )
        )

    def _rule_daytime_normal_amount(self, txn, user, merchant) -> RuleFinding:
        """
        Daytime Normal Amount:
        Transaction during business hours + amount close to user's average.
        Most legitimate transactions look exactly like this.
        """
        is_daytime = txn.hour_of_day in range(9, 21)
        is_normal_amount = abs(txn.amount - user.avg_txn_amount) < user.std_txn_amount

        is_trusted = is_daytime and is_normal_amount
        contribution = -0.10 if is_trusted else 0.0

        return RuleFinding(
            rule_name='DAYTIME_NORMAL_AMOUNT',
            fired=is_trusted,
            severity='TRUST_BOOST',
            contribution=contribution,
            evidence={
                'hour_of_day': txn.hour_of_day,
                'amount':      txn.amount,
                'user_avg':    user.avg_txn_amount,
                'user_std':    user.std_txn_amount
            },
            message=(
                "Daytime transaction within normal amount range — score reduced"
                if is_trusted else "Not a typical daytime normal transaction"
            )
        )


# ============================================================
# QUICK TEST — python rule_engine.py
# ============================================================
if __name__ == "__main__":
    engine = RuleEngine()

    # Shared user contexts
    trusted_user = UserContext(
        user_id=1, account_age_days=500, avg_txn_amount=2000,
        std_txn_amount=600, total_txns=150, risk_tier='LOW',
        known_device_types=['mobile', 'web']
    )
    new_user = UserContext(
        user_id=2, account_age_days=4, avg_txn_amount=800,
        std_txn_amount=300, total_txns=2, risk_tier='MEDIUM',
        known_device_types=['mobile']
    )
    safe_merchant = MerchantContext(
        merchant_id=1, fraud_rate=0.01,
        is_blacklisted=False, registered_days_ago=400, category='Groceries'
    )
    risky_merchant = MerchantContext(
        merchant_id=2, fraud_rate=0.22,
        is_blacklisted=True, registered_days_ago=5, category='Electronics'
    )

    test_cases = [
        (
            "✅ Normal daytime transaction",
            TransactionContext(
                amount=1800, hour_of_day=14, day_of_week=1,
                device_type='mobile', velocity_1min=0, velocity_5min=1,
                velocity_1hr=2, amount_sum_1hr=3000,
                is_first_txn_at_merchant=False
            ),
            trusted_user, safe_merchant
        ),
        (
            "🔴 Card testing attack",
            TransactionContext(
                amount=25, hour_of_day=3, day_of_week=6,
                device_type='unknown', velocity_1min=7, velocity_5min=9,
                velocity_1hr=12, amount_sum_1hr=200,
                is_first_txn_at_merchant=True
            ),
            new_user, safe_merchant
        ),
        (
            "🔴 Blacklisted merchant + high amount",
            TransactionContext(
                amount=75000, hour_of_day=22, day_of_week=5,
                device_type='web', velocity_1min=0, velocity_5min=1,
                velocity_1hr=1, amount_sum_1hr=75000,
                is_first_txn_at_merchant=True
            ),
            new_user, risky_merchant
        ),
        (
            "🟡 Borderline — new user, large amount, odd hour",
            TransactionContext(
                amount=45000, hour_of_day=2, day_of_week=3,
                device_type='web', velocity_1min=0, velocity_5min=1,
                velocity_1hr=1, amount_sum_1hr=45000,
                is_first_txn_at_merchant=True
            ),
            new_user, safe_merchant
        ),
    ]

    print("=" * 70)
    print("FinShield Rule Engine — Test Cases")
    print("=" * 70)

    for name, txn, user, merchant in test_cases:
        result = engine.evaluate(txn, user, merchant)
        d = result.to_dict()

        print(f"\n{name}")
        print(f"  Decision:      {d['decision']}")
        print(f"  Score:         {d['score']}")
        print(f"  Hard blocked:  {d['hard_blocked']}")
        print(f"  Risk factors:  {d['risk_factors']}")
        print(f"  Trust factors: {d['trust_factors']}")
        print(f"  Send to ML:    {result.send_to_ml}")
        if d['findings']:
            print(f"  Findings:")
            for f in d['findings']:
                print(f"    [{f['severity']:<12}] {f['rule']:<35} → {f['message']}")
