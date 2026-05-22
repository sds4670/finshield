"""
ml_model.py — Day 4
=====================
XGBoost fraud detection model for FinShield.

Two responsibilities:
1. FraudModelTrainer  — trains on historical DB data, saves model
2. FraudModelScorer   — loads saved model, scores live transactions

Run training:
    python ml_model.py

This generates:
    models/fraud_model_xgb_v1.pkl
    models/feature_names.pkl
    models/model_metrics.json
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()
import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import shap
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# Force UTF-8 output to prevent emoji crash on Windows terminals
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     5432,
    "database": os.getenv("DB_NAME", "finshield"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model_xgb_v1.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
EXPLAINER_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")


# ============================================================
# FEATURE ENGINEERING
# This is the most important part of the entire ML pipeline.
# The model is only as good as the features you give it.
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transaction + context data into ML features.

    Input:  raw dataframe from DB join
    Output: feature matrix ready for XGBoost

    Why engineer features instead of using raw columns?
    Raw columns like user_id and merchant_id are IDs — they have no
    meaning to the model. Engineered features like amount_z_score
    and merchant_fraud_rate encode the actual signal.
    """

    features = pd.DataFrame()
    
    # Cast Decimal types from DB to float for numpy operations
    for col in ['amount', 'avg_txn_amount', 'std_txn_amount', 'fraud_rate']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # ---- Amount Features ----
    # Raw amount alone is weak — ₹5000 means different things for different users
    features['amount'] = df['amount']
    features['amount_log'] = np.log1p(df['amount'])
    # log transform compresses large values — ₹1000 vs ₹10000 is less extreme
    # after log than before. Prevents large amounts from dominating.

    features['amount_z_score'] = (
        (df['amount'] - df['avg_txn_amount']) /
        df['std_txn_amount'].replace(0, 1)   # avoid division by zero
    ).clip(lower=0, upper=10)
    # Z-score personalises the amount signal per user
    # Clipped at 10 — beyond 10 std deviations, additional distance doesn't matter

    features['amount_to_avg_ratio'] = (
        df['amount'] / df['avg_txn_amount'].replace(0, 1)
    ).clip(upper=20)
    # How many times their average is this transaction?
    # Clipped at 20 to prevent extreme outliers

    # ---- Time Features ----
    features['hour_of_day'] = df['hour_of_day']
    features['day_of_week'] = df['day_of_week']
    features['is_odd_hour'] = df['hour_of_day'].apply(
        lambda h: 1 if h in range(1, 5) else 0
    )
    features['is_weekend'] = df['day_of_week'].apply(
        lambda d: 1 if d >= 5 else 0
    )
    # Why hour AND is_odd_hour as separate features?
    # XGBoost can use hour_of_day to learn fine-grained time patterns
    # while is_odd_hour gives it a direct binary signal for the known
    # high-fraud window. Having both is more powerful than either alone.

    # ---- User Features ----
    features['account_age_days'] = df['account_age_days']
    features['account_age_log'] = np.log1p(df['account_age_days'])
    features['total_txns'] = df['total_txns']
    features['total_txns_log'] = np.log1p(df['total_txns'])

    features['risk_tier_encoded'] = df['risk_tier'].map(
        {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
    ).fillna(1)
    # Ordinal encoding — LOW=0, MEDIUM=1, HIGH=2
    # Ordinal (not one-hot) because risk_tier has a natural order

    features['is_new_account'] = (df['account_age_days'] < 30).astype(int)
    features['is_very_new_account'] = (df['account_age_days'] < 7).astype(int)

    # ---- Device Features ----
    device_encoder = {'mobile': 0, 'web': 1, 'POS': 2, 'unknown': 3}
    features['device_type_encoded'] = df['device_type'].map(device_encoder).fillna(3)
    features['is_unknown_device'] = (df['device_type'] == 'unknown').astype(int)

    # ---- Merchant Features ----
    features['merchant_fraud_rate'] = df['fraud_rate'].clip(upper=1.0)
    features['merchant_fraud_rate_log'] = np.log1p(df['fraud_rate'] * 100)
    features['is_blacklisted_merchant'] = df['is_blacklisted'].astype(int)
    features['merchant_age_days'] = df['merchant_age_days'].fillna(30)

    # ---- Interaction Features ----
    # These capture combined signals that neither feature alone represents
    features['new_account_x_high_amount'] = (
        features['is_new_account'] * features['amount_z_score']
    )
    # New account making a high-z-score transaction = bust-out fraud pattern

    features['odd_hour_x_unknown_device'] = (
        features['is_odd_hour'] * features['is_unknown_device']
    )
    # Odd hour + unknown device = account takeover pattern

    features['high_risk_merchant_x_large_amount'] = (
        features['merchant_fraud_rate'] * features['amount_log']
    )
    # Risky merchant + large amount = strong fraud signal

    features['new_account_x_high_risk_merchant'] = (
        features['is_new_account'] * features['merchant_fraud_rate']
    )

    # ---- Fill any remaining NaN ----
    features = features.fillna(0)

    return features


FEATURE_COLUMNS = [
    'amount', 'amount_log', 'amount_z_score', 'amount_to_avg_ratio',
    'hour_of_day', 'day_of_week', 'is_odd_hour', 'is_weekend',
    'account_age_days', 'account_age_log', 'total_txns', 'total_txns_log',
    'risk_tier_encoded', 'is_new_account', 'is_very_new_account',
    'device_type_encoded', 'is_unknown_device',
    'merchant_fraud_rate', 'merchant_fraud_rate_log',
    'is_blacklisted_merchant', 'merchant_age_days',
    'new_account_x_high_amount', 'odd_hour_x_unknown_device',
    'high_risk_merchant_x_large_amount', 'new_account_x_high_risk_merchant'
]


# ============================================================
# DATA LOADER
# Pulls training data from PostgreSQL with all needed joins
# ============================================================

def load_training_data(conn) -> pd.DataFrame:
    """
    Loads historical transactions with all user and merchant context.

    Why a JOIN instead of loading tables separately?
    The model needs features from all three tables for each transaction.
    A single JOIN is cleaner and faster than merging in Python.
    """
    query = """
        SELECT
            t.transaction_id,
            t.amount,
            t.hour_of_day,
            t.day_of_week,
            t.device_type,
            t.status,

            u.account_age_days,
            u.avg_txn_amount,
            u.std_txn_amount,
            u.total_txns,
            u.risk_tier,

            m.fraud_rate,
            m.is_blacklisted,
            EXTRACT(DAY FROM NOW() - m.registered_date)::INT as merchant_age_days,
            m.category,

            rs.score as rule_score

        FROM transactions t
        JOIN users u         ON t.user_id     = u.user_id
        JOIN merchants m     ON t.merchant_id = m.merchant_id
        LEFT JOIN risk_scores rs ON t.transaction_id = rs.transaction_id
        WHERE t.created_at < NOW() - INTERVAL '1 minute'
        -- Exclude very recent transactions — they may not be fully processed
    """

    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} transactions from DB")
    return df


def create_labels(df: pd.DataFrame) -> pd.Series:
    """
    Creates binary fraud label from transaction status.

    APPROVED → 0 (legitimate)
    FLAGGED  → 1 (fraud)
    BLOCKED  → 1 (fraud)

    Why combine FLAGGED and BLOCKED as fraud=1?
    Both represent transactions the system determined were fraudulent.
    The distinction (flagged vs blocked) is about severity, not label.
    """
    return df['status'].apply(lambda s: 0 if s == 'APPROVED' else 1)


# ============================================================
# MODEL TRAINER
# ============================================================

class FraudModelTrainer:
    """
    Trains XGBoost on historical transaction data from the DB.
    Evaluates using precision, recall, ROC-AUC, and PR-AUC.
    Saves model + SHAP explainer for use in the live pipeline.
    """

    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = FEATURE_COLUMNS
        os.makedirs(MODEL_DIR, exist_ok=True)

    def train(self, conn):
        print("\n📊 Loading training data...")
        df = load_training_data(conn)

        if len(df) < 100:
            print("❌ Not enough data. Run seed_data_v2.py first.")
            return

        # Create features and labels
        print("⚙️  Engineering features...")
        X = engineer_features(df)[FEATURE_COLUMNS]
        y = create_labels(df)

        fraud_count = y.sum()
        legit_count = len(y) - fraud_count
        print(f"  Class distribution: {legit_count} legitimate, {fraud_count} fraud")
        print(f"  Fraud rate: {fraud_count/len(y)*100:.1f}%")

        # Train/test split — stratified to preserve fraud ratio in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y   # ensures fraud % is same in train and test
        )
        print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        # ---- XGBoost Configuration ----
        scale_pos_weight = legit_count / max(fraud_count, 1)
        # This is the key imbalance fix — tells XGBoost to weight fraud
        # samples more heavily during training

        self.model = XGBClassifier(
            n_estimators=300,
            # 300 trees — enough to capture complex patterns without overfitting
            # More trees = better fit but slower inference

            max_depth=5,
            # Maximum depth of each tree
            # Shallow trees (3-4) = underfits, too deep (8+) = overfits
            # 5 is the sweet spot for fraud detection on tabular data

            learning_rate=0.05,
            # How much each tree corrects the previous one
            # Lower = more trees needed but more robust generalisation
            # Combined with 300 trees, 0.05 gives good coverage

            subsample=0.8,
            # Each tree is trained on 80% of training data (random sample)
            # Prevents any single tree from memorising the training data

            colsample_bytree=0.8,
            # Each tree uses 80% of features (random selection)
            # Forces trees to find different patterns = better ensemble

            scale_pos_weight=scale_pos_weight,
            # THE key parameter for imbalanced data
            # Multiplies the loss for fraud samples

            min_child_weight=5,
            # Minimum sum of sample weights in a leaf node
            # Higher value = more conservative splitting = less overfitting
            # Especially important with imbalanced data

            gamma=0.1,
            # Minimum loss reduction required to make a split
            # Acts as regularisation — prevents useless splits

            reg_alpha=0.1,
            # L1 regularisation — drives some feature weights to zero
            # Helps with feature selection implicitly

            reg_lambda=1.0,
            # L2 regularisation — penalises large weights
            # Standard regularisation, prevents overfitting

            eval_metric='aucpr',
            # Area under Precision-Recall curve
            # Better than AUC-ROC for imbalanced datasets
            # PR-AUC focuses on the minority (fraud) class performance

            random_state=42,
            use_label_encoder=False
        )

        print("\n🚀 Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50   # print progress every 50 trees
        )

        # ---- Evaluation ----
        print("\n📈 Evaluating model...")
        self._evaluate(X_test, y_test)

        # ---- SHAP Explainer ----
        print("\n🔍 Building SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        # TreeExplainer is optimised for tree-based models like XGBoost
        # It's exact (not approximate) and fast

        # ---- Save everything ----
        print("\n💾 Saving model artifacts...")
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.feature_names, FEATURES_PATH)
        joblib.dump(self.explainer, EXPLAINER_PATH)
        print(f"  ✅ Model saved to {MODEL_PATH}")
        print(f"  ✅ Explainer saved to {EXPLAINER_PATH}")

    def _evaluate(self, X_test, y_test):
        """
        Evaluates model performance with fraud-relevant metrics.

        Why not just use accuracy?
        With 90% legitimate transactions, a model predicting "all legitimate"
        achieves 90% accuracy but catches zero fraud — useless.

        Better metrics for fraud:
        - Precision: of transactions we flagged, how many were actually fraud?
        - Recall:    of all actual fraud, how many did we catch?
        - PR-AUC:    area under precision-recall curve (best for imbalance)
        - ROC-AUC:   standard discrimination metric

        The precision-recall tradeoff is the core tension in fraud:
        High recall = catch more fraud but also flag more legitimate (annoying customers)
        High precision = fewer false flags but miss some fraud (real cost)
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        print(f"\n  ROC-AUC:  {roc_auc:.4f}  (target: > 0.85)")
        print(f"  PR-AUC:   {pr_auc:.4f}  (target: > 0.70)")
        print(f"\n  Classification Report (threshold=0.5):")
        print(classification_report(y_test, y_pred,
              target_names=['Legitimate', 'Fraud'], digits=3))

        # Feature importance
        importance = dict(zip(
            FEATURE_COLUMNS,
            self.model.feature_importances_
        ))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
        print("  Top 8 features by importance:")
        for feat, score in top_features:
            bar = '█' * int(score * 100)
            print(f"    {feat:<40} {score:.4f} {bar}")

        # Save metrics
        metrics = {
            'roc_auc':   round(float(roc_auc), 4),
            'pr_auc':    round(float(pr_auc), 4),
            'top_features': [
                {'feature': f, 'importance': round(float(s), 4)}
                for f, s in top_features
            ]
        }
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✅ Metrics saved to {METRICS_PATH}")


# ============================================================
# MODEL SCORER
# Used by the live pipeline to score individual transactions
# ============================================================

class FraudModelScorer:
    """
    Loads the saved model and scores individual live transactions.

    Designed to be instantiated ONCE when the pipeline starts,
    then called repeatedly for each transaction.

    Why load once?
    Loading a model from disk takes ~100ms. If you loaded it per
    transaction, your pipeline would be 100ms slower per transaction.
    Keeping it in memory makes inference < 1ms.
    """

    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Run: python ml_model.py to train first."
            )
        self.model        = joblib.load(MODEL_PATH)
        self.explainer    = joblib.load(EXPLAINER_PATH)
        self.feature_names = joblib.load(FEATURES_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")

    def score(self, txn_row: dict) -> dict:
        """
        Scores a single transaction.

        txn_row: dict with all raw transaction + user + merchant fields
                 (same shape as what load_training_data() returns)

        Returns: dict with score, decision, and SHAP explanation
        """
        # Build single-row dataframe
        df = pd.DataFrame([txn_row])
        X = engineer_features(df)[self.feature_names]

        # Get fraud probability
        fraud_proba = float(self.model.predict_proba(X)[0, 1])

        # SHAP explanation for this specific transaction
        shap_values = self.explainer.shap_values(X)

        # Build explanation — top 5 features driving this score
        feature_impacts = list(zip(
            self.feature_names,
            shap_values[0]
        ))
        # Sort by absolute impact — biggest drivers first
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_explanations = [
            {
                'feature':   feat,
                'impact':    round(float(val), 4),
                'direction': 'increases_risk' if val > 0 else 'decreases_risk',
                'value':     round(float(X[feat].iloc[0]), 4)
            }
            for feat, val in feature_impacts[:5]
        ]

        # Decision based on probability
        if fraud_proba >= 0.75:
            decision = 'BLOCKED'
        elif fraud_proba >= 0.45:
            decision = 'FLAGGED'
        else:
            decision = 'APPROVED'

        return {
            'ml_score':        round(fraud_proba, 4),
            'ml_decision':     decision,
            'top_explanations': top_explanations,
            'model_version':   'xgb_v1.0'
        }


# ============================================================
# COMBINED PIPELINE DECISION
# Merges rule engine result with ML score into final decision
# ============================================================

def make_final_decision(rule_result: dict, ml_result: dict) -> dict:
    """
    Combines rule engine and ML scorer outputs into one final decision.

    Decision hierarchy:
    1. If rule engine hard-blocked → BLOCKED (ML output ignored)
    2. If rule engine didn't send to ML → use rule decision as-is
    3. If both agree → use that decision
    4. If they disagree → weighted combination, ML gets more weight

    Why does ML get more weight in disagreement?
    Rules are deterministic and can be wrong in edge cases.
    ML has learned from thousands of examples and understands
    subtle patterns rules can't express.
    """
    # Rule engine made a hard decision
    if rule_result.get('hard_blocked'):
        return {
            'final_decision': 'BLOCKED',
            'final_score':    1.0,
            'decided_by':     'RULE_ENGINE_HARD_BLOCK',
            'rule_decision':  rule_result['decision'],
            'ml_decision':    ml_result.get('ml_decision', 'N/A'),
            'ml_score':       ml_result.get('ml_score', 0.0)
        }

    rule_score = rule_result.get('score', 0.0)
    ml_score   = ml_result.get('ml_score', rule_score)

    # Weighted combination: 35% rule engine, 65% ML
    # Why 65% ML? Because ML learned from data — rules are heuristics.
    combined_score = (0.35 * rule_score) + (0.65 * ml_score)

    if combined_score >= 0.70:
        final_decision = 'BLOCKED'
    elif combined_score >= 0.40:
        final_decision = 'FLAGGED'
    else:
        final_decision = 'APPROVED'

    return {
        'final_decision': final_decision,
        'final_score':    round(combined_score, 4),
        'rule_score':     round(rule_score, 4),
        'ml_score':       round(ml_score, 4),
        'decided_by':     'COMBINED',
        'rule_decision':  rule_result['decision'],
        'ml_decision':    ml_result.get('ml_decision', 'N/A'),
        'explanations':   ml_result.get('top_explanations', [])
    }


# ============================================================
# QUICK TEST — python ml_model.py
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FinShield — Day 4: ML Model Training")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    trainer = FraudModelTrainer()
    trainer.train(conn)
    conn.close()

    print("\n" + "=" * 60)
    print("Testing scorer on a sample transaction...")
    print("=" * 60)

    scorer = FraudModelScorer()

    # Sample transaction — new account, high amount, odd hour
    sample = {
        'amount':           45000,
        'hour_of_day':      3,
        'day_of_week':      5,
        'device_type':      'unknown',
        'account_age_days': 5,
        'avg_txn_amount':   800,
        'std_txn_amount':   300,
        'total_txns':       2,
        'risk_tier':        'MEDIUM',
        'fraud_rate':       0.08,
        'is_blacklisted':   False,
        'merchant_age_days': 20,
        'category':         'Electronics',
        'rule_score':       0.65
    }

    result = scorer.score(sample)

    print(f"\n  ML Score:    {result['ml_score']}")
    print(f"  ML Decision: {result['ml_decision']}")
    print(f"\n  SHAP Explanation (why this score):")
    for exp in result['top_explanations']:
        arrow = '↑' if exp['direction'] == 'increases_risk' else '↓'
        print(
            f"    {arrow} {exp['feature']:<35} "
            f"impact: {exp['impact']:+.4f}  "
            f"value: {exp['value']}"
        )

    print("\n✅ Day 4 complete.")
    print("Model trained, saved, and scorer tested.")
    print("Next: processor.py (Day 5) — wires everything together.")
