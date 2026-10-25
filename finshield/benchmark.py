"""Independent-label, temporal fraud benchmark. Never trains on decision statuses.

CSV contract: Time (numeric event time), Class (0/1), numeric features.
Choose features explicitly: never pass outcomes, identifiers or future aggregates.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             confusion_matrix, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def temporal_split(frame):
    """Keep equal timestamps together; reject unusable single-class periods."""
    times = np.sort(frame['Time'].unique())
    if len(times) < 10:
        raise ValueError('Need at least 10 distinct event times')
    first, second = times[int(len(times) * .6)], times[int(len(times) * .8)]
    parts = [frame[frame.Time < first],
             frame[(frame.Time >= first) & (frame.Time < second)],
             frame[frame.Time >= second]]
    if any(set(p.Class.unique()) != {0, 1} for p in parts):
        raise ValueError('Each chronological partition must contain both classes')
    return parts


def select_threshold(y, scores, fp_cost=1., fn_cost=25.):
    """Exact O(n log n) validation sweep, including the predict-none policy."""
    if not np.isfinite([fp_cost, fn_cost]).all() or fp_cost <= 0 or fn_cost <= 0:
        raise ValueError('Costs must be positive')
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.ndim != 1 or not len(y) or len(y) != len(scores):
        raise ValueError('Labels and scores must be nonempty aligned vectors')
    if not set(np.unique(y)) <= {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Require binary labels and finite scores')
    order = np.argsort(-scores, kind='stable')
    ordered, labels = scores[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered)-1]
    tp = np.cumsum(labels)[ends]
    fp = ends + 1 - tp
    costs = np.r_[y.sum()*fn_cost, fp*fp_cost + (y.sum()-tp)*fn_cost]
    thresholds = np.r_[np.nextafter(ordered[0], np.inf), ordered[ends]]
    # On a tie prefer the higher threshold / fewer alerts.
    return float(thresholds[np.argmin(costs)])


def evaluate(y, scores, threshold, fp_cost, fn_cost):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(average_precision=float(average_precision_score(y, scores)),
                roc_auc=float(roc_auc_score(y, scores)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                false_positive_rate=float(fp / (fp+tn)),
                alert_rate=float(np.mean(pred)),
                cost_per_transaction=float((fp*fp_cost+fn*fn_cost)/len(y)),
                confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))


def run(csv, output, features, provenance, fp_cost=1., fn_cost=25.):
    if not np.isfinite([fp_cost, fn_cost]).all() or fp_cost <= 0 or fn_cost <= 0:
        raise ValueError('Costs must be positive')
    if not features or len(set(features)) != len(features) or {'Time','Class'} & set(features):
        raise ValueError('Specify unique features excluding Time and Class')
    output = Path(output)
    if any((output / name).exists() for name in ['report.json','model.joblib','requirements-resolved.txt']):
        raise ValueError('Use a fresh output directory; existing experiment artifacts are immutable')
    frame = pd.read_csv(csv)
    if not {'Time', 'Class', *features} <= set(frame.columns):
        raise ValueError('CSV is missing required columns')
    if frame[['Time','Class']].isna().any().any() or set(frame.Class.unique()) != {0,1}:
        raise ValueError('Time and binary Class must be present for every row')
    for col in ['Time', *features]:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if np.isinf(frame[col]).any():
            raise ValueError('Infinite inputs are not supported')
    train, valid, test = temporal_split(frame)
    if train[features].isna().all().any():
        raise ValueError('Training features cannot be entirely missing')
    models = {
        'prior': make_pipeline(SimpleImputer(strategy='median'), DummyClassifier(strategy='prior')),
        'logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
        'xgboost': make_pipeline(SimpleImputer(strategy='median'), XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=.05, subsample=.8,
            colsample_bytree=.8, tree_method='hist', n_jobs=2, random_state=42,
            scale_pos_weight=float((train.Class == 0).sum()/train.Class.sum()), eval_metric='aucpr')),
    }
    report = dict(dataset_sha256=hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
                  provenance=provenance, label='Externally supplied Class; independence must be audited by dataset owner',
                  split='60/20/20 distinct timestamps; equal timestamps never cross boundaries',
                  features=features, fp_cost=fp_cost, fn_cost=fn_cost,
                  python=platform.python_version(),
                  partitions={name: dict(rows=len(p), fraud_rate=float(p.Class.mean()),
                              start=float(p.Time.min()), end=float(p.Time.max()))
                              for name,p in zip(['train','validation','test'],[train,valid,test])},
                  models={})
    for name, model in models.items():
        model.fit(train[features], train.Class)
        scores = model.predict_proba(valid[features])[:,1]
        threshold = select_threshold(valid.Class, scores, fp_cost, fn_cost)
        report['models'][name] = dict(threshold=threshold,
            validation=evaluate(valid.Class, scores, threshold, fp_cost, fn_cost))
    # Select without test feedback, then evaluate every frozen candidate for comparison.
    winner = min(models, key=lambda n: report['models'][n]['validation']['cost_per_transaction'])
    report['selected_by_validation_cost'] = winner
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        report['models'][name]['test'] = evaluate(test.Class, model.predict_proba(test[features])[:,1],
            report['models'][name]['threshold'], fp_cost, fn_cost)
    joblib.dump(dict(model=models[winner], features=features,
        threshold=report['models'][winner]['threshold'], dataset_sha256=report['dataset_sha256']), output/'model.joblib')
    report['artifact_sha256'] = hashlib.sha256((output/'model.joblib').read_bytes()).hexdigest()
    from importlib.metadata import distributions
    (output/'requirements-resolved.txt').write_text('\n'.join(sorted(
        f'{d.metadata["Name"]}=={d.version}' for d in distributions()))+'\n')
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--provenance', required=True, help='Dataset source, version, license and label origin')
    parser.add_argument('--fp-cost', type=float, default=1.)
    parser.add_argument('--fn-cost', type=float, default=25.)
    args = parser.parse_args()
    result = run(args.csv, args.output, args.features, args.provenance, args.fp_cost, args.fn_cost)
    print(json.dumps(result, indent=2))
