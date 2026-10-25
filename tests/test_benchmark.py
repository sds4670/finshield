import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finshield'))
from benchmark import temporal_split, select_threshold, evaluate, run
from ml_model import engineer_features, make_final_decision


class BenchmarkTests(unittest.TestCase):
    def test_equal_timestamps_never_cross_partitions(self):
        df = pd.DataFrame({'Time':np.repeat(np.arange(30),2), 'Class':[0,1]*30})
        a,b,c = temporal_split(df.sample(frac=1, random_state=2))
        self.assertLess(a.Time.max(), b.Time.min())
        self.assertLess(b.Time.max(), c.Time.min())
        self.assertEqual(sum(map(len,[a,b,c])), len(df))

    def test_reject_single_class_period(self):
        with self.assertRaises(ValueError):
            temporal_split(pd.DataFrame({'Time':range(50),'Class':[0]*49+[1]}))

    def test_threshold_matches_brute_force_with_ties(self):
        y=np.array([0,1,0,1,1,0]); scores=np.array([.2,.2,.7,.7,.9,.1])
        threshold=select_threshold(y,scores,2,5)
        actual=evaluate(y,scores,threshold,2,5)['cost_per_transaction']
        possible=[evaluate(y,scores,t,2,5)['cost_per_transaction'] for t in [0,.1,.2,.7,.9,1.01]]
        self.assertAlmostEqual(actual,min(possible))

    def test_invalid_costs_and_scores_rejected(self):
        for cost in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): select_threshold([0,1],[.1,.9],cost,1)
        for y, scores in [([],[]),([0,1],[.1]),([0,1],[.1,float('nan')]),([0,2],[.1,.9])]:
            with self.assertRaises(ValueError): select_threshold(y,scores)

    def test_cost_changes_policy(self):
        y=np.array([0,1,0,1]); s=np.array([.8,.7,.6,.5])
        self.assertGreater(select_threshold(y,s,100,1), select_threshold(y,s,1,100))

    def test_metrics_count_errors(self):
        m=evaluate(np.array([0,0,1,1]),np.array([.1,.9,.2,.8]),.5,2,5)
        self.assertEqual(m['confusion_matrix'],dict(tn=1,fp=1,fn=1,tp=1))
        self.assertEqual(m['cost_per_transaction'],1.75)

    def test_end_to_end_artifact_and_repeatability(self):
        rng=np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as d:
            df=pd.DataFrame({'Time':np.arange(600),'signal':rng.normal(size=600),'noise':rng.normal(size=600)})
            df['Class']=((df.signal+rng.normal(size=600))>1.5).astype(int)
            df.loc[0,'noise']=np.nan
            csv=Path(d)/'fixture.csv'; df.to_csv(csv,index=False)
            a=run(csv,Path(d)/'one',['signal','noise'],'synthetic test fixture only')
            b=run(csv,Path(d)/'two',['signal','noise'],'synthetic test fixture only')
            self.assertEqual(a,b)
            import joblib
            artifact=joblib.load(Path(d)/'one/model.joblib')
            self.assertEqual(artifact['features'],['signal','noise'])
            self.assertEqual(artifact['model'].predict_proba(df[['signal','noise']].iloc[:3]).shape,(3,2))
            self.assertEqual(json.loads((Path(d)/'one/report.json').read_text()),a)
            self.assertNotIn('Time',a['features'])
            with self.assertRaises(ValueError):
                run(csv,Path(d)/'bad',['Class'],'invalid')

    def test_feature_engineering_preserves_index_and_input(self):
        df=pd.DataFrame([dict(amount=10,avg_txn_amount=0,std_txn_amount=0,hour_of_day=3,
            day_of_week=6,account_age_days=2,total_txns=0,risk_tier='NEW',device_type='new',
            fraud_rate=.2,is_blacklisted=False,merchant_age_days=None)],index=[9])
        before=df.copy(deep=True)
        features=engineer_features(df)
        pd.testing.assert_frame_equal(df,before)
        self.assertEqual(list(features.index),[9])
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertEqual(features.loc[9,'device_type_encoded'],3)

    def test_hard_block_overrides_ml(self):
        result=make_final_decision({'hard_blocked':True,'decision':'BLOCKED'}, {'ml_score':0.,'ml_decision':'APPROVED'})
        self.assertEqual(result['final_decision'],'BLOCKED')


if __name__ == '__main__': unittest.main()
