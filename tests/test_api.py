import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'finshield'))
from fastapi.testclient import TestClient
import api

PAYLOAD=dict(amount=45000,hour_of_day=3,day_of_week=5,device_type='unknown',
    account_age_days=5,avg_txn_amount=800,std_txn_amount=300,total_txns=2,
    risk_tier='MEDIUM',fraud_rate=.08,is_blacklisted=False,merchant_age_days=20)

class ApiTests(unittest.TestCase):
    def setUp(self): self.client=TestClient(api.app)

    def test_prediction_delegates_validated_input(self):
        scorer=Mock(); scorer.score.return_value={'ml_score':.8,'explanation_units':'log_odds'}
        with patch.object(api,'get_scorer',return_value=scorer):
            response=self.client.post('/api/predict',json=PAYLOAD)
        self.assertEqual(response.status_code,200)
        scorer.score.assert_called_once_with(PAYLOAD)

    def test_invalid_requests_do_not_load_model(self):
        with patch.object(api,'get_scorer') as load:
            for change in [dict(amount=-1),dict(hour_of_day=24),dict(device_type='invalid'),dict(fraud_rate=2),dict(unexpected=1)]:
                self.assertEqual(self.client.post('/api/predict',json=PAYLOAD|change).status_code,422)
            load.assert_not_called()

    def test_missing_model_is_service_unavailable(self):
        with patch.object(api,'get_scorer',side_effect=FileNotFoundError):
            self.assertEqual(self.client.post('/api/predict',json=PAYLOAD).status_code,503)

    def test_health_database_failure(self):
        import psycopg2
        with patch.object(api,'query_one',side_effect=psycopg2.OperationalError):
            self.assertEqual(self.client.get('/api/health').status_code,503)
