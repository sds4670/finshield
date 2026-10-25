"""Opt-in integration check: requires a seeded DB and trained demo artifacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finshield'))
from fastapi.testclient import TestClient
from api import app
from test_api import PAYLOAD

with TestClient(app) as client:
    for path in ['health','stats','live-feed','merchant-insights','risk-analytics','model-insights']:
        response=client.get('/api/'+path)
        assert response.status_code == 200, (path,response.text)
        assert response.json(), path
    result=client.post('/api/predict',json=PAYLOAD)
    assert result.status_code == 200, result.text
    result=result.json()
    assert 0 <= result['ml_score'] <= 1
    assert len(result['top_explanations']) == 5
    assert result['explanation_units'] == 'log_odds'
    print('All six read endpoints and real model/SHAP prediction passed')
