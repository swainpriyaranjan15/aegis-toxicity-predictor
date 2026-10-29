import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import app
from src.config import LABEL_COLUMNS, SEVERITY_LEVELS, MODERATION_ACTIONS

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data
    assert "labels" in data
    assert data["labels"] == LABEL_COLUMNS
    assert "model_version" in data

def test_serve_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Toxicity & Cyberbullying Predictor" in response.text or "text/html" in response.headers.get("content-type", "")

def test_benign_comment(client):
    payload = {"text": "People should respect each other's beliefs and treat each other kindly."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Schema check
    assert data["prediction"] in ["Toxic", "Non-toxic"]
    assert 0.0 <= data["score"] <= 1.0
    assert 0.0 <= data["score_percent"] <= 100.0
    assert data["highest_label"] in LABEL_COLUMNS
    assert data["severity"] in SEVERITY_LEVELS
    assert data["moderation_action"] in MODERATION_ACTIONS
    assert isinstance(data["reason"], str) and len(data["reason"]) > 0
    assert data["threshold"] == 0.5
    assert data["inference_ms"] >= 0.0
    for lbl in LABEL_COLUMNS:
        assert lbl in data["probabilities"]
        assert 0.0 <= data["probabilities"][lbl] <= 1.0

def test_toxic_comment(client):
    payload = {"text": "You are a stupid idiot and I hate you so much!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Toxic", "Non-toxic"]
    assert data["severity"] in SEVERITY_LEVELS
    assert data["moderation_action"] in MODERATION_ACTIONS

def test_multi_label_comment(client):
    payload = {"text": "I will kill you, you disgusting piece of trash!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["detected_labels"], list)

def test_empty_input_422(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_whitespace_only_422(client):
    response = client.post("/predict", json={"text": "     \n\t   "})
    assert response.status_code == 422

def test_overlength_input_422(client):
    long_text = "word " * 500  # > 2000 chars
    assert len(long_text) > 2000
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422

def test_special_characters_emoji_url_200(client):
    text = "Check this out https://example.com/test! 😡🔥 #toxic #hate @someone 12345 %^&*()"
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data

def test_predict_batch(client):
    payload = {
        "texts": [
            "Hello there, have a great day!",
            "You are an idiot!",
            "I love programming and open source software."
        ]
    }
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["count"] == 3
    assert len(data["results"]) == 3
    for res in data["results"]:
        assert "prediction" in res
        assert "score" in res
        assert "severity" in res

def test_predict_batch_over_50_items_422(client):
    payload = {"texts": [f"Sample comment {i}" for i in range(51)]}
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 422

def test_predict_cyberbully(client):
    payload = {"text": "I hate you because of your religion and race!"}
    response = client.post("/predict-cyberbully", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "cyberbully_detected" in data
    assert "top_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 4
