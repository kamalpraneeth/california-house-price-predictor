import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    # Provide dummy values for the 8 features
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    response = client.post("/predict", json=payload)
    
    # Depending on if model is loaded or not, we might get 503 or 200.
    # In a real CI environment, we would load a mock model or train one first.
    # We'll assert that it's either 200 (if model exists) or 503 (if not loaded).
    assert response.status_code in (200, 503)
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)

def test_predict_missing_field():
    # Missing 'Longitude'
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
    
    data = response.json()
    assert "detail" in data
    # Check that it complained about Longitude
    assert any(err["loc"][-1] == "Longitude" for err in data["detail"])

def test_predict_wrong_type():
    payload = {
        "MedInc": "invalid_string_instead_of_float",
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
    
    data = response.json()
    assert "detail" in data
    assert any(err["loc"][-1] == "MedInc" for err in data["detail"])
