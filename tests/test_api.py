import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import src.api.main as api_main
from src.api.main import app, build_metadata, get_models_for_brand, get_supported_brands
from src.api.schemas import CarInput


class FakeModel:
    def predict(self, input_df):
        return [12345.67]


def sample_processed_data():
    return pd.DataFrame(
        {
            "brand": ["Ford", "Ford", "BMW"],
            "model": ["Fiesta", "Focus", "320i"],
            "year": [2018, 2017, 2020],
            "price": [10000, 9500, 22000],
            "transmission": ["Manual", "Automatic", "Manual"],
            "mileage": [40000, 50000, 20000],
            "fuelType": ["Petrol", "Diesel", "Petrol"],
            "tax": [145, 150, 165],
            "mpg": [58.9, 60.1, 48.7],
            "engineSize": [1.0, 1.5, 2.0],
            "car_age": [8, 9, 6],
        }
    )


def valid_payload():
    return {
        "brand": "Ford",
        "model": "Fiesta",
        "year": 2018,
        "transmission": "Manual",
        "mileage": 40000,
        "fuelType": "Petrol",
        "tax": 145,
        "mpg": 58.9,
        "engineSize": 1.0,
    }


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setattr(api_main, "load_model", lambda: FakeModel())
    monkeypatch.setattr(api_main, "load_processed_data", sample_processed_data)
    return TestClient(app)


def test_car_input_schema_includes_brand():
    car = CarInput(
        brand="Ford",
        model="Fiesta",
        year=2018,
        transmission="Manual",
        mileage=40000,
        fuelType="Petrol",
        tax=145,
        mpg=58.9,
        engineSize=1.0,
    )

    assert car.brand == "Ford"


def test_valid_prediction_input_passes(api_client):
    response = api_client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    assert response.json() == {"predicted_price": 12345.67}


def test_root_returns_frontend_html(api_client):
    response = api_client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Used Car Price Prediction" in response.text


def test_static_frontend_files_are_served(api_client):
    css_response = api_client.get("/static/styles.css")
    js_response = api_client.get("/static/app.js")

    assert css_response.status_code == 200
    assert js_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert "javascript" in js_response.headers["content-type"]


def test_health_endpoint_returns_status_ok(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "Used Car Price Prediction API is running",
    }


def test_invalid_brand_returns_error(api_client):
    payload = valid_payload()
    payload["brand"] = "Tesla"

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported brand: Tesla"


def test_invalid_model_for_brand_returns_error(api_client):
    payload = valid_payload()
    payload["brand"] = "BMW"
    payload["model"] = "Focus"

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported model 'Focus' for brand 'BMW'"


def test_invalid_year_fails_validation():
    payload = valid_payload()
    payload["year"] = 1989

    with pytest.raises(ValidationError):
        CarInput(**payload)


def test_negative_mileage_fails_validation():
    payload = valid_payload()
    payload["mileage"] = -1

    with pytest.raises(ValidationError):
        CarInput(**payload)


def test_get_supported_brands_from_processed_dataset():
    df = pd.DataFrame({"brand": ["Ford", "Audi", "Ford"], "model": ["Fiesta", "A3", "Focus"]})

    assert get_supported_brands(df) == ["Audi", "Ford"]


def test_get_supported_brands_falls_back_to_ford_for_ford_only_dataset():
    df = pd.DataFrame({"model": ["Fiesta", "Focus"]})

    assert get_supported_brands(df) == ["Ford"]


def test_get_models_for_brand_filters_case_insensitively():
    df = pd.DataFrame(
        {
            "brand": ["Ford", "Ford", "Audi"],
            "model": ["Fiesta", "Focus", "A3"],
        }
    )

    assert get_models_for_brand(df, "ford") == ["Fiesta", "Focus"]


def test_get_models_for_brand_returns_all_models_for_ford_only_dataset():
    df = pd.DataFrame({"model": ["Fiesta", "Focus", "Fiesta"]})

    assert get_models_for_brand(df, "Ford") == ["Fiesta", "Focus"]


def test_metadata_returns_expected_keys(api_client):
    response = api_client.get("/metadata")

    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "brands",
        "models_by_brand",
        "transmissions",
        "fuel_types",
        "year_range",
        "mileage_range",
        "tax_range",
        "mpg_range",
        "engine_size_range",
    }


def test_build_metadata_uses_processed_dataset_values():
    metadata = build_metadata(sample_processed_data())

    assert metadata["brands"] == ["BMW", "Ford"]
    assert metadata["models_by_brand"]["Ford"] == ["Fiesta", "Focus"]
    assert metadata["transmissions"] == ["Automatic", "Manual"]
    assert metadata["fuel_types"] == ["Diesel", "Petrol"]
    assert metadata["year_range"] == {"min": 2017, "max": 2020}


def test_brands_endpoint_returns_list(api_client):
    response = api_client.get("/brands")

    assert response.status_code == 200
    assert response.json() == ["BMW", "Ford"]


def test_models_endpoint_returns_models_for_valid_brand(api_client):
    response = api_client.get("/models/Ford")

    assert response.status_code == 200
    assert response.json() == ["Fiesta", "Focus"]


def test_models_endpoint_returns_error_for_invalid_brand(api_client):
    response = api_client.get("/models/Tesla")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unsupported brand: Tesla"
