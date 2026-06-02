import joblib
import pandas as pd
import pytest

import src.train_models as train_models
from src.train_models import (
    prepare_features,
    resolve_dataset_choice,
    run_training_pipeline,
    train_all_models,
    train_decision_tree,
    train_gradient_boosting,
    train_linear_regression,
    train_random_forest,
)


def sample_training_data():
    return pd.DataFrame(
        {
            "model": ["Fiesta", "Focus", "Fiesta", "Focus", "Mondeo", "Fiesta", "Focus", "Mondeo", "Fiesta", "Focus"],
            "brand": ["Ford"] * 10,
            "year": [2018, 2017, 2019, 2016, 2018, 2020, 2015, 2017, 2016, 2019],
            "transmission": ["Manual", "Manual", "Automatic", "Manual", "Automatic", "Manual", "Manual", "Automatic", "Manual", "Manual"],
            "mileage": [40000, 50000, 20000, 70000, 35000, 10000, 80000, 45000, 65000, 25000],
            "fuelType": ["Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Petrol", "Diesel", "Petrol", "Diesel", "Petrol"],
            "tax": [145, 150, 145, 30, 145, 150, 20, 145, 30, 145],
            "mpg": [58.9, 60.1, 55.4, 67.3, 50.4, 57.7, 70.6, 48.7, 65.7, 56.5],
            "engineSize": [1.0, 1.5, 1.0, 1.6, 2.0, 1.0, 1.5, 2.0, 1.6, 1.0],
            "car_age": [8, 9, 7, 10, 8, 6, 11, 9, 10, 7],
            "price": [10000, 9500, 13000, 7000, 14500, 16000, 6500, 13500, 7200, 14000],
        }
    )


def test_prepare_features_encodes_categorical_columns():
    X, y = prepare_features(sample_training_data())

    assert "price" not in X.columns
    assert len(y) == len(X)
    assert all(dtype.kind in "biufc" for dtype in X.dtypes)


def test_models_can_train_without_errors():
    X, y = prepare_features(sample_training_data())

    models = [
        train_linear_regression(X, y),
        train_decision_tree(X, y),
        train_random_forest(X, y),
        train_gradient_boosting(X, y),
    ]

    assert all(hasattr(model, "predict") for model in models)


def test_train_all_models_returns_expected_models():
    X, y = prepare_features(sample_training_data())

    models = train_all_models(X, y)

    assert set(models.keys()) == {
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    }


def test_training_pipeline_saves_best_model(tmp_path):
    data_path = tmp_path / "ford_cleaned.csv"
    model_path = tmp_path / "models" / "best_model.pkl"
    sample_training_data().to_csv(data_path, index=False)

    best_model, comparison, saved_path = run_training_pipeline(data_path, model_path)

    assert saved_path.exists()
    assert saved_path.name == "best_model.pkl"
    assert hasattr(best_model, "predict")
    assert not comparison.empty
    assert joblib.load(saved_path) is not None


def test_training_pipeline_saves_compatibility_model(tmp_path):
    data_path = tmp_path / "all_brands_cleaned.csv"
    model_path = tmp_path / "models" / "best_model_all_brands.pkl"
    compatibility_path = tmp_path / "models" / "best_model.pkl"
    sample_training_data().to_csv(data_path, index=False)

    run_training_pipeline(data_path, model_path, compatibility_path)

    assert model_path.exists()
    assert compatibility_path.exists()


def test_resolve_dataset_choice_defaults_to_all_when_available(tmp_path, monkeypatch):
    all_data_path = tmp_path / "all_brands_cleaned.csv"
    ford_data_path = tmp_path / "ford_cleaned.csv"
    all_data_path.touch()

    monkeypatch.setattr(
        train_models,
        "DATASET_CONFIG",
        {
            "all": {"data_path": all_data_path, "model_path": tmp_path / "best_model_all_brands.pkl"},
            "ford": {"data_path": ford_data_path, "model_path": tmp_path / "best_model_ford.pkl"},
        },
    )

    dataset, data_path, model_path = resolve_dataset_choice()

    assert dataset == "all"
    assert data_path == all_data_path
    assert model_path.name == "best_model_all_brands.pkl"


def test_resolve_dataset_choice_defaults_to_ford_when_all_missing(tmp_path, monkeypatch):
    all_data_path = tmp_path / "all_brands_cleaned.csv"
    ford_data_path = tmp_path / "ford_cleaned.csv"

    monkeypatch.setattr(
        train_models,
        "DATASET_CONFIG",
        {
            "all": {"data_path": all_data_path, "model_path": tmp_path / "best_model_all_brands.pkl"},
            "ford": {"data_path": ford_data_path, "model_path": tmp_path / "best_model_ford.pkl"},
        },
    )

    dataset, data_path, model_path = resolve_dataset_choice()

    assert dataset == "ford"
    assert data_path == ford_data_path
    assert model_path.name == "best_model_ford.pkl"


def test_resolve_dataset_choice_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        resolve_dataset_choice("toyota")
