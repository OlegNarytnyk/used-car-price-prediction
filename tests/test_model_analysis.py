import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.model_analysis import (
    calculate_permutation_importance,
    calculate_residuals,
    generate_predictions,
    get_feature_importance,
    load_dataset,
    load_model,
    save_analysis_summary,
    save_largest_prediction_errors,
    split_features_target,
)


def sample_analysis_data():
    return pd.DataFrame(
        {
            "mileage": [10000, 20000, 30000, 40000, 50000, 60000],
            "car_age": [2, 3, 4, 5, 6, 7],
            "engineSize": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "brand": ["Ford", "Ford", "BMW", "BMW", "Audi", "Audi"],
            "price": [16000, 14500, 18000, 17000, 21000, 19500],
        }
    )


def train_sample_model():
    X, y = split_features_target(sample_analysis_data())
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_load_dataset_and_model(tmp_path):
    data_path = tmp_path / "cars.csv"
    model_path = tmp_path / "model.pkl"
    sample_analysis_data().to_csv(data_path, index=False)
    model, _, _ = train_sample_model()
    joblib.dump(model, model_path)

    loaded_df = load_dataset(data_path)
    loaded_model = load_model(model_path)

    assert loaded_df.shape[0] == 6
    assert hasattr(loaded_model, "predict")


def test_split_features_target_encodes_categorical_columns():
    X, y = split_features_target(sample_analysis_data())

    assert "price" not in X.columns
    assert len(X) == len(y)
    assert all(dtype.kind in "biufc" for dtype in X.dtypes)


def test_generate_predictions_returns_one_prediction_per_row():
    model, X, _ = train_sample_model()

    predictions = generate_predictions(model, X)

    assert len(predictions) == len(X)


def test_residual_calculation_works():
    residuals = calculate_residuals(np.array([100, 200, 300]), np.array([90, 220, 310]))

    assert residuals.tolist() == [10, -20, -10]


def test_feature_importance_extraction_works_for_tree_model():
    model, X, _ = train_sample_model()

    importance = get_feature_importance(model, X.columns, top_n=3)

    assert importance.shape[0] == 3
    assert importance.columns.tolist() == ["feature", "importance"]
    assert importance["importance"].notna().all()


def test_permutation_importance_returns_expected_structure():
    model, X, y = train_sample_model()

    importance = calculate_permutation_importance(model, X, y, top_n=2)

    assert importance.shape[0] == 2
    assert importance.columns.tolist() == ["feature", "importance_mean", "importance_std"]


def test_analysis_summary_file_can_be_created(tmp_path):
    output_path = tmp_path / "analysis" / "summary.csv"
    top_features = pd.DataFrame(
        {
            "feature": ["mileage", "car_age"],
            "importance": [0.7, 0.3],
        }
    )

    saved_path = save_analysis_summary(output_path, {"MAE": 100.0}, top_features)

    saved_df = pd.read_csv(saved_path)
    assert saved_path.exists()
    assert "MAE" in saved_df["metric"].tolist()
    assert "top_feature_1" in saved_df["metric"].tolist()


def test_largest_prediction_errors_file_can_be_generated(tmp_path):
    output_path = tmp_path / "analysis" / "largest_errors.csv"
    X = pd.DataFrame({"mileage": [10000, 20000, 30000]})
    y_true = pd.Series([10000, 12000, 15000])
    y_pred = np.array([9000, 16000, 15100])

    saved_path = save_largest_prediction_errors(output_path, X, y_true, y_pred, top_n=2)

    saved_df = pd.read_csv(saved_path)
    assert saved_path.exists()
    assert saved_df.shape[0] == 2
    assert saved_df.iloc[0]["absolute_error"] == 4000
