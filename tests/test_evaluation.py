import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.evaluate_models import compare_models, evaluate_regression_model


def test_evaluate_regression_model_returns_numeric_metrics():
    X = pd.DataFrame({"mileage": [10000, 20000, 30000, 40000]})
    y = pd.Series([15000, 13000, 11000, 9000])
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    metrics = evaluate_regression_model(model, X, y)

    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_compare_models_returns_dataframe_with_metrics():
    X = pd.DataFrame({"mileage": [10000, 20000, 30000, 40000]})
    y = pd.Series([15000, 13000, 11000, 9000])
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    comparison = compare_models({"Decision Tree": model}, X, y)

    assert comparison.shape[0] == 1
    assert comparison.columns.tolist() == ["Model", "MAE", "RMSE", "R2"]
    assert comparison.loc[0, "Model"] == "Decision Tree"
