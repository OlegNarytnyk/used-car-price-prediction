import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }


def compare_models(models, X_test, y_test):
    results = []

    for model_name, model in models.items():
        metrics = evaluate_regression_model(model, X_test, y_test)
        metrics["Model"] = model_name
        results.append(metrics)

    return pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2"]]
