import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import src.model_tuning as model_tuning


class FakeGridSearchCV:
    def __init__(self, estimator, param_grid, cv, scoring, n_jobs):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def fit(self, X_train, y_train):
        self.best_estimator_ = self.estimator.fit(X_train, y_train)
        self.best_params_ = {"n_estimators": 100, "max_depth": 10}
        self.best_score_ = 0.9
        return self


def test_tune_random_forest_returns_best_estimator(monkeypatch):
    monkeypatch.setattr(model_tuning, "GridSearchCV", FakeGridSearchCV)

    X = pd.DataFrame(
        {
            "mileage": [10000, 20000, 30000, 40000, 50000],
            "car_age": [3, 4, 5, 6, 7],
        }
    )
    y = pd.Series([15000, 13000, 11000, 9000, 8000])

    best_estimator, best_params, best_score = model_tuning.tune_random_forest(X, y)

    assert isinstance(best_estimator, RandomForestRegressor)
    assert best_params["n_estimators"] == 100
    assert best_score == 0.9
