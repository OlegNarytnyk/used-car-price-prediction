from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def tune_random_forest(X_train, y_train):
    parameter_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    model = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
