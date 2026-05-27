from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train):
    return {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Decision Tree": train_decision_tree(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "Gradient Boosting": train_gradient_boosting(X_train, y_train),
    }
