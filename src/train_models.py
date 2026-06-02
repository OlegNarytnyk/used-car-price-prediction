import argparse
from pathlib import Path
import shutil

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from evaluate_models import compare_models
except ModuleNotFoundError:
    from src.evaluate_models import compare_models

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


DATASET_CONFIG = {
    "all": {
        "data_path": Path("data/processed/all_brands_cleaned.csv"),
        "model_path": Path("models/best_model_all_brands.pkl"),
    },
    "ford": {
        "data_path": Path("data/processed/ford_cleaned.csv"),
        "model_path": Path("models/best_model_ford.pkl"),
    },
}
COMPATIBILITY_MODEL_PATH = Path("models/best_model.pkl")


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


def prepare_features(df):
    X = df.drop(columns=["price"])
    y = df["price"]

    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.columns.difference(numeric_columns).tolist()

    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns)

    return X, y


def run_training_pipeline(
    data_path="data/processed/ford_cleaned.csv",
    model_path="models/best_model.pkl",
    compatibility_model_path=None,
):
    data = pd.read_csv(data_path)
    X, y = prepare_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = train_all_models(X_train, y_train)
    comparison = compare_models(models, X_test, y_test)
    comparison = comparison.sort_values(["R2", "RMSE"], ascending=[False, True]).reset_index(drop=True)

    best_model_name = comparison.loc[0, "Model"]
    best_model = models[best_model_name]

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path, compress=3)

    if compatibility_model_path is not None:
        compatibility_model_path = Path(compatibility_model_path)
        compatibility_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_path, compatibility_model_path)

    print(comparison.to_string(index=False))
    print(f"Best model: {best_model_name}")
    print(f"Saved model to: {model_path}")
    if compatibility_model_path is not None:
        print(f"Saved compatibility model to: {compatibility_model_path}")

    return best_model, comparison, model_path


def resolve_dataset_choice(dataset=None):
    if dataset is None:
        dataset = "all" if DATASET_CONFIG["all"]["data_path"].exists() else "ford"

    if dataset not in DATASET_CONFIG:
        raise ValueError("dataset must be one of: all, ford")

    config = DATASET_CONFIG[dataset]
    return dataset, config["data_path"], config["model_path"]


def train_selected_dataset(dataset=None):
    dataset, data_path, model_path = resolve_dataset_choice(dataset)

    if not data_path.exists():
        raise FileNotFoundError(f"Selected dataset not found: {data_path}")

    print(f"Training dataset: {dataset}")
    print(f"Data path: {data_path}")
    return run_training_pipeline(
        data_path=data_path,
        model_path=model_path,
        compatibility_model_path=COMPATIBILITY_MODEL_PATH,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train used car price prediction models.")
    parser.add_argument(
        "--dataset",
        choices=["all", "ford"],
        default=None,
        help="Dataset to train on. Defaults to all if available, otherwise ford.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_selected_dataset(args.dataset)
