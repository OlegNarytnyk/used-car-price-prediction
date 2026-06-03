from pathlib import Path
import os
import tempfile

import joblib
MPLCONFIG_DIR = Path(tempfile.gettempdir()) / "used_car_price_prediction_matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split


ALL_BRANDS_DATA_PATH = Path("data/processed/all_brands_cleaned.csv")
FORD_DATA_PATH = Path("data/processed/ford_cleaned.csv")
MODEL_PATH = Path("models/best_model.pkl")
ANALYSIS_DIR = Path("results/analysis")
PLOTS_DIR = Path("results/plots")


def load_dataset(path):
    return pd.read_csv(path)


def load_model(path):
    return joblib.load(path)


def split_features_target(df, target_col="price"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.columns.difference(numeric_columns).tolist()

    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns)

    return X, y


def generate_predictions(model, X_test):
    return model.predict(X_test)


def calculate_residuals(y_true, y_pred):
    return np.asarray(y_true) - np.asarray(y_pred)


def get_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def calculate_permutation_importance(model, X_test, y_test, top_n=15):
    X_sample = X_test
    y_sample = y_test

    if len(X_test) > 500:
        X_sample = X_test.sample(n=500, random_state=42)
        y_sample = y_test.loc[X_sample.index]

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=2,
        random_state=42,
        n_jobs=1,
        scoring="neg_mean_absolute_error",
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance_df.sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)


def save_analysis_summary(output_path, metrics_dict, top_features):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [{"metric": key, "value": value} for key, value in metrics_dict.items()]

    for index, row in top_features.reset_index(drop=True).iterrows():
        rows.append(
            {
                "metric": f"top_feature_{index + 1}",
                "value": row["feature"],
            }
        )
        if "importance" in row:
            rows.append(
                {
                    "metric": f"top_feature_{index + 1}_importance",
                    "value": row["importance"],
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)
    return output_path


def save_largest_prediction_errors(output_path, X_test, y_true, y_pred, top_n=25, original_features=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors_df = original_features.copy() if original_features is not None else X_test.copy()
    errors_df["actual_price"] = np.asarray(y_true)
    errors_df["predicted_price"] = np.asarray(y_pred)
    errors_df["residual"] = calculate_residuals(y_true, y_pred)
    errors_df["absolute_error"] = errors_df["residual"].abs()
    errors_df = errors_df.sort_values("absolute_error", ascending=False).head(top_n)
    errors_df.to_csv(output_path, index=False)
    return output_path


def plot_feature_importance(importance_df, output_path, title, value_column="importance"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x=value_column, y="feature", color="#2f80ed")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_actual_vs_predicted(y_true, y_pred, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.35, edgecolor=None)
    plt.plot([min_value, max_value], [min_value, max_value], color="#d64545", linewidth=2)
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_residual_distribution(residuals, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, bins=50, kde=True, color="#2f80ed")
    plt.axvline(0, color="#d64545", linewidth=2)
    plt.title("Residual Error Distribution")
    plt.xlabel("Residual: Actual Price - Predicted Price")
    plt.ylabel("Number of Cars")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def choose_dataset_path():
    return ALL_BRANDS_DATA_PATH if ALL_BRANDS_DATA_PATH.exists() else FORD_DATA_PATH


def run_model_analysis():
    data_path = choose_dataset_path()

    if not data_path.exists():
        raise FileNotFoundError("No processed dataset found in data/processed/")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found: models/best_model.pkl")

    df = load_dataset(data_path)
    model = load_model(MODEL_PATH)
    original_features = df.drop(columns=["price"])
    X, y = split_features_target(df)

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    _ = X_train, y_train

    y_pred = generate_predictions(model, X_test)
    residuals = calculate_residuals(y_test, y_pred)
    average_error = float(np.mean(np.abs(residuals)))
    mean_residual = float(np.mean(residuals))

    if mean_residual > 0:
        prediction_bias = "model tends to underpredict prices"
    elif mean_residual < 0:
        prediction_bias = "model tends to overpredict prices"
    else:
        prediction_bias = "model has no average prediction bias"

    metrics = {
        "dataset_path": str(data_path),
        "rows": len(df),
        "test_rows": len(X_test),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "mean_residual": mean_residual,
        "median_absolute_error": float(median_absolute_error(y_test, y_pred)),
        "average_prediction_error": average_error,
        "prediction_bias": prediction_bias,
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_importance = get_feature_importance(model, X.columns, top_n=15)
    permutation_importance_df = calculate_permutation_importance(model, X_test, y_test, top_n=15)

    summary_path = save_analysis_summary(
        ANALYSIS_DIR / "model_analysis_summary.csv",
        metrics,
        feature_importance,
    )
    largest_errors_path = save_largest_prediction_errors(
        ANALYSIS_DIR / "largest_prediction_errors.csv",
        X_test,
        y_test,
        y_pred,
        original_features=original_features.loc[X_test.index],
    )

    if not feature_importance.empty:
        plot_feature_importance(
            feature_importance,
            PLOTS_DIR / "explainability_feature_importance.png",
            "Model Feature Importance",
        )

    if not permutation_importance_df.empty:
        plot_feature_importance(
            permutation_importance_df,
            PLOTS_DIR / "explainability_permutation_importance.png",
            "Permutation Importance",
            value_column="importance_mean",
        )

    plot_actual_vs_predicted(y_test, y_pred, PLOTS_DIR / "prediction_actual_vs_predicted.png")
    plot_residual_distribution(residuals, PLOTS_DIR / "prediction_residual_distribution.png")

    print(f"Dataset: {data_path}")
    print(f"Rows: {len(df)}")
    print(f"Summary saved to: {summary_path}")
    print(f"Largest errors saved to: {largest_errors_path}")
    print(f"Average prediction error: {average_error:.2f}")
    print(f"Prediction bias: {prediction_bias}")

    return metrics


if __name__ == "__main__":
    run_model_analysis()
