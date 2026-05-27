from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PLOTS_DIR = Path("results/plots")


def _save_plot(filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return output_path


def plot_price_distribution(df):
    plt.figure(figsize=(9, 5))
    sns.histplot(df["price"], bins=40, kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Number of Cars")
    return _save_plot("price_distribution.png")


def plot_price_vs_mileage(df):
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x="mileage", y="price", alpha=0.45)
    plt.title("Price vs Mileage")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    return _save_plot("price_vs_mileage.png")


def plot_price_by_year(df):
    plt.figure(figsize=(10, 5))
    yearly_price = df.groupby("year", as_index=False)["price"].mean()
    sns.lineplot(data=yearly_price, x="year", y="price", marker="o")
    plt.title("Average Price by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Price")
    return _save_plot("price_by_year.png")


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["number"])

    plt.figure(figsize=(9, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    return _save_plot("correlation_heatmap.png")


def plot_actual_vs_predicted(y_true, y_pred, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.45)
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="red")
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_residual_distribution(y_true, y_pred, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=40, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Number of Cars")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_feature_importance(model, feature_names, output_path, top_n=15):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_).ravel()
    else:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    )
    importance = importance.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="importance", y="feature")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path
