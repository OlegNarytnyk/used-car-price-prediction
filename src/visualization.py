from pathlib import Path

import matplotlib.pyplot as plt
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
