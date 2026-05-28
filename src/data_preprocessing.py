from pathlib import Path

import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    cleaned_df = df.copy()

    cleaned_df = cleaned_df.drop_duplicates()

    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns
    categorical_columns = cleaned_df.columns.difference(numeric_columns)

    for column in numeric_columns:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    for column in categorical_columns:
        mode_values = cleaned_df[column].mode()
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    return cleaned_df


def add_features(df):
    featured_df = df.copy()
    featured_df["car_age"] = 2026 - featured_df["year"]
    return featured_df


def save_processed_data(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    raw_path = Path("data/raw/ford.csv")
    processed_path = Path("data/processed/ford_cleaned.csv")

    data = load_data(raw_path)
    data = clean_data(data)
    data = add_features(data)
    save_processed_data(data, processed_path)
    print(f"Processed data saved to {processed_path}")
