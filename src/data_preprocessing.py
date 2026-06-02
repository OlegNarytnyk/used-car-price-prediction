from pathlib import Path

import pandas as pd


SUPPORTED_BRAND_DATASETS = {
    "audi.csv": "Audi",
    "bmw.csv": "BMW",
    "ford.csv": "Ford",
    "merc.csv": "Mercedes",
    "skoda.csv": "Skoda",
    "toyota.csv": "Toyota",
    "vw.csv": "Volkswagen",
    "hyundi.csv": "Hyundai",
}


def load_data(path):
    return pd.read_csv(path)


def load_brand_dataset(file_path, brand_name):
    df = load_data(file_path)
    df["brand"] = brand_name
    return df


def get_available_brand_datasets(raw_data_dir="data/raw"):
    raw_data_dir = Path(raw_data_dir)
    return [
        (raw_data_dir / file_name, brand_name)
        for file_name, brand_name in SUPPORTED_BRAND_DATASETS.items()
        if (raw_data_dir / file_name).exists()
    ]


def load_all_brand_datasets(raw_data_dir="data/raw"):
    datasets = [
        load_brand_dataset(file_path, brand_name)
        for file_path, brand_name in get_available_brand_datasets(raw_data_dir)
    ]

    if not datasets:
        raise FileNotFoundError(f"No supported brand CSV files found in {raw_data_dir}")

    return pd.concat(datasets, ignore_index=True)


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
