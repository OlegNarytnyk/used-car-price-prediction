import pytest
import pandas as pd

from src.build_multibrand_dataset import build_multibrand_dataset
from src.data_preprocessing import (
    add_features,
    clean_data,
    load_all_brand_datasets,
    load_brand_dataset,
    load_data,
    save_processed_data,
)


def test_load_data_reads_csv(tmp_path):
    data_path = tmp_path / "cars.csv"
    pd.DataFrame(
        {
            "model": ["Fiesta"],
            "year": [2018],
            "price": [10000],
        }
    ).to_csv(data_path, index=False)

    df = load_data(data_path)

    assert df.shape == (1, 3)
    assert "price" in df.columns


def test_load_brand_dataset_adds_brand_column(tmp_path):
    data_path = tmp_path / "ford.csv"
    pd.DataFrame({"model": ["Fiesta"], "price": [10000]}).to_csv(data_path, index=False)

    df = load_brand_dataset(data_path, "Ford")

    assert df.loc[0, "brand"] == "Ford"


def test_load_all_brand_datasets_combines_existing_supported_files(tmp_path):
    pd.DataFrame({"model": ["Fiesta"], "price": [10000]}).to_csv(tmp_path / "ford.csv", index=False)
    pd.DataFrame({"model": ["A3"], "price": [15000]}).to_csv(tmp_path / "audi.csv", index=False)

    df = load_all_brand_datasets(tmp_path)

    assert df.shape[0] == 2
    assert set(df["brand"]) == {"Ford", "Audi"}


def test_load_all_brand_datasets_raises_when_no_supported_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_brand_datasets(tmp_path)


def test_clean_data_removes_duplicates_and_missing_values():
    df = pd.DataFrame(
        {
            "model": ["Fiesta", "Fiesta", None],
            "year": [2018, 2018, None],
            "price": [10000, 10000, 12000],
        }
    )

    cleaned_df = clean_data(df)

    assert cleaned_df.duplicated().sum() == 0
    assert cleaned_df.isna().sum().sum() == 0


def test_add_features_creates_car_age():
    df = pd.DataFrame({"year": [2018, 2020], "price": [10000, 13000]})

    featured_df = add_features(df)

    assert "car_age" in featured_df.columns
    assert featured_df["car_age"].tolist() == [8, 6]


def test_save_processed_data_creates_file(tmp_path):
    output_path = tmp_path / "processed" / "ford_cleaned.csv"
    df = pd.DataFrame({"price": [10000]})

    save_processed_data(df, output_path)

    assert output_path.exists()


def test_build_multibrand_dataset_saves_cleaned_all_brands_file(tmp_path):
    raw_path = tmp_path / "raw"
    output_path = tmp_path / "processed" / "all_brands_cleaned.csv"
    raw_path.mkdir()
    sample = pd.DataFrame(
        {
            "model": ["Fiesta", "Fiesta"],
            "year": [2018, 2018],
            "price": [10000, 10000],
            "transmission": ["Manual", "Manual"],
            "mileage": [40000, 40000],
            "fuelType": ["Petrol", "Petrol"],
            "tax": [145, 145],
            "mpg": [58.9, 58.9],
            "engineSize": [1.0, 1.0],
        }
    )
    sample.to_csv(raw_path / "ford.csv", index=False)

    df, brands, saved_path = build_multibrand_dataset(raw_path, output_path)

    assert saved_path.exists()
    assert brands == ["Ford"]
    assert len(df) == 1
    assert "brand" in df.columns
    assert "car_age" in df.columns
