import pandas as pd

from src.data_preprocessing import add_features, clean_data, load_data, save_processed_data


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
