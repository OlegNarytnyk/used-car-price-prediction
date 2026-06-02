from pathlib import Path

try:
    from data_preprocessing import (
        add_features,
        clean_data,
        get_available_brand_datasets,
        load_all_brand_datasets,
        save_processed_data,
    )
except ModuleNotFoundError:
    from src.data_preprocessing import (
        add_features,
        clean_data,
        get_available_brand_datasets,
        load_all_brand_datasets,
        save_processed_data,
    )


def build_multibrand_dataset(
    raw_data_dir="data/raw",
    output_path="data/processed/all_brands_cleaned.csv",
):
    available_brand_files = get_available_brand_datasets(raw_data_dir)
    loaded_brands = [brand_name for _, brand_name in available_brand_files]

    data = load_all_brand_datasets(raw_data_dir)
    data = clean_data(data)
    data = add_features(data)
    save_processed_data(data, output_path)

    return data, loaded_brands, Path(output_path)


if __name__ == "__main__":
    combined_data, brands, saved_path = build_multibrand_dataset()

    print(f"Loaded brands: {', '.join(brands)}")
    print(f"Total rows: {len(combined_data)}")
    print(f"Output path: {saved_path}")
