import pandas as pd

from src.api.main import get_models_for_brand, get_supported_brands
from src.api.schemas import CarInput


def test_car_input_schema_includes_brand():
    car = CarInput(
        brand="Ford",
        model="Fiesta",
        year=2018,
        transmission="Manual",
        mileage=40000,
        fuelType="Petrol",
        tax=145,
        mpg=58.9,
        engineSize=1.0,
    )

    assert car.brand == "Ford"


def test_get_supported_brands_from_processed_dataset():
    df = pd.DataFrame({"brand": ["Ford", "Audi", "Ford"], "model": ["Fiesta", "A3", "Focus"]})

    assert get_supported_brands(df) == ["Audi", "Ford"]


def test_get_supported_brands_falls_back_to_ford_for_ford_only_dataset():
    df = pd.DataFrame({"model": ["Fiesta", "Focus"]})

    assert get_supported_brands(df) == ["Ford"]


def test_get_models_for_brand_filters_case_insensitively():
    df = pd.DataFrame(
        {
            "brand": ["Ford", "Ford", "Audi"],
            "model": ["Fiesta", "Focus", "A3"],
        }
    )

    assert get_models_for_brand(df, "ford") == ["Fiesta", "Focus"]


def test_get_models_for_brand_returns_all_models_for_ford_only_dataset():
    df = pd.DataFrame({"model": ["Fiesta", "Focus", "Fiesta"]})

    assert get_models_for_brand(df, "Ford") == ["Fiesta", "Focus"]
