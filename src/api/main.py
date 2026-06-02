from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import CarInput, PredictionResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
ALL_BRANDS_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_brands_cleaned.csv"
FORD_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ford_cleaned.csv"
CATEGORICAL_COLUMNS = ["brand", "model", "transmission", "fuelType"]


app = FastAPI(
    title="Used Car Price Prediction API",
    description="FastAPI service for predicting used car prices using a classical machine learning model.",
)


def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model file not found: models/best_model.pkl")

    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {error}")


def load_processed_data():
    data_path = ALL_BRANDS_DATA_PATH if ALL_BRANDS_DATA_PATH.exists() else FORD_DATA_PATH

    if not data_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Processed dataset not found: data/processed/all_brands_cleaned.csv or data/processed/ford_cleaned.csv",
        )

    try:
        return pd.read_csv(data_path)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to load processed dataset: {error}")


def match_category_value(df, column, value):
    if column not in df.columns:
        return value

    values = df[column].dropna().astype(str).unique()

    if value in values:
        return value

    normalized_value = value.strip().lower()

    for existing_value in values:
        if existing_value.strip().lower() == normalized_value:
            return existing_value

    return value


def get_training_columns(processed_df, model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    X = processed_df.drop(columns=["price"])
    X = pd.get_dummies(X, columns=[column for column in CATEGORICAL_COLUMNS if column in X.columns])
    return X.columns.tolist()


def get_supported_brands(processed_df):
    if "brand" not in processed_df.columns:
        return ["Ford"]

    return sorted(processed_df["brand"].dropna().astype(str).unique().tolist())


def get_models_for_brand(processed_df, brand):
    if "model" not in processed_df.columns:
        return []

    if "brand" in processed_df.columns:
        brand_values = processed_df["brand"].dropna().astype(str)
        matched_brands = brand_values[brand_values.str.lower() == brand.strip().lower()].unique()

        if len(matched_brands) == 0:
            return []

        processed_df = processed_df[processed_df["brand"].astype(str) == matched_brands[0]]

    return sorted(processed_df["model"].dropna().astype(str).unique().tolist())


def prepare_input(car_input, processed_df, model):
    input_data = car_input.model_dump() if hasattr(car_input, "model_dump") else car_input.dict()

    for column in CATEGORICAL_COLUMNS:
        input_data[column] = match_category_value(processed_df, column, input_data[column])

    input_data["car_age"] = 2026 - input_data["year"]

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(
        input_df,
        columns=[column for column in CATEGORICAL_COLUMNS if column in input_df.columns],
    )

    training_columns = get_training_columns(processed_df, model)
    return input_df.reindex(columns=training_columns, fill_value=0)


@app.get("/")
def root():
    return {"message": "Used Car Price Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(car_input: CarInput):
    model = load_model()
    processed_df = load_processed_data()

    try:
        input_df = prepare_input(car_input, processed_df, model)
        predicted_price = model.predict(input_df)[0]
        return PredictionResponse(predicted_price=round(float(predicted_price), 2))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")


@app.get("/brands")
def brands():
    processed_df = load_processed_data()
    return get_supported_brands(processed_df)


@app.get("/models/{brand}")
def models(brand: str):
    processed_df = load_processed_data()
    return get_models_for_brand(processed_df, brand)
