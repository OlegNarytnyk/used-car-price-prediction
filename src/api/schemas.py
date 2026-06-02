from pydantic import BaseModel, Field

try:
    from pydantic import field_validator
except ImportError:
    from pydantic import validator as field_validator


class CarInput(BaseModel):
    brand: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    year: int = Field(..., ge=1990, le=2026)
    transmission: str = Field(..., min_length=1)
    mileage: int
    fuelType: str = Field(..., min_length=1)
    tax: float
    mpg: float
    engineSize: float

    @field_validator("brand", "model", "transmission", "fuelType")
    @classmethod
    def text_fields_must_not_be_empty(cls, value):
        if value.strip() == "":
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("mileage")
    @classmethod
    def mileage_must_be_non_negative(cls, value):
        if value < 0:
            raise ValueError("mileage must be greater than or equal to 0")
        return value

    @field_validator("tax")
    @classmethod
    def tax_must_be_non_negative(cls, value):
        if value < 0:
            raise ValueError("tax must be greater than or equal to 0")
        return value

    @field_validator("mpg")
    @classmethod
    def mpg_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError("mpg must be greater than 0")
        return value

    @field_validator("engineSize")
    @classmethod
    def engine_size_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError("engineSize must be greater than 0")
        return value


class PredictionResponse(BaseModel):
    predicted_price: float
