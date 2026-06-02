from pydantic import BaseModel


class CarInput(BaseModel):
    brand: str
    model: str
    year: int
    transmission: str
    mileage: int
    fuelType: str
    tax: float
    mpg: float
    engineSize: float


class PredictionResponse(BaseModel):
    predicted_price: float
