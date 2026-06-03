# Used Car Price Prediction

Machine learning project for predicting used car prices using classical regression algorithms.

## Dataset

The project uses the Kaggle "100,000 UK Used Car Data set".

- Raw data: `data/raw/ford.csv`
- Processed data: `data/processed/ford_cleaned.csv`
- Combined processed data: `data/processed/all_brands_cleaned.csv`

## Project Structure

```text
used-car-price-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── model_tuning.py
│   ├── visualization.py
│   └── api/
│       ├── main.py
│       └── schemas.py
├── models/
├── results/
│   ├── plots/
│   └── metrics/
├── requirements.txt
└── README.md
```

## Technologies

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter
- joblib
- FastAPI
- Uvicorn
- Pydantic

## Data Analysis and Preprocessing

Exploratory data analysis is completed in `notebooks/01_eda.ipynb`.

Main EDA steps:

- basic dataset inspection
- missing value analysis
- duplicate row analysis
- target variable analysis
- price, mileage, and year distributions
- price relationships with mileage, year, and engine size
- average price by transmission and fuel type
- correlation heatmap for numeric columns

Preprocessing is completed in `notebooks/02_preprocessing.ipynb`.

Main preprocessing steps:

- duplicate row removal
- safe missing value handling
- `car_age = 2026 - year` feature creation
- categorical encoding setup
- train/test split
- numeric scaling setup
- saving cleaned data to `data/processed/ford_cleaned.csv`

## Machine Learning

The project uses classical machine learning methods only.

Trained regression models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Model evaluation metrics:

- MAE
- RMSE
- R2 Score

Random Forest hyperparameter tuning is implemented with `GridSearchCV` in `src/model_tuning.py`.

## Multi-brand Dataset Support

Supported raw datasets:

- `audi.csv` -> Audi
- `bmw.csv` -> BMW
- `ford.csv` -> Ford
- `merc.csv` -> Mercedes
- `skoda.csv` -> Skoda
- `toyota.csv` -> Toyota
- `vw.csv` -> Volkswagen
- `hyundi.csv` -> Hyundai

Build the combined cleaned dataset:

```bash
python src/build_multibrand_dataset.py
```

Train on all supported brands:

```bash
python src/train_models.py --dataset all
```

Train on Ford only:

```bash
python src/train_models.py --dataset ford
```

The selected best model is also saved to `models/best_model.pkl` for API compatibility.

## Results

Generated results are saved in:

- model comparison: `results/metrics/model_comparison.csv`
- best trained model: `models/best_model.pkl`
- all-brands best trained model: `models/best_model_all_brands.pkl`
- Ford-only best trained model: `models/best_model_ford.pkl`
- plots: `results/plots/`

Model visualizations include:

- actual vs predicted prices
- residual distribution
- feature importance, if supported by the selected model

## FastAPI Prediction Service

Run the API:

```bash
python -m uvicorn src.api.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Example request body for `POST /predict`:

```json
{
  "brand": "Ford",
  "model": "Fiesta",
  "year": 2018,
  "transmission": "Manual",
  "mileage": 40000,
  "fuelType": "Petrol",
  "tax": 145,
  "mpg": 58.9,
  "engineSize": 1.0
}
```

Metadata endpoints:

```text
GET /metadata
GET /brands
GET /models/{brand}
```

Example response:

```json
{
  "predicted_price": 10500.25
}
```

## Prediction Input Validation

The prediction API validates input before calling the trained model.

- `brand`, `model`, `transmission`, and `fuelType` must not be empty.
- `year` must be between 1990 and 2026.
- `mileage` and `tax` must be greater than or equal to 0.
- `mpg` and `engineSize` must be greater than 0.
- Brand, model, transmission, and fuel type are validated against the processed dataset.
- Models are validated per brand, so a Ford model cannot be submitted as a BMW model.

Dropdown-ready metadata is available from:

```text
GET /metadata
GET /brands
GET /models/{brand}
```

`GET /metadata` returns supported brands, models grouped by brand, transmissions, fuel types, and numeric ranges from the processed dataset.

## Model Explainability and Prediction Analysis

Run model analysis:

```bash
python src/model_analysis.py
```

The analysis explains which input features influence the predicted car price the most. For example, mileage and car age usually matter because older cars with more distance driven tend to lose value. Engine size can affect price because larger or more powerful cars often belong to higher-value market segments. Brand and model are also important because different manufacturers and vehicle models have different market demand.

Residual analysis compares the real price with the predicted price. A residual is the difference between actual price and predicted price. If residuals are mostly close to zero, the model is usually making accurate predictions. If residuals are often positive or negative, the model may be underpredicting or overpredicting.

The actual vs predicted plot shows how close model predictions are to real prices. Points near the diagonal line indicate better predictions.

Generated analysis files:

- `results/analysis/model_analysis_summary.csv`
- `results/analysis/largest_prediction_errors.csv`

Generated plots:

- `results/plots/explainability_feature_importance.png`
- `results/plots/explainability_permutation_importance.png`
- `results/plots/prediction_actual_vs_predicted.png`
- `results/plots/prediction_residual_distribution.png`

## Frontend Demo

The project includes a minimal HTML/CSS/JavaScript demo interface served directly by FastAPI.

Start the backend:

```bash
python -m uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The frontend loads brands and models from the same FastAPI app, sends prediction requests to `POST /predict`, and displays the predicted price or API validation errors.

## Docker

The Docker container serves both the FastAPI API and the static frontend demo.

Make sure the processed data and trained model files exist before building the image:

- `data/processed/all_brands_cleaned.csv` or `data/processed/ford_cleaned.csv`
- `models/best_model.pkl`

Build:

```bash
docker build -t used-car-price-prediction .
```

Run:

```bash
docker run -p 8000:8000 used-car-price-prediction
```

Open frontend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_model_training.ipynb
```

Alternatively, preprocessing functions can be run from:

```bash
python src/data_preprocessing.py
```

Build the multi-brand processed dataset:

```bash
python src/build_multibrand_dataset.py
```
