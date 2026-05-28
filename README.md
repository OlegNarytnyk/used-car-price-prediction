# Used Car Price Prediction

Machine learning project for predicting used Ford car prices using classical regression algorithms.

## Dataset

The project uses the Kaggle "100,000 UK Used Car Data set".  
Current work is focused on the Ford dataset:

- Raw data: `data/raw/ford.csv`
- Processed data: `data/processed/ford_cleaned.csv`

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

## Results

Generated results are saved in:

- model comparison: `results/metrics/model_comparison.csv`
- best trained model: `models/best_model.pkl`
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

Example response:

```json
{
  "predicted_price": 10500.25
}
```

## Docker

Build:

```bash
docker build -t used-car-price-prediction .
```

Run:

```bash
docker run -p 8000:8000 used-car-price-prediction
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
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
