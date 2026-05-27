# Used Car Price Prediction

Machine learning project for predicting used car prices using classical regression algorithms.

## Dataset

100,000 UK Used Car Data Set from Kaggle.

## Technologies

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter

## Phase 2: Data Preparation and EDA

This phase focuses on exploratory data analysis and preprocessing for a classical machine learning regression project.

- Dataset used: Kaggle "100,000 UK Used Car Data set", currently using `data/raw/ford.csv`.
- EDA completed in `notebooks/01_eda.ipynb`: dataset inspection, missing values, duplicates, target variable analysis, distribution plots, relationship plots, category-based average price plots, and numeric correlation heatmap.
- Preprocessing completed in `notebooks/02_preprocessing.ipynb`: duplicate removal, safe missing-value handling, `car_age = 2026 - year` feature creation, categorical encoding, train/test split, and numeric scaling setup.
- Cleaned data is saved to `data/processed/ford_cleaned.csv`.
- The project uses classical machine learning methods for used car price regression. No deep learning, API, frontend, or deployment logic is included in this phase.

## Planned Classical ML Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
