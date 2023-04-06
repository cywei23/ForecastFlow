# Time Series Forecasting Library

A Python library for time series forecasting, providing data preprocessing, feature extraction, forecasting models, and model evaluation functions.

## Features

- Data preprocessing: Handle missing data, resampling, and detrending
- Feature extraction: Extract lag features, rolling statistics, and other time series features
- Forecasting models: ARIMA, Holt-Winters, Exponential Smoothing State Space Model, and more
- Model evaluation: Mean Absolute Error, Mean Squared Error, R-squared, etc.

## Installation

To install the library, use the following command:

```bash
pip install git+https://github.com/cywei23/ForecastFlow.git
```

## Usage
import pandas as pd
from your_library import handle_missing_data, feature_extraction, model_evaluation, forecasting_models

### Load and preprocess data
data = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
handle_missing_data(data, method='interpolation')

### Extract features
features = feature_extraction(data)

### Train and evaluate models
model = forecasting_models.ARIMA(data)
predictions = model.forecast(steps=10)
error = model_evaluation(predictions, data)
