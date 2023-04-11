import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import DateOffset
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class BaseForecastModel(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, start, end):
        pass

    @abstractmethod
    def evaluate(self, data, train_size, test_size):
        pass


class ARIMAForecastModel(BaseForecastModel):
    def __init__(self, order=(1, 0, 0)):
        self.model = None
        self.order = order

    def fit(self, data):
        self.model = ARIMA(data, order=self.order)
        self.model = self.model.fit()

    def predict(self, start, end):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")
        
        return self.model.predict(start=start, end=end)

    def evaluate(self, data, train_size, test_size):
        train_data = data[:train_size]
        test_data = data[train_size:train_size + test_size]
        self.fit(train_data)
        predictions = self.predict(train_data.index[-1] + DateOffset(1), test_data.index[-1])
        mse = mean_squared_error(test_data, predictions)
        return mse
    

class SARIMAXForecastModel(BaseForecastModel):
    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
        self.model = None
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, data):
        self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
        self.model = self.model.fit()

    def predict(self, start, end):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")
        
        return self.model.predict(start=start, end=end)

    def evaluate(self, data, train_size, test_size):
        train_data = data[:train_size]
        test_data = data[train_size:train_size + test_size]
        self.fit(train_data)
        predictions = self.predict(train_data.index[-1] + DateOffset(1), test_data.index[-1])
        mse = mean_squared_error(test_data, predictions)
        return mse

        
class LSTMForecastModel(BaseForecastModel):
    def __init__(self, look_back=1, lstm_units=50, epochs=50, batch_size=1):
        self.model = None
        self.look_back = look_back
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size

    def create_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back), 0])
            y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(y)

    def fit(self, data):
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data.values.reshape(-1, 1))

        # Create dataset
        X, y = self.create_dataset(data)

        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # Define and fit the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_units, input_shape=(1, self.look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
    
    def predict(self, data, start, end, scaler):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Start and end indices must be integers.")

        if end < start:
            raise ValueError("End index must be greater than or equal to start index.")

        num_predictions = end - start + 1

        # Generate predictions
        predictions = []
        input_data = data[start - self.look_back:start].values.reshape(-1, 1)
        input_data = scaler.transform(input_data)
        # Change the shape of the input data based on the look_back value
        input_data = input_data.reshape(1, 1, self.look_back)

        for _ in range(num_predictions):
            prediction = self.model.predict(input_data)
            predictions.append(prediction[0, 0])

            # Update input_data by removing the first element and appending the new prediction
            input_data = np.append(input_data[0, 0, :][1:], prediction)
            input_data = input_data.reshape(1, 1, self.look_back)  # Updated line

        # Inverse scale predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return pd.Series(predictions.ravel(), index=data.index[start:end+1])

    def evaluate(self, data, train_size, test_size):
        train_data = data[:train_size]
        test_data = data[train_size:train_size+test_size]
        self.fit(train_data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data.values.reshape(-1, 1))
        predictions = self.predict(data, train_size, train_size + test_size - 1, scaler)
        mse = mean_squared_error(test_data, predictions)
        return mse


class ProphetForecastModel(BaseForecastModel):
    def __init__(self):
        self.model = None

    def fit(self, data):
        data = data.reset_index()
        data.columns = ['ds', 'y']
        self.model = Prophet()
        self.model.fit(data)

    def predict(self, start, end):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")

        if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
            raise ValueError("Start and end should be pandas Timestamp objects.")

        future_dates = pd.date_range(start=start, end=end)
        future_df = pd.DataFrame(future_dates, columns=['ds'])

        forecast = self.model.predict(future_df)
        predictions = forecast[['ds', 'yhat']]
        predictions.set_index('ds', inplace=True)

        return predictions.loc[start:end]

    def evaluate(self, data, train_size, test_size):
        train_data = data[:train_size]
        test_data = data[train_size:train_size+test_size]
        self.fit(train_data)
        predictions = self.predict(test_data.index[0], test_data.index[-1]).reindex(test_data.index)
        mse = mean_squared_error(test_data, predictions)
        return mse



class XGBoostForecastModel(BaseForecastModel):
    def __init__(self, look_back=1, max_depth=3, n_estimators=100, learning_rate=0.1):
        self.model = None
        self.look_back = look_back
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def create_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back), 0])
            y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(y)

    def fit(self, data):
        # Create dataset
        X, y = self.create_dataset(data.values)

        # Define and fit the XGBoost model
        self.model = xgb.XGBRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
        self.model.fit(X, y)

    def predict(self, data, start, end):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")
        
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Start and end indices must be integers.")

        if end < start:
            raise ValueError("End index must be greater than or equal to start index.")
        
        num_predictions = end - start + 1

        # Generate predictions
        predictions = []
        input_data = data[start - self.look_back:start].values.reshape(1, -1)
        
        for _ in range(num_predictions):
            prediction = self.model.predict(input_data)
            predictions.append(prediction[0])

            # Update input_data by removing the first element and appending the new prediction
            input_data = np.append(input_data[0, 1:], prediction).reshape(1, -1)
        
        return pd.Series(predictions, index=data.index[start:end+1])

    def evaluate(self, data, train_size, test_size):
        train_data = data[:train_size]
        test_data = data[train_size:train_size+test_size]
        self.fit(train_data)
        predictions = self.predict(data, train_size, train_size + test_size - 1)
        mse = mean_squared_error(test_data, predictions)
        return mse




class TransformerForecastModel():
    def __init__(
        self, 
        look_back=10, 
        num_layers=2, 
        d_model=128, 
        nhead=4, 
        dim_feedforward=512, 
        epochs=50, 
        batch_size=1, 
        learning_rate=0.001
    ):
        self.look_back = look_back
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        # Add an embedding layer
        self.embedding = nn.Linear(1, self.d_model)

    def create_dataset(self, data):
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back), 0])
            y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(y)

    def fit(self, data):
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data.values.reshape(-1, 1))

        # Create dataset
        X, y = self.create_dataset(data)

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Create DataLoader
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Define the Transformer model
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward),
            num_layers=self.num_layers
        )
        self.model.to("cpu")

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.epochs):
            for batch_X, batch_y in train_loader:
                # Project the input data to the required embedding dimension
                batch_X = batch_X.view(self.batch_size, self.look_back, -1).transpose(0, 1).to("cpu")
                batch_X = self.embedding(batch_X)
                batch_y = batch_y.view(self.batch_size, 1, -1).transpose(0, 1).to("cpu")

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output[-1], batch_y)
                loss.backward()
                optimizer.step()
        
    def predict(self, data, start, end):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Start and end indices must be integers.")

        if end < start:
            raise ValueError("End index must be greater than or equal to start index.")

        num_predictions = end - start + 1

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data.values.reshape(-1, 1))

        predictions = []
        for _ in range(num_predictions):
            input_seq = data[start - self.look_back:start]
            input_seq = torch.tensor(input_seq, dtype=torch.float32).view(1, self.look_back, -1).transpose(0, 1).to("cpu")

            # Project the input data to the required embedding dimension
            input_seq = self.embedding(input_seq)

            with torch.no_grad():
                output = self.model(input_seq)
                prediction = output[-1, 0, 0].item()

            predictions.append(prediction)
            start += 1

        # Inverse scale predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return predictions


    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return mse



