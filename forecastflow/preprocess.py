import numpy as np
import pandas as pd
import pywt
from sklearn.impute import KNNImputer
from scipy.signal import detrend
from statsmodels.tsa.seasonal import seasonal_decompose

def handle_missing_data(data, method='interpolation', inplace=True, axis=0, limit=None, fill_value=None, order=None, k=None):
    """
    Handle missing data in a time series dataset.

    Parameters:
    data (pd.DataFrame or pd.Series): The time series data with missing values.
    method (str): The method to handle missing data. Options are 'forward_fill', 'backward_fill', 'interpolation',
                  'mean', 'median', 'mode'.
    inplace (bool): If True, perform the operation inplace and return None. Otherwise, return a new DataFrame or Series.
    axis (int): 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
    limit (int): Optional. Maximum number of consecutive missing values to fill.
    fill_value (float, dict, or Series): Optional. Value to use for filling holes in data.
    
    order (int): Optional. The order of the polynomial for polynomial interpolation.
    k (int): Optional. The number of nearest neighbors to consider for KNN imputation.

    Returns:
    pd.DataFrame or pd.Series or None: The DataFrame or Series with missing data handled.
    """
    if method == 'forward_fill':
        if inplace:
            data.ffill(axis=axis, limit=limit, inplace=True)
        else:
            return data.ffill(axis=axis, limit=limit)
    elif method == 'backward_fill':
        if inplace:
            data.bfill(axis=axis, limit=limit, inplace=True)
        else:
            return data.bfill(axis=axis, limit=limit)
    elif method == 'interpolation':
        if order is not None:
            method = 'polynomial'
        else:
            method = 'linear'
        if inplace:
            data.interpolate(method=method, order=order, axis=axis, limit=limit, inplace=True)
        else:
            return data.interpolate(method=method, order=order, axis=axis, limit=limit)
    elif method == 'mean':
        if fill_value is None:
            fill_value = data.mean(axis=axis)
        if inplace:
            data.fillna(value=fill_value, axis=axis, inplace=True)
        else:
            return data.fillna(value=fill_value, axis=axis)
    elif method == 'median':
        if fill_value is None:
            fill_value = data.median(axis=axis)
        if inplace:
            data.fillna(value=fill_value, axis=axis, inplace=True)
        else:
            return data.fillna(value=fill_value, axis=axis)
    elif method == 'mode':
        if fill_value is None:
            fill_value = data.mode(axis=axis).iloc[0]
        if inplace:
            data.fillna(value=fill_value, axis=axis, inplace=True)
        else:
            return data.fillna(value=fill_value, axis=axis)
    elif method == 'knn':
        if k is None:
            k = 3
        imputer = KNNImputer(n_neighbors=k)
        if isinstance(data, pd.DataFrame):
            columns = data.columns
            index = data.index
            imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=columns, index=index)
        elif isinstance(data, pd.Series):
            index = data.index
            imputed_data = pd.Series(imputer.fit_transform(data.to_frame()).ravel(), index=index)
        if inplace:
            data.update(imputed_data)
        else:
            return imputed_data
    else:
        raise ValueError("Invalid method. Options are 'forward_fill', 'backward_fill', 'interpolation', "
                         "'mean', 'median', 'mode'.")


def resample_data(data, freq='D', method='mean'):
    """
    Resample time series data based on a specified frequency and aggregation method.

    Parameters:
    data (pd.DataFrame or pd.Series): The time series data to be resampled.
    freq (str): The new frequency for the resampled data.
    method (str): The aggregation method to use when resampling. Options are 'mean', 'sum', 'min', 'max', 'first', 'last', 'median'.

    Returns:
    pd.DataFrame or pd.Series: The resampled time series data.
    """
    if method == 'mean':
        return data.resample(freq).mean()
    elif method == 'sum':
        return data.resample(freq).sum()
    elif method == 'min':
        return data.resample(freq).min()
    elif method == 'max':
        return data.resample(freq).max()
    elif method == 'first':
        return data.resample(freq).first()
    elif method == 'last':
        return data.resample(freq).last()
    elif method == 'median':
        return data.resample(freq).median()
    else:
        raise ValueError("Invalid method. Options are 'mean', 'sum', 'min', 'max', 'first', 'last', 'median'.")

def differencing(data, lag=1):
    """
    Detrend time series data by computing the difference between consecutive observations.

    Parameters:
    data (pd.DataFrame or pd.Series): The time series data to be detrended.
    lag (int): The number of lags to use for differencing. Default is 1.

    Returns:
    pd.DataFrame or pd.Series: The detrended time series data.
    """
    return data.diff(lag).dropna()

def seasonal_decomposition(data, freq=None, model='additive'):
    """
    Detrend time series data using seasonal decomposition.

    Parameters:
    data (pd.DataFrame or pd.Series): The time series data to be detrended.
    freq (int): The number of periods in a season. If not provided, it will be inferred.
    model (str): The model used for seasonal decomposition. Options are 'additive', 'multiplicative'.

    Returns:
    pd.DataFrame or pd.Series: The detrended time series data (residual component).
    """
    decomposition = seasonal_decompose(data, freq=freq, model=model)
    return decomposition.resid.dropna()


def wavelet_detrend(data, wavelet='db4', level=1, threshold=None, mode='soft'):
    """
    Detrend time series data using wavelet transform.

    Parameters:
    data (pd.DataFrame or pd.Series): The time series data to be detrended.
    wavelet (str): The type of wavelet to use for wavelet transform. Default is 'db4'.
    level (int): The level of decomposition for the wavelet transform. Default is 1.
    threshold (float): The threshold for wavelet coefficient thresholding. If not provided, it will be calculated.
    mode (str): The thresholding mode. Options are 'soft', 'hard'.

    Returns:
    pd.DataFrame or pd.Series: The detrended time series data.
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    if threshold is None:
        # Calculate threshold using universal threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

    if mode == 'soft':
        thresholding_func = pywt.threshold_soft
    elif mode == 'hard':
        thresholding_func = pywt.threshold_hard
    else:
        raise ValueError("Invalid mode. Options are 'soft', 'hard'.")

    detrended_coeffs = [thresholding_func(c, threshold) for c in coeffs]
    detrended_data = pywt.waverec(detrended_coeffs, wavelet)

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(detrended_data[:len(data)], columns=data.columns, index=data.index)
    elif isinstance(data, pd.Series):
        return pd.Series(detrended_data[:len(data)], index=data.index)


def detrend_data(data, method='subtraction', lag=1, freq=None, model='additive', wavelet='db4', level=1, threshold=None, mode='soft'):
    """
    Detrend time series data by removing the linear trend.

    Parameters:
    data (pd.DataFrame or pd.Series): The time series data to be detrended.
    method (str): The method to use for detrending. Options are 'subtraction', 'division'.

    Returns:
    pd.DataFrame or pd.Series: The detrended time series data.
    """
    if method == 'subtraction':
        detrended_data = detrend(data, type='linear')
    elif method == 'division':
        linear_trend = np.polyfit(np.arange(len(data)), data, 1)[0] * np.arange(len(data))
        detrended_data = data / (linear_trend + 1)
    elif method == 'differencing':
        return differencing(data, lag=lag)
    elif method == 'seasonal_decomposition':
        return seasonal_decomposition(data, freq=freq, model=model)
    elif method == 'wavelet':
        return wavelet_detrend(data, wavelet=wavelet, level=level, threshold=threshold, mode=mode)
    else:
        raise ValueError("Invalid method. Options are 'subtraction', 'division'.")

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(detrended_data, columns=data.columns, index=data.index)
    elif isinstance(data, pd.Series):
        return pd.Series(detrended_data, index=data.index)



  
# Example usage
if __name__ == "__main__":
    # ...

    # Create another time series with missing data
    data3 = pd.DataFrame({"value": [7, None, None, 10, None, 12]}, index=pd.date_range("2021-01-01", periods=6, freq="D"))

    print("\nOriginal data with missing values:")
    print(data3)

    # Handle missing data using polynomial interpolation with order=2
    handle_missing_data(data3, method='interpolation', order=2)

    print("\nData after handling missing values with polynomial interpolation:")
    print(data3)

    # Create another time series with missing data
    data4 = pd.DataFrame({"value": [1, None, 3, None, 5, None]}, index=pd.date_range("2021-01-01", periods=6, freq="D"))

    print("\nOriginal data with missing values:")
    print(data4)

    # Handle missing data using KNN imputation with k=2
    handle_missing_data(data4, method='knn', k=2)

    print("\nData after handling missing values with KNN imputation:")
    print(data4)

    # Handle missing data using KNN imputation with k=2
    handle_missing_data(data4, method='knn', k=2)

    print("\nData after handling missing values with KNN imputation:")
    print(data4)
