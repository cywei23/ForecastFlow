import numpy as np
import pandas as pd
import pywt
from sklearn.impute import KNNImputer
from scipy.signal import detrend
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import welch
from scipy.stats import entropy

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

def seasonal_decomposition(data, period=None, model='additive'):
    """
    Detrend time series data using seasonal decomposition.

    Parameters:
    data (pd.DataFrame or pd.Series): The time series data to be detrended.
    freq (int): The number of periods in a season. If not provided, it will be inferred.
    model (str): The model used for seasonal decomposition. Options are 'additive', 'multiplicative'.

    Returns:
    pd.DataFrame or pd.Series: The detrended time series data (residual component).
    """
    decomposition = seasonal_decompose(data, period=period, model=model)
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
    if threshold is None:
        threshold = 0.5

    # Apply wavelet transform
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Apply soft or hard thresholding
    coeffs[1:] = [pywt.threshold(coeff, value=threshold, mode=mode) for coeff in coeffs[1:]]

    # Reconstruct data
    detrended_data = pywt.waverec(coeffs, wavelet)

    return pd.Series(detrended_data, index=data.index)


def detrend_data(data, method='subtraction', lag=1, period=None, model='additive', wavelet='db4', level=1, threshold=None, mode='soft'):
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
        if period is None:
            raise ValueError("The 'period' parameter must be provided for seasonal_decomposition method.")
        return seasonal_decomposition(data, period=period, model=model)
    elif method == 'wavelet':
        return wavelet_detrend(data, wavelet=wavelet, level=level, threshold=threshold, mode=mode)
    else:
        raise ValueError("Invalid method. Options are 'subtraction', 'division'.")

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(detrended_data, columns=data.columns, index=data.index)
    elif isinstance(data, pd.Series):
        return pd.Series(detrended_data, index=data.index)


def extract_summary_statistics(data):
    """
    Extract summary statistics from time series data.

    Parameters:
    data (pd.Series): The time series data.

    Returns:
    dict: A dictionary containing summary statistics.
    """
    return {
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'skew': skew(data),
        'kurtosis': kurtosis(data)
    }

def extract_autocorrelation(data, nlags=20):
    """
    Extract autocorrelation features from time series data.

    Parameters:
    data (pd.Series): The time series data.
    nlags (int): Number of lags to include in the autocorrelation function.

    Returns:
    np.array: Array of autocorrelation values.
    """
    return acf(data, nlags=nlags)

def extract_partial_autocorrelation(data, nlags=20):
    """
    Extract partial autocorrelation features from time series data.

    Parameters:
    data (pd.Series): The time series data.
    nlags (int): Number of lags to include in the partial autocorrelation function.

    Returns:
    np.array: Array of partial autocorrelation values.
    """
    return pacf(data, nlags=nlags)

def extract_rolling_statistics(data, window=10):
    """
    Extract rolling statistics from time series data.

    Parameters:
    data (pd.Series): The time series data.
    window (int): Size of the rolling window.

    Returns:
    dict: A dictionary containing rolling statistics.
    """
    return {
        'rolling_mean': data.rolling(window=window).mean(),
        'rolling_std': data.rolling(window=window).std(),
        'rolling_min': data.rolling(window=window).min(),
        'rolling_max': data.rolling(window=window).max(),
    }

def extract_features(data, nlags=20, window=10):
    """
    Extract features from time series data.

    Parameters:
    data (pd.Series): The time series data.
    nlags (int): Number of lags to include in the autocorrelation and partial autocorrelation functions.
    window (int): Size of the rolling window.

    Returns:
    dict: A dictionary containing various features extracted from the time series data.
    """
    features = {}
    features.update(extract_summary_statistics(data))
    features['autocorrelation'] = extract_autocorrelation(data, nlags=nlags)
    features['partial_autocorrelation'] = extract_partial_autocorrelation(data, nlags=nlags)
    features.update(extract_rolling_statistics(data, window=window))
    
    return features


def extract_dwt_coefficients(data, wavelet='db1', level=None):
    """
    Extract Discrete Wavelet Transform (DWT) coefficients from time series data.

    Parameters:
    data (pd.Series): The time series data.
    wavelet (str): The wavelet to use for the DWT. Default is 'db1' (Daubechies wavelet).
    level (int): The decomposition level. If None, the maximum possible level is used.

    Returns:
    dict: A dictionary containing DWT coefficients.
    """
    if level is None:
        level = pywt.dwt_max_level(data_len=len(data), filter_len=pywt.Wavelet(wavelet).dec_len)
    
    coeffs = pywt.wavedec(data, wavelet=wavelet, level=level)
    coeffs_dict = {f'coeff_{i}': coeff for i, coeff in enumerate(coeffs)}
    
    return coeffs_dict

def extract_psd_features(data, nperseg=None):
    """
    Extract Power Spectral Density (PSD) features from time series data.

    Parameters:
    data (pd.Series): The time series data.
    nperseg (int): Length of each segment. If None, the default value is used.

    Returns:
    dict: A dictionary containing PSD features.
    """
    freqs, psd = welch(data, nperseg=nperseg)
    psd_dict = {f'psd_{i}': p for i, p in enumerate(psd)}
    
    return psd_dict

def extract_entropy_features(data, method='shannon'):
    """
    Extract entropy-based features from time series data.

    Parameters:
    data (pd.Series): The time series data.
    method (str): The entropy method to use. Default is 'shannon'.

    Returns:
    float: The entropy value.
    """
    if method == 'shannon':
        return entropy(data.value_counts(normalize=True))
    else:
        raise ValueError("Invalid method. Currently supported: 'shannon'.")

def extract_advanced_features(data, wavelet='db1', level=None, nperseg=None, method='shannon'):
    """
    Extract advanced features from time series data.

    Parameters:
    data (pd.Series): The time series data.
    wavelet (str): The wavelet to use for the DWT. Default is 'db1' (Daubechies wavelet).
    level (int): The decomposition level. If None, the maximum possible level is used.
    nperseg (int): Length of each segment. If None, the default value is used.
    method (str): The entropy method to use. Default is 'shannon'.

    Returns:
    dict: A dictionary containing various advanced features extracted from the time series data.
    """
    features = {}
    features.update(extract_dwt_coefficients(data, wavelet=wavelet, level=level))
    features.update(extract_psd_features(data, nperseg=nperseg))
    features['entropy'] = extract_entropy_features(data, method=method)
    
    return features

def extract_all_features(data, nlags=20, window=10, wavelet='db1', level=None, nperseg=None, method='shannon'):
    """
    Extract all features from time series data.

    Parameters:
    data (pd.Series): The time series data.
    nlags (int): Number of lags to include in the autocorrelation and partial autocorrelation functions.
    window (int): Size of the rolling window.
    wavelet (str): The wavelet to use for the DWT. Default is 'db1' (Daubechies wavelet).
    level (int): The decomposition level. If None, the maximum possible level is used.
    nperseg (int): Length of each segment. If None, the default value is used.
    method (str): The entropy method to use. Default is 'shannon'.

    Returns:
    dict: A dictionary containing various features extracted from the time series data.
    """
    features = {}
    features.update(extract_summary_statistics(data))
    features['autocorrelation'] = extract_autocorrelation(data, nlags=nlags)
    features['partial_autocorrelation'] = extract_partial_autocorrelation(data, nlags=nlags)
    features.update(extract_rolling_statistics(data, window=window))
    features.update(extract_advanced_features(data, wavelet=wavelet, level=level, nperseg=nperseg, method=method))
    
    return features

