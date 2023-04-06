import unittest
import pandas as pd
import numpy as np
from forecastflow import handle_missing_data, resample_data, detrend_data

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        self.data = np.sin(2 * np.pi * self.date_rng.dayofyear / 365)
        self.noise = np.random.normal(0, 0.1, len(self.date_rng))
        self.data_with_noise = self.data + self.noise

        # Insert missing values
        self.data_with_missing = self.data_with_noise.copy()
        self.data_with_missing[30:35] = np.nan
        self.data_with_missing[100:110] = np.nan

    def test_handle_missing_data(self):
        filled_data = handle_missing_data(pd.Series(self.data_with_missing, index=self.date_rng), method='linear')
        self.assertFalse(filled_data.isnull().any(), "Filled data should not contain missing values")

    def test_resample_data(self):
        resampled_data = resample_data(pd.Series(self.data_with_noise, index=self.date_rng), freq='W', method='mean')
        self.assertEqual(len(resampled_data), 53, "Resampled data should have 53 weeks")

    def test_detrend_data(self):
        detrended_data_sub = detrend_data(pd.Series(self.data_with_noise, index=self.date_rng), method='subtraction')
        self.assertEqual(len(detrended_data_sub), len(self.date_rng), "Detrended data should have the same length as original data")

        detrended_data_diff = detrend_data(pd.Series(self.data_with_noise, index=self.date_rng), method='differencing', lag=1)
        self.assertEqual(len(detrended_data_diff), len(self.date_rng) - 1, "Detrended data should have one less data point than original data")

        detrended_data_seasonal = detrend_data(pd.Series(self.data_with_noise, index=self.date_rng), method='seasonal_decomposition', freq=365)
        self.assertEqual(len(detrended_data_seasonal), len(self.date_rng), "Detrended data should have the same length as original data")

        detrended_data_wavelet = detrend_data(pd.Series(self.data_with_noise, index=self.date_rng), method='wavelet', wavelet='db4', level=1)
        self.assertEqual(len(detrended_data_wavelet), len(self.date_rng), "Detrended data should have the same length as original data")

if __name__ == '__main__':
    unittest.main()
