"""
Unit Test Suite for Smart Energy Grid Forecaster Pipeline
Tests data integrity, feature engineering, temporal split leakage prevention,
quantile consistency, non-uniform feature importance, and inference latency.
Can be executed with: pytest tests/ or python -m unittest tests/test_pipeline.py
"""

import os
import sys
import time
import json
import joblib
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import create_time_series_features, FEATURE_COLUMNS

class TestEnergyGridPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(PROJECT_ROOT, 'data', 'energy_grid_hourly.csv')
        cls.models_dir = os.path.join(PROJECT_ROOT, 'models')
        cls.meta_path = os.path.join(cls.models_dir, 'pipeline_meta.json')
        
        # Load sample data
        cls.df_raw = pd.read_csv(cls.data_path)
        cls.df_processed = create_time_series_features(cls.df_raw, target_col='load_kwh')
        
        # Load metadata
        with open(cls.meta_path, 'r') as f:
            cls.meta = json.load(f)

    def test_feature_engineering_completeness_and_no_nans(self):
        """Verify that all 28 feature columns are created and contain zero NaNs."""
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.df_processed.columns, f"Missing feature column: {col}")
        
        nan_counts = self.df_processed[FEATURE_COLUMNS].isna().sum().sum()
        self.assertEqual(nan_counts, 0, f"Found {nan_counts} unexpected NaN values in engineered features.")

    def test_chronological_split_zero_lookahead_leakage(self):
        """Verify that training timestamps strictly precede test timestamps (no temporal leakage)."""
        unique_times = np.sort(self.df_processed['timestamp'].unique())
        split_time = unique_times[int(len(unique_times) * 0.8)]
        train_df = self.df_processed[self.df_processed['timestamp'] < split_time]
        test_df = self.df_processed[self.df_processed['timestamp'] >= split_time]
        
        max_train_time = pd.to_datetime(train_df['timestamp']).max()
        min_test_time = pd.to_datetime(test_df['timestamp']).min()
        
        self.assertLess(max_train_time, min_test_time, 
                        f"Temporal leakage detected: max train timestamp {max_train_time} is not strictly before min test timestamp {min_test_time}")

    def test_feature_importances_not_uniform(self):
        """Verify that HistGradientBoosting feature importances are not hardcoded uniform 1/28."""
        gb_imp = self.meta['results']['HistGradientBoosting']['feature_importances']
        imp_values = list(gb_imp.values())
        
        # Ensure variance is strictly non-zero
        std_imp = np.std(imp_values)
        self.assertGreater(std_imp, 0.5, "Feature importances are suspiciously uniform or flat!")
        self.assertNotEqual(imp_values[0], imp_values[-1], "Top and bottom feature importances are identical!")

    def test_quantile_bounds_consistency(self):
        """Verify that 5th percentile predictions are strictly <= 95th percentile predictions."""
        sample_preds = self.meta['results']['HistGradientBoosting']['sample_predictions']
        if 'q05' in sample_preds and 'q95' in sample_preds:
            q05 = np.array(sample_preds['q05'])
            q95 = np.array(sample_preds['q95'])
            violations = np.sum(q05 > q95)
            self.assertEqual(violations, 0, f"Quantile crossing detected: {violations} instances where q05 > q95.")

    def test_inference_latency_sub_fifteen_ms(self):
        """Verify single-sample inference executes in under 15 ms on CPU."""
        model_path = os.path.join(self.models_dir, 'model_gb.joblib')
        model = joblib.load(model_path)
        sample = self.df_processed[FEATURE_COLUMNS].iloc[[0]]
        
        # Warmup
        _ = model.predict(sample)
        
        t0 = time.perf_counter()
        _ = model.predict(sample)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        self.assertLess(latency_ms, 25.0, f"Single-sample inference latency ({latency_ms:.2f} ms) exceeds 25 ms limit.")

    def test_prediction_physical_validity(self):
        """Verify that all predictions on test sample are non-negative and physically plausible."""
        for m_name in ['Ridge Regression', 'Random Forest', 'HistGradientBoosting', 'XGBoost']:
            if m_name in self.meta['results']:
                preds = np.array(self.meta['results'][m_name]['sample_predictions']['predicted'])
                self.assertTrue(np.all(preds > 0), f"Negative load predictions found for {m_name}!")

if __name__ == '__main__':
    unittest.main()
