"""
Model Training & Evaluation Pipeline for Energy Grid Forecaster
Implements chronological train-test splitting, feature scaling, model fitting,
evaluation metrics (RMSE, MAE, R², MAPE), Naïve Baseline comparison,
monetary cost impact calculations, and binary artifact export (.joblib).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from features import create_time_series_features, FEATURE_COLUMNS

def calculate_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def run_training_pipeline():
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, 'data', 'energy_grid_hourly.csv')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("[INFO] Loading dataset and constructing time-series features...")
    df_raw = pd.read_csv(data_path)
    df_processed = create_time_series_features(df_raw, target_col='load_kwh')

    X = df_processed[FEATURE_COLUMNS]
    y = df_processed['load_kwh']
    timestamps = df_processed['timestamp'].astype(str)

    # Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(df_processed) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    time_train, time_test = timestamps.iloc[:split_idx], timestamps.iloc[split_idx:]

    print(f"[INFO] Train set size: {len(X_train)} samples | Test set size: {len(X_test)} samples")

    # Fit scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Naïve Baseline Model (Predicts today's demand as yesterday's demand t-24)
    pred_naive_train = X_train['load_lag_24']
    pred_naive_test = X_test['load_lag_24']
    naive_importances = {col: (1.0 if col == 'load_lag_24' else 0.0) for col in FEATURE_COLUMNS}

    # 2. ML Models
    models = {
        'Naïve Baseline (t-24)': None,
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=8, random_state=42)
    }

    results = {}
    fitted_models = {}

    print("[INFO] Training & Evaluating ML Models + Baseline...")
    for name, model in models.items():
        if name == 'Naïve Baseline (t-24)':
            pred_train = pred_naive_train
            pred_test = pred_naive_test
            importances = naive_importances
        elif name == 'Ridge Regression':
            model.fit(X_train_scaled, y_train)
            pred_train = model.predict(X_train_scaled)
            pred_test = model.predict(X_test_scaled)
            importances = dict(zip(FEATURE_COLUMNS, np.abs(model.coef_)))
            fitted_models[name] = model
        else:
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
            else:
                importances = {col: 0.04 for col in FEATURE_COLUMNS}
            fitted_models[name] = model

        train_metrics = calculate_metrics(y_train, pred_train)
        test_metrics = calculate_metrics(y_test, pred_test)

        # Monetary Impact Metric: Error penalty at $0.08 per kWh error ($80/MWh)
        # Annual grid operational error penalty = MAE * 8760 hours * $0.08
        annual_penalty_usd = float(test_metrics['mae'] * 8760 * 0.08)

        # Normalize feature importances to percentage
        total_imp = sum(importances.values())
        norm_importances = {k: float(v / total_imp) for k, v in importances.items()} if total_imp > 0 else importances
        sorted_imp = dict(sorted(norm_importances.items(), key=lambda x: x[1], reverse=True))

        results[name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'annual_penalty_usd': annual_penalty_usd,
            'feature_importances': sorted_imp,
            'sample_predictions': {
                'timestamps': list(time_test.iloc[-168:]),
                'actual': list(np.round(y_test.iloc[-168:], 2)),
                'predicted': list(np.round(pred_test[-168:], 2))
            }
        }

        print(f"\n--- {name} Results ---")
        print(f"Test RMSE: {test_metrics['rmse']:.2f} kWh")
        print(f"Test MAE:  {test_metrics['mae']:.2f} kWh")
        print(f"Test R²:   {test_metrics['r2']:.4f}")
        print(f"Test MAPE: {test_metrics['mape']:.2f}%")
        print(f"Est. Annual Grid Error Penalty: ${annual_penalty_usd:,.2f}")

    # Compute Net Annual Monetary Cost Savings achieved by Gradient Boosting
    gb_penalty = results['Gradient Boosting']['annual_penalty_usd']
    naive_penalty = results['Naïve Baseline (t-24)']['annual_penalty_usd']
    ridge_penalty = results['Ridge Regression']['annual_penalty_usd']

    annual_savings_vs_naive = naive_penalty - gb_penalty
    annual_savings_vs_ridge = ridge_penalty - gb_penalty

    print(f"\n[FINANCIAL IMPACT]")
    print(f"Gradient Boosting Savings vs. Naive Rule: ${annual_savings_vs_naive:,.2f} / year")
    print(f"Gradient Boosting Savings vs. Ridge Baseline: ${annual_savings_vs_ridge:,.2f} / year")

    # Export joblib binaries
    print("\n[INFO] Exporting model binaries and metadata...")
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(fitted_models['Ridge Regression'], os.path.join(models_dir, 'model_ridge.joblib'))
    joblib.dump(fitted_models['Random Forest'], os.path.join(models_dir, 'model_rf.joblib'))
    joblib.dump(fitted_models['Gradient Boosting'], os.path.join(models_dir, 'model_gb.joblib'))

    # Save Pipeline Metadata JSON
    pipeline_meta = {
        'feature_columns': FEATURE_COLUMNS,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'results': results,
        'monetary_impact': {
            'penalty_rate_per_kwh': 0.08,
            'annual_savings_vs_naive': float(annual_savings_vs_naive),
            'annual_savings_vs_ridge': float(annual_savings_vs_ridge)
        },
        'grid_capacity_max_kwh': 50000.0,
        'peak_demand_threshold_kwh': 38000.0
    }

    with open(os.path.join(models_dir, 'pipeline_meta.json'), 'w') as f:
        json.dump(pipeline_meta, f, indent=2)

    print(f"[SUCCESS] All model binaries and metadata exported to {models_dir}")

if __name__ == '__main__':
    run_training_pipeline()
