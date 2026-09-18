"""
Model Training & Evaluation Pipeline for Energy Grid Forecaster
Implements:
1. 5-Fold Walk-Forward Cross-Validation (TimeSeriesSplit) to prove seasonal stability.
2. Quantile Regression (HistGradientBoostingRegressor loss='quantile') for 90% Prediction Intervals (q05, q95).
3. Dual-Task Grid Safety Evaluation: Overload Classification Metrics (Recall, Precision, F1, FNR, Confusion Matrix).
4. Local & Global Explainable AI (SHAP TreeExplainer serialization).
5. Empirical Utility Benchmark Evaluation (PJM / Open Grid reference load).
6. Artifact and Model Serialization (.joblib & pipeline_meta.json).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, confusion_matrix
)
import shap

from features import create_time_series_features, FEATURE_COLUMNS

def calculate_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def calculate_classification_metrics(y_true, y_pred, capacity_series, threshold_ratio=0.90):
    y_true_vals = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    y_pred_vals = y_pred.values if hasattr(y_pred, 'values') else np.array(y_pred)
    cap_vals = capacity_series.values if hasattr(capacity_series, 'values') else np.array(capacity_series)

    overload_threshold = cap_vals * threshold_ratio
    y_true_binary = (y_true_vals >= overload_threshold).astype(int)
    y_pred_binary = (y_pred_vals >= overload_threshold).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'precision': round(precision * 100.0, 2),
        'recall': round(recall * 100.0, 2),
        'specificity': round(specificity * 100.0, 2),
        'f1': round(f1 * 100.0, 2),
        'fnr': round(fnr * 100.0, 2),
        'total_overload_events': int(tp + fn),
        'predicted_overload_events': int(tp + fp)
    }

def run_walk_forward_cv(X, y, n_splits=5):
    print(f"\n[INFO] Executing {n_splits}-Fold Walk-Forward Cross-Validation (TimeSeriesSplit)...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = {
        'Ridge Regression': [],
        'Random Forest': [],
        'Gradient Boosting': [],
        'Naïve Baseline (t-24)': []
    }

    fold = 1
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Naive Baseline
        pred_naive = X_val['load_lag_24']
        cv_results['Naïve Baseline (t-24)'].append(calculate_metrics(y_val, pred_naive))

        # Ridge
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        ridge = Ridge(alpha=10.0).fit(X_tr_sc, y_tr)
        pred_ridge = ridge.predict(X_val_sc)
        cv_results['Ridge Regression'].append(calculate_metrics(y_val, pred_ridge))

        # Random Forest (fast config for CV folds)
        rf = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        pred_rf = rf.predict(X_val)
        cv_results['Random Forest'].append(calculate_metrics(y_val, pred_rf))

        # Gradient Boosting
        gb = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_depth=8, random_state=42).fit(X_tr, y_tr)
        pred_gb = gb.predict(X_val)
        cv_results['Gradient Boosting'].append(calculate_metrics(y_val, pred_gb))

        print(f"  Fold {fold}/{n_splits} | Train: {len(X_tr)} | Val: {len(X_val)} | GB Val RMSE: {cv_results['Gradient Boosting'][-1]['rmse']:.2f} kWh")
        fold += 1

    # Summarize mean & std
    summary = {}
    for model_name, folds in cv_results.items():
        summary[model_name] = {
            'rmse_mean': round(float(np.mean([f['rmse'] for f in folds])), 2),
            'rmse_std': round(float(np.std([f['rmse'] for f in folds])), 2),
            'mae_mean': round(float(np.mean([f['mae'] for f in folds])), 2),
            'mae_std': round(float(np.std([f['mae'] for f in folds])), 2),
            'r2_mean': round(float(np.mean([f['r2'] for f in folds])), 4),
            'r2_std': round(float(np.std([f['r2'] for f in folds])), 4),
            'mape_mean': round(float(np.mean([f['mape'] for f in folds])), 2),
            'mape_std': round(float(np.std([f['mape'] for f in folds])), 2),
            'fold_details': folds
        }
    return summary

def run_training_pipeline():
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, 'data', 'energy_grid_hourly.csv')
    benchmark_path = os.path.join(project_root, 'data', 'empirical_grid_benchmark.csv')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("[INFO] Loading primary dataset and engineering time-series features...")
    df_raw = pd.read_csv(data_path)
    df_processed = create_time_series_features(df_raw, target_col='load_kwh')

    X = df_processed[FEATURE_COLUMNS]
    y = df_processed['load_kwh']
    timestamps = df_processed['timestamp'].astype(str)
    capacities = df_processed['transformer_capacity']

    # 1. 5-Fold Walk-Forward Cross-Validation
    cv_summary = run_walk_forward_cv(X, y, n_splits=5)

    # 2. Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(df_processed) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    time_train, time_test = timestamps.iloc[:split_idx], timestamps.iloc[split_idx:]
    cap_test = capacities.iloc[split_idx:]

    print(f"\n[INFO] Chronological Split: Train={len(X_train)} samples | Test={len(X_test)} samples")

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Training
    models = {
        'Naïve Baseline (t-24)': None,
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=8, random_state=42)
    }

    results = {}
    fitted_models = {}

    print("\n[INFO] Training Primary Regressors...")
    for name, model in models.items():
        if name == 'Naïve Baseline (t-24)':
            pred_train = X_train['load_lag_24']
            pred_test = X_test['load_lag_24']
            importances = {col: (1.0 if col == 'load_lag_24' else 0.0) for col in FEATURE_COLUMNS}
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
        classification_metrics = calculate_classification_metrics(y_test, pred_test, cap_test, threshold_ratio=0.90)

        # Operational penalty at $0.08/kWh ($80/MWh)
        annual_penalty_usd = float(test_metrics['mae'] * 8760 * 0.08)

        total_imp = sum(importances.values())
        norm_importances = {k: float(v / total_imp) for k, v in importances.items()} if total_imp > 0 else importances
        sorted_imp = dict(sorted(norm_importances.items(), key=lambda x: x[1], reverse=True))

        results[name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'classification_metrics': classification_metrics,
            'annual_penalty_usd': annual_penalty_usd,
            'feature_importances': sorted_imp,
            'sample_predictions': {
                'timestamps': list(time_test.iloc[-168:]),
                'actual': list(np.round(y_test.iloc[-168:], 2)),
                'predicted': list(np.round(pred_test[-168:], 2)),
                'capacity': list(np.round(cap_test.iloc[-168:], 2))
            }
        }

        print(f"--- {name} ---")
        print(f"  Test RMSE: {test_metrics['rmse']:.2f} kWh | MAE: {test_metrics['mae']:.2f} kWh | R²: {test_metrics['r2']:.4f}")
        print(f"  Overload Recall (Sensitivity): {classification_metrics['recall']:.1f}% | Precision: {classification_metrics['precision']:.1f}% | FNR: {classification_metrics['fnr']:.1f}%")

    # 4. Train Quantile Regressors for 90% Prediction Interval (5th & 95th Percentile)
    print("\n[INFO] Training Quantile Regressors (5th & 95th percentiles)...")
    model_q05 = HistGradientBoostingRegressor(loss="quantile", quantile=0.05, max_iter=150, learning_rate=0.08, max_depth=8, random_state=42)
    model_q95 = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, max_iter=150, learning_rate=0.08, max_depth=8, random_state=42)

    model_q05.fit(X_train, y_train)
    model_q95.fit(X_train, y_train)

    pred_test_q05 = model_q05.predict(X_test)
    pred_test_q95 = model_q95.predict(X_test)

    # Coverage probability and interval width
    coverage_prob = float(np.mean((y_test.values >= pred_test_q05) & (y_test.values <= pred_test_q95)) * 100.0)
    mpiw = float(np.mean(pred_test_q95 - pred_test_q05))
    print(f"  90% Prediction Interval Coverage: {coverage_prob:.2f}% (Target: ~90%)")
    print(f"  Mean Prediction Interval Width (MPIW): {mpiw:.2f} kWh")

    # Embed quantile predictions in Gradient Boosting results
    results['Gradient Boosting']['quantile_metrics'] = {
        'target_coverage_pct': 90.0,
        'empirical_coverage_pct': round(coverage_prob, 2),
        'mean_interval_width_kwh': round(mpiw, 2)
    }
    results['Gradient Boosting']['sample_predictions']['q05'] = list(np.round(pred_test_q05[-168:], 2))
    results['Gradient Boosting']['sample_predictions']['q95'] = list(np.round(pred_test_q95[-168:], 2))

    # 5. Local & Global Explainability via SHAP TreeExplainer
    print("\n[INFO] Fitting SHAP TreeExplainer on Champion Gradient Boosting Model...")
    champion_gb = fitted_models['Gradient Boosting']
    explainer = shap.TreeExplainer(champion_gb)
    
    # Calculate SHAP values for the test 168-hour window
    test_168_X = X_test.iloc[-168:]
    shap_vals_168 = explainer(test_168_X)

    # Global SHAP importance (mean absolute SHAP across 500 samples)
    sample_500 = X_test.sample(n=min(500, len(X_test)), random_state=42)
    shap_vals_500 = explainer(sample_500)
    global_shap_imp = dict(zip(FEATURE_COLUMNS, np.mean(np.abs(shap_vals_500.values), axis=0)))
    total_shap = sum(global_shap_imp.values())
    norm_global_shap = {k: round(float(v / total_shap) * 100.0, 2) for k, v in sorted(global_shap_imp.items(), key=lambda x: x[1], reverse=True)}

    base_val = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)

    shap_summary = {
        'base_value': round(base_val, 2),
        'feature_names': FEATURE_COLUMNS,
        'timestamps_168': list(time_test.iloc[-168:]),
        'features_168': test_168_X.values.tolist(),
        'shap_values_168': np.round(shap_vals_168.values, 4).tolist(),
        'global_shap_importance_pct': norm_global_shap
    }

    # 6. Empirical Benchmark Validation (PJM reference load)
    print("\n[INFO] Evaluating Generalization on Empirical Utility Benchmark...")
    benchmark_results = {}
    if os.path.exists(benchmark_path):
        df_bench_raw = pd.read_csv(benchmark_path)
        df_bench = create_time_series_features(df_bench_raw, target_col='load_kwh')
        X_bench = df_bench[FEATURE_COLUMNS]
        y_bench = df_bench['load_kwh']
        cap_bench = df_bench['transformer_capacity']
        bench_split = int(len(df_bench) * 0.8)

        X_b_train, X_b_test = X_bench.iloc[:bench_split], X_bench.iloc[bench_split:]
        y_b_train, y_b_test = y_bench.iloc[:bench_split], y_bench.iloc[bench_split:]
        cap_b_test = cap_bench.iloc[bench_split:]

        # Train a benchmark reference model
        gb_bench = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=8, random_state=42)
        gb_bench.fit(X_b_train, y_b_train)
        pred_b_test = gb_bench.predict(X_b_test)
        pred_b_naive = X_b_test['load_lag_24']

        bench_gb_metrics = calculate_metrics(y_b_test, pred_b_test)
        bench_naive_metrics = calculate_metrics(y_b_test, pred_b_naive)
        bench_class_metrics = calculate_classification_metrics(y_b_test, pred_b_test, cap_b_test, threshold_ratio=0.90)

        benchmark_results = {
            'dataset_name': 'Empirical Substation Utility Benchmark (PJM / Open Power Load)',
            'total_hours': len(df_bench),
            'test_hours': len(X_b_test),
            'gradient_boosting_metrics': bench_gb_metrics,
            'naive_baseline_metrics': bench_naive_metrics,
            'classification_metrics': bench_class_metrics,
            'rmse_reduction_pct': round((1.0 - (bench_gb_metrics['rmse'] / bench_naive_metrics['rmse'])) * 100.0, 2)
        }
        print(f"  Benchmark Test RMSE: {bench_gb_metrics['rmse']:.2f} kWh vs Naive {bench_naive_metrics['rmse']:.2f} kWh ({benchmark_results['rmse_reduction_pct']}% error reduction)")
        print(f"  Benchmark Overload Recall: {bench_class_metrics['recall']:.1f}% | Precision: {bench_class_metrics['precision']:.1f}%")

    # 7. Export Model Binaries and Explainer
    print("\n[INFO] Serializing Model Binaries & Artifacts...")
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(fitted_models['Ridge Regression'], os.path.join(models_dir, 'model_ridge.joblib'))
    joblib.dump(fitted_models['Random Forest'], os.path.join(models_dir, 'model_rf.joblib'))
    joblib.dump(fitted_models['Gradient Boosting'], os.path.join(models_dir, 'model_gb.joblib'))
    joblib.dump(model_q05, os.path.join(models_dir, 'model_gb_q05.joblib'))
    joblib.dump(model_q95, os.path.join(models_dir, 'model_gb_q95.joblib'))
    joblib.dump(shap_summary, os.path.join(models_dir, 'shap_summary.joblib'))

    # Monetary cost savings
    gb_penalty = results['Gradient Boosting']['annual_penalty_usd']
    naive_penalty = results['Naïve Baseline (t-24)']['annual_penalty_usd']
    ridge_penalty = results['Ridge Regression']['annual_penalty_usd']
    annual_savings_vs_naive = naive_penalty - gb_penalty
    annual_savings_vs_ridge = ridge_penalty - gb_penalty

    # Save Comprehensive Pipeline Metadata JSON
    pipeline_meta = {
        'feature_columns': FEATURE_COLUMNS,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'walk_forward_cv_summary': cv_summary,
        'results': results,
        'empirical_benchmark_results': benchmark_results,
        'monetary_impact': {
            'penalty_rate_per_kwh': 0.08,
            'annual_savings_vs_naive': round(float(annual_savings_vs_naive), 2),
            'annual_savings_vs_ridge': round(float(annual_savings_vs_ridge), 2)
        },
        'grid_capacity_max_kwh': round(float(df_processed['load_kwh'].max() * 1.15), 1),
        'peak_demand_threshold_kwh': round(float(df_processed['load_kwh'].quantile(0.92)), 1)
    }

    with open(os.path.join(models_dir, 'pipeline_meta.json'), 'w') as f:
        json.dump(pipeline_meta, f, indent=2)

    print(f"\n[SUCCESS] Pipeline completed successfully! Artifacts written to {models_dir}")

if __name__ == '__main__':
    run_training_pipeline()
