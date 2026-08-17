"""
Smart Energy Grid Peak-Demand Forecaster
3rd Year AIML Capstone Project Streamlit Web Application
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

# Set Page Config
st.set_page_config(
    page_title="Smart Energy Grid Forecaster | AIML Capstone",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling (Dark Glassmorphism & Neon Accents)
st.markdown("""
<style>
    /* Global Styling */
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }
    
    /* Metric Card Styling */
    .metric-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Alert Status Cards */
    .status-normal {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }
    
    .status-critical {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_models_and_meta():
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        ridge = joblib.load(os.path.join(MODELS_DIR, 'model_ridge.joblib'))
        rf = joblib.load(os.path.join(MODELS_DIR, 'model_rf.joblib'))
        gb = joblib.load(os.path.join(MODELS_DIR, 'model_gb.joblib'))
        
        with open(os.path.join(MODELS_DIR, 'pipeline_meta.json'), 'r') as f:
            meta = json.load(f)
            
        return {
            'scaler': scaler,
            'models': {
                'Ridge Regression': ridge,
                'Random Forest': rf,
                'Gradient Boosting': gb
            },
            'meta': meta
        }
    except Exception as e:
        return None

artifacts = load_models_and_meta()



# Top Header
st.markdown("""
<div class="header-card">
    <div class="header-title">⚡ Smart Energy Grid Peak-Demand Forecaster</div>
    <div class="header-subtitle">3rd Year AIML Capstone Project | Short-Term Electricity Load & Peak Strain Forecasting Pipeline</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "📌 Capstone Overview",
    "📊 Model Evaluation & Metrics",
    "⚡ Live 24h Peak Forecaster",
    "🔍 Feature Engineering & Importance"
])
tab1, tab2, tab3, tab4 = tabs

@st.cache_data
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

DATA_DIR = os.path.join(BASE_DIR, 'data')
tab_bg_map = {
    "📌 Capstone Overview": os.path.join(DATA_DIR, "CapstoneOverview.webp"),
    "📊 Model Evaluation & Metrics": os.path.join(DATA_DIR, "ModelEvalution.webp"),
    "⚡ Live 24h Peak Forecaster": os.path.join(DATA_DIR, "LIVE24.avif"),
    "🔍 Feature Engineering & Importance": os.path.join(DATA_DIR, "Feature Engineering.avif")
}



b64_img = get_base64_image(os.path.join(DATA_DIR, "CapstoneOverview.webp"))
st.markdown(f"""
<style>
.stApp {{
    background-image: linear-gradient(rgba(11, 15, 25, 0.8), rgba(11, 15, 25, 0.8)), url("data:image/webp;base64,{b64_img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)






if artifacts is None:
    st.error("⚠️ Model artifacts not found! Please run model training script (`python src/train.py`) first to generate `.joblib` models.")
    st.stop()

models_dict = artifacts['models']
meta = artifacts['meta']
results = meta['results']
feature_cols = meta['feature_columns']

# ==========================================
# TAB 1: CAPSTONE OVERVIEW & ARCHITECTURE
# ==========================================
with tab1:
    st.header("📌 Capstone Project Overview")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Problem Context & Significance
        Electricity generation must equal consumption in real-time across regional grids. Sudden demand spikes (e.g. during heatwaves or extreme weather) can overwhelm substations, cause high peak-demand tariffs, or lead to catastrophic blackouts.
        
        **The AIML Solution**:
        Rather than relying on computationally heavy deep learning (LSTM/RNN) models, this capstone project proves that **Time-Series Lag Feature Engineering** ($t-1, t-2, t-24, t-168$, 24h rolling window metrics, and cyclic time encodings) allows supervised tree and linear models to achieve high accuracy ($R^2 > 0.95$) with sub-millisecond inference latency suitable for real-time grid control.
        
        ### 🎓 Academic Metadata
        - **Domain**: Smart Grid Logistics & Energy Management  
        - **Pipeline**: Time-Series Lag Engineering + Chronological ML  
        - **Evaluation Metrics**: RMSE, MAE, R², MAPE  
        - **Primary Models**: Ridge, Random Forest, HistGradientBoosting
        
        ### 👥 Team Members
        - **Pratik Sawant** (20240802324)
        - **Pranav Karne** (20240802356)
        - **Amar Nimbalkar** (20240802392)
        """)
        
    with col2:
        st.markdown("""
        <div style="background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155;">
            <h4 style="color: #38bdf8; margin-top: 0;">🎓 Capstone Highlights</h4>
            <ul>
                <li><b>Chronological Train-Test Split (80/20)</b> prevents data leakage across time boundaries.</li>
                <li><b>24 Feature Pipeline</b> combining lagged metrics, weather parameters, and cyclic encodings.</li>
                <li><b>Multi-Model Comparison</b>: Linear/Ridge Baseline vs. Random Forest vs. Gradient Boosting.</li>
                <li><b>Automated Grid Alert System</b>: Categorizes predicted demand into Normal, Warning, or Critical Peak Strain.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("🛠️ End-to-End System Architecture")
    
    st.markdown("""
    ```
    ┌─────────────────────────┐      ┌──────────────────────────────┐      ┌─────────────────────────────┐
    │  1-Year Hourly Dataset  │ ───► │  Lag Feature Engineering     │ ───► │ Chronological Split (80/20) │
    │  (8,760 observations)   │      │  (t-1, t-24, rolling mean)  │      │ (No Temporal Data Leakage)  │
    └─────────────────────────┘      └──────────────────────────────┘      └─────────────────────────────┘
                                                                                      │
                                                                                      ▼
    ┌─────────────────────────┐      ┌──────────────────────────────┐      ┌─────────────────────────────┐
    │ Streamlit Live Web App  │ ◄─── │ Model Export (.joblib)       │ ◄─── │ Benchmarking & Evaluation   │
    │ (5 Interactive Tabs)    │      │ (Ridge, Random Forest, GB)   │      │ (RMSE, MAE, R², MAPE)       │
    └─────────────────────────┘      └──────────────────────────────┘      └─────────────────────────────┘
    ```
    """)
    
    st.subheader("📈 Quick Model Benchmarks")
    cols = st.columns(len(results))
    for idx, (m_name, m_res) in enumerate(results.items()):
        test_m = m_res['test_metrics']
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">{m_name}</div>
                <div class="metric-value">{test_m['r2']:.4f} <span style="font-size: 1rem; color: #94a3b8;">R²</span></div>
                <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;">
                    RMSE: <b>{test_m['rmse']:.1f} kWh</b> | MAE: <b>{test_m['mae']:.1f} kWh</b><br>
                    MAPE: <b>{test_m['mape']:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: MODEL EVALUATION & METRICS
# ==========================================
with tab2:
    st.header("📊 Model Evaluation & Benchmarking")
    st.markdown("Detailed performance comparison across test dataset (20% chronological holdout split).")
    
    # 1. Metrics Comparison Table & Bar Charts
    metrics_data = []
    for m_name, m_res in results.items():
        tm = m_res['test_metrics']
        metrics_data.append({
            'Model': m_name,
            'RMSE (kWh)': round(tm['rmse'], 2),
            'MAE (kWh)': round(tm['mae'], 2),
            'R² Score': round(tm['r2'], 4),
            'MAPE (%)': round(tm['mape'], 2)
        })
    df_metrics = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📋 Evaluation Summary Table")
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
    with col2:
        st.subheader("📊 R² Score Comparison")
        fig_r2 = px.bar(
            df_metrics, x='Model', y='R² Score',
            color='Model', text='R² Score',
            color_discrete_sequence=['#38bdf8', '#818cf8', '#34d399']
        )
        fig_r2.update_layout(yaxis_range=[0.8, 1.0], showlegend=False, template="plotly_dark", height=280)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 168-Hour (1-Week) Actual vs. Predicted Test Forecast Overlay")
    
    # Overlay Chart
    sample_preds = results['Gradient Boosting']['sample_predictions']
    timestamps = sample_preds['timestamps']
    actual = sample_preds['actual']
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=timestamps, y=actual, mode='lines', name='Actual Load (kWh)', line=dict(color='#38bdf8', width=2)))
    
    colors = {'Naïve Baseline (t-24)': '#94a3b8', 'Ridge Regression': '#f59e0b', 'Random Forest': '#a855f7', 'Gradient Boosting': '#34d399'}
    for m_name, m_res in results.items():
        preds = m_res['sample_predictions']['predicted']
        fig_time.add_trace(go.Scatter(x=timestamps, y=preds, mode='lines', name=f'{m_name} Pred', line=dict(color=colors[m_name], width=1.5, dash='dash')))
        
    fig_time.add_hline(y=meta['peak_demand_threshold_kwh'], line_dash="dot", line_color="#ef4444", annotation_text="Peak Demand Warning (38,000 kWh)")
    fig_time.update_layout(
        xaxis_title="Timestamp (Hourly)",
        yaxis_title="Load (kWh)",
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("📉 Residual Error Analysis (Actual - Predicted)")
    res_cols = st.columns(len(results))
    for idx, (m_name, m_res) in enumerate(results.items()):
        act = np.array(m_res['sample_predictions']['actual'])
        prd = np.array(m_res['sample_predictions']['predicted'])
        residuals = act - prd
        
        fig_res = px.histogram(residuals, nbins=25, title=f"{m_name} Residuals", labels={'value': 'Error (kWh)'}, color_discrete_sequence=[colors[m_name]])
        fig_res.update_layout(template="plotly_dark", height=240, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        with res_cols[idx]:
            st.plotly_chart(fig_res, use_container_width=True)

# ==========================================
# TAB 3: LIVE 24-HOUR PEAK FORECASTER
# ==========================================
with tab3:
    st.header("⚡ Live Peak Load Predictor & Grid Warning System")
    st.markdown("Select model parameters, environmental conditions, and past lag inputs to forecast electricity load.")
    
    col_input, col_out = st.columns([5, 4])
    
    with col_input:
        st.subheader("🎛️ Input Parameters & Lag Features")
        
        model_choice = st.selectbox("Select Model Architecture:", list(models_dict.keys()), index=2)
        
        c1, c2 = st.columns(2)
        with c1:
            input_date = st.date_input("Forecast Date", datetime(2025, 7, 15))
            input_hour = st.slider("Hour of Day", 0, 23, 19)
            temp_c = st.slider("Ambient Temperature (°C)", -5.0, 48.0, 34.0, step=0.5)
            humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, 65.0, step=1.0)
        with c2:
            solar = st.slider("Solar Irradiance (W/m²)", 0.0, 1000.0, 350.0, step=10.0)
            wind = st.slider("Wind Speed (km/h)", 0.0, 60.0, 12.0, step=1.0)
            is_weekend = st.checkbox("Is Weekend?", value=False)
            is_holiday = st.checkbox("Is Public Holiday?", value=False)
            
        st.markdown("**Time Series Lag Memory (kWh)**")
        lc1, lc2 = st.columns(2)
        with lc1:
            lag_1 = st.number_input("Load at t-1 (Previous Hour)", value=32500.0, step=500.0)
            lag_2 = st.number_input("Load at t-2 (2 Hours Ago)", value=31000.0, step=500.0)
        with lc2:
            lag_24 = st.number_input("Load at t-24 (Yesterday Same Hour)", value=33000.0, step=500.0)
            lag_168 = st.number_input("Load at t-168 (Last Week Same Hour)", value=31500.0, step=500.0)
            
        roll_mean_6 = (lag_1 + lag_2) / 2.0
        roll_mean_24 = lag_24 * 0.95
        roll_std_24 = 1800.0

    with col_out:
        st.subheader("🔮 Forecast Inference Output")
        
        # Build feature vector dictionary matching FEATURE_COLUMNS
        dt = pd.to_datetime(input_date)
        month = dt.month
        dayofweek = dt.dayofweek
        dayofyear = dt.dayofyear

        feat_dict = {
            'temperature_c': temp_c,
            'humidity_pct': humidity,
            'solar_irradiance_wm2': solar,
            'wind_speed_kmh': wind,
            'is_weekend': int(is_weekend),
            'is_holiday': int(is_holiday),
            'hour': input_hour,
            'dayofweek': dayofweek,
            'month': month,
            'dayofyear': dayofyear,
            'sin_hour': np.sin(2 * np.pi * input_hour / 24.0),
            'cos_hour': np.cos(2 * np.pi * input_hour / 24.0),
            'sin_month': np.sin(2 * np.pi * month / 12.0),
            'cos_month': np.cos(2 * np.pi * month / 12.0),
            'temp_squared': temp_c ** 2,
            'temp_humidity_idx': temp_c * (humidity / 100.0),
            'is_extreme_temp': int((temp_c > 32) or (temp_c < 6)),
            'load_lag_1': lag_1,
            'load_lag_2': lag_2,
            'load_lag_24': lag_24,
            'load_lag_168': lag_168,
            'rolling_mean_6': roll_mean_6,
            'rolling_mean_24': roll_mean_24,
            'rolling_std_24': roll_std_24
        }
        
        input_df = pd.DataFrame([feat_dict])[feature_cols]
        selected_model = models_dict[model_choice]
        
        if model_choice == 'Ridge Regression':
            scaled_df = artifacts['scaler'].transform(input_df)
            pred_kwh = float(selected_model.predict(scaled_df)[0])
        else:
            pred_kwh = float(selected_model.predict(input_df)[0])
            
        max_capacity = meta['grid_capacity_max_kwh']
        peak_threshold = meta['peak_demand_threshold_kwh']
        load_pct = (pred_kwh / max_capacity) * 100.0
        
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 14px; padding: 24px; text-align: center; border: 1px solid #334155;">
            <div style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;">Predicted 1-Hour Electricity Load</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: #38bdf8; margin: 8px 0;">
                {pred_kwh:,.1f} <span style="font-size: 1.4rem;">kWh</span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.95rem;">Grid Load Utilization: <b>{load_pct:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🚨 Grid Strain & Peak Status")
        
        if pred_kwh >= peak_threshold:
            st.markdown(f"""
            <div class="status-critical">
                ⚠️ CRITICAL GRID PEAK STRAIN ALERT ({pred_kwh:,.0f} kWh)<br>
                Demand exceeds peak threshold ({peak_threshold:,.0f} kWh). Risk of blackout! Activate demand response & peaking units immediately.
            </div>
            """, unsafe_allow_html=True)
        elif pred_kwh >= peak_threshold * 0.85:
            st.markdown(f"""
            <div class="status-warning">
                ⚡ ELEVATED DEMAND WARNING ({pred_kwh:,.0f} kWh)<br>
                Grid load approaching capacity. Monitor regional HVAC consumption.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-normal">
                ✅ NORMAL OPERATING LOAD ({pred_kwh:,.0f} kWh)<br>
                Grid operating well within safety reserves.
            </div>
            """, unsafe_allow_html=True)

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_kwh,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Grid Load Level (kWh)"},
            gauge = {
                'axis': {'range': [0, max_capacity]},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0, peak_threshold * 0.85], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [peak_threshold * 0.85, peak_threshold], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [peak_threshold, max_capacity], 'color': "rgba(239, 68, 68, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': peak_threshold
                }
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# TAB 4: FEATURE ENGINEERING & IMPORTANCE
# ==========================================
with tab4:
    st.header("🔍 Feature Engineering & Predictive Power Analysis")
    st.markdown("Proving how domain-specific time-series feature engineering powers model decision-making.")
    
    model_sel = st.selectbox("Select Model Feature Importance:", list(models_dict.keys()), index=2)
    
    imp_dict = results[model_sel]['feature_importances']
    df_imp = pd.DataFrame(list(imp_dict.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_imp = px.bar(
            df_imp, x='Importance', y='Feature', orientation='h',
            title=f"Feature Importances ({model_sel})",
            color='Importance', color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(template="plotly_dark", height=540)
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col2:
        st.subheader("💡 Key Academic Insights")
        st.markdown("""
        - **Dominance of Lagged Memory**: `load_lag_1` (t-1) and `load_lag_24` (t-24 yesterday) account for **over 60% of predictive power**.
        - **Diurnal Cycles**: Cyclic sine/cosine features (`sin_hour`, `cos_hour`) capture smooth non-linear transitions without arbitrary step discontinuities.
        - **HVAC Non-Linearity**: `temp_squared` captures the exponential rise in electricity consumption during extreme summer heatwaves (>35°C).
        """)
        st.dataframe(df_imp.sort_values('Importance', ascending=False), use_container_width=True, hide_index=True)


# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b;'>⚡ Smart Energy Grid Forecaster | 3rd Year AIML Capstone Project</div>", unsafe_allow_html=True)
