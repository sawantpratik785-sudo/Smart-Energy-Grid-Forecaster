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
        margin-top: 10px !important;
        margin-bottom: 20px !important;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
    }
    
    .status-critical {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
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
    background-image: linear-gradient(rgba(11, 15, 25, 0.88), rgba(11, 15, 25, 0.88)), url("data:image/webp;base64,{b64_img}");
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
        ### Problem Context & Significance (Indian Smart Grid Context)
        In rapidly developing economies like India, localized electricity demand surges—intensified by high population density, heavy commercial/MNC hubs, and severe summer heatwaves—routinely overload neighborhood distribution transformers.
        
        This leads to catastrophic transformer blasts, short circuits, and unscheduled blackouts. Traditional approaches rely on reactive load shedding *after* damage has occurred.
        
        **The AIML Solution**:
        This project proves that **Time-Series Lag Feature Engineering** ($t-1, t-2, t-24, t-168$, 24h rolling stats, cyclic time encodings) combined with **Demographic & Zoning Indicators** (Population Density, MNC Commercial Hubs, Population-Temperature Index) allows supervised regression models to predict transformer strain in advance with sub-millisecond inference latency, enabling proactive, targeted load shedding to prevent transformer fires.
        
        ### 🎓 Academic Metadata
        - **Domain**: Smart Grid Logistics & Infrastructure Protection  
        - **Pipeline**: Time-Series Lag Engineering + Socio-Demographic Features + Chronological ML  
        - **Evaluation Metrics**: RMSE, MAE, R², MAPE  
        - **Primary Models**: Ridge Regression, Random Forest, Gradient Boosting
        
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
                <li><b>Multi-Zone Demographic Simulation</b>: Residential, Commercial/MNC, and Industrial sectors with realistic population scaling.</li>
                <li><b>Chronological Train-Test Split (80/20)</b>: Strictly prevents temporal data leakage across time boundaries.</li>
                <li><b>27 Feature Pipeline</b>: Combining memory lags, rolling stats, cyclic encodings, and socio-meteorological interaction features.</li>
                <li><b>Transformer Blast Risk & Load Shedding</b>: Automated calculation of exact kWh reduction needed to prevent transformer meltdown.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("🛠️ End-to-End System Architecture")
    
    st.markdown("""
    ```
    ┌──────────────────────────────────┐      ┌──────────────────────────────────┐      ┌─────────────────────────────┐
    │ Multi-Zone Grid & Weather Data   │ ───► │ Socio-Temporal Lag Engineering   │ ───► │ Chronological Split (80/20) │
    │ (26,280 obs: Pop, MNC, Weather)  │      │ (Lags, Pop-Temp Index, Rolling)  │      │ (Zero Temporal Leakage)     │
    └──────────────────────────────────┘      └──────────────────────────────────┘      └─────────────────────────────┘
                                                                                                       │
                                                                                                       ▼
    ┌──────────────────────────────────┐      ┌──────────────────────────────────┐      ┌─────────────────────────────┐
    │ Streamlit Live Web App           │ ◄─── │ Model Export (.joblib)           │ ◄─── │ Benchmarking & Evaluation   │
    │ (Transformer Blast Early Warning)│      │ (Ridge, Random Forest, GB)       │      │ (RMSE, MAE, R², MAPE)       │
    └──────────────────────────────────┘      └──────────────────────────────────┘      └─────────────────────────────┘
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
        fig_r2.update_layout(yaxis_range=[0.0, 1.0], showlegend=False, template="plotly_dark", height=280)
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
        
    peak_thresh = meta.get('peak_demand_threshold_kwh', 3500.0)
    fig_time.add_hline(y=peak_thresh, line_dash="dot", line_color="#ef4444", annotation_text=f"Peak Demand Warning ({peak_thresh:,.0f} kWh)")
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
    
    # 1-Click Indian Grid Scenario Presets
    st.markdown("**⚡ Quick Indian Grid Demonstration Presets**")
    p1, p2, p3 = st.columns(3)
    
    if 'preset_hour' not in st.session_state:
        st.session_state.preset_hour = 19
        st.session_state.preset_temp = 34.0
        st.session_state.preset_humidity = 65.0
        st.session_state.preset_pop = 25000
        st.session_state.preset_cap = 2500.0
        st.session_state.preset_zone = "Residential"
        st.session_state.preset_weekend = False
        st.session_state.preset_lag1 = 1200.0
        st.session_state.preset_lag24 = 1250.0

    with p1:
        if st.button("🔥 May Heatwave Peak (42°C, 8 PM)", use_container_width=True):
            st.session_state.preset_hour = 20
            st.session_state.preset_temp = 42.0
            st.session_state.preset_humidity = 40.0
            st.session_state.preset_pop = 52000
            st.session_state.preset_cap = 2500.0
            st.session_state.preset_zone = "Residential"
            st.session_state.preset_weekend = False
            st.session_state.preset_lag1 = 2600.0
            st.session_state.preset_lag24 = 2650.0
            st.rerun()

    with p2:
        if st.button("🏢 Cyber City MNC Afternoon (2 PM)", use_container_width=True):
            st.session_state.preset_hour = 14
            st.session_state.preset_temp = 35.0
            st.session_state.preset_humidity = 55.0
            st.session_state.preset_pop = 15000
            st.session_state.preset_cap = 3500.0
            st.session_state.preset_zone = "Commercial / MNC Hub"
            st.session_state.preset_weekend = False
            st.session_state.preset_lag1 = 2800.0
            st.session_state.preset_lag24 = 2750.0
            st.rerun()

    with p3:
        if st.button("🌱 Monsoon Normal Evening (25°C, 7 PM)", use_container_width=True):
            st.session_state.preset_hour = 19
            st.session_state.preset_temp = 25.0
            st.session_state.preset_humidity = 82.0
            st.session_state.preset_pop = 22000
            st.session_state.preset_cap = 2500.0
            st.session_state.preset_zone = "Residential"
            st.session_state.preset_weekend = True
            st.session_state.preset_lag1 = 1100.0
            st.session_state.preset_lag24 = 1150.0
            st.rerun()

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    
    col_input, col_out = st.columns([5, 4])
    
    with col_input:
        st.subheader("🎛️ Input Parameters & Lag Features")
        
        model_choice = st.selectbox("Select Model Architecture:", list(models_dict.keys()), index=2)
        
        c1, c2 = st.columns(2)
        with c1:
            input_date = st.date_input("Forecast Date", datetime(2025, 7, 15))
            input_hour = st.slider("Hour of Day", 0, 23, st.session_state.preset_hour)
            temp_c = st.slider("Ambient Temperature (°C)", -5.0, 48.0, st.session_state.preset_temp, step=0.5)
            humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, st.session_state.preset_humidity, step=1.0)
        with c2:
            solar = st.slider("Solar Irradiance (W/m²)", 0.0, 1000.0, 350.0, step=10.0)
            wind = st.slider("Wind Speed (km/h)", 0.0, 60.0, 12.0, step=1.0)
            is_weekend = st.checkbox("Is Weekend?", value=st.session_state.preset_weekend)
            is_holiday = st.checkbox("Is Public Holiday?", value=False)
            
        st.markdown("**Demographics & Local Grid (Indian Context)**")
        dc1, dc2 = st.columns(2)
        with dc1:
            population = st.slider("Zone Population", 1000, 100000, st.session_state.preset_pop, step=1000)
            transformer_cap = st.slider("Transformer Capacity (kWh)", 1000.0, 8000.0, st.session_state.preset_cap, step=100.0)
        with dc2:
            zone_idx = 1 if st.session_state.preset_zone == "Commercial / MNC Hub" else 0
            zone_type = st.radio("Zone Type", ["Residential", "Commercial / MNC Hub"], index=zone_idx)
            is_mnc = 1 if zone_type == "Commercial / MNC Hub" else 0
            
        st.markdown("**Time Series Lag Memory (kWh)**")
        lc1, lc2 = st.columns(2)
        with lc1:
            lag_1 = st.number_input("Load at t-1 (Previous Hour)", value=st.session_state.preset_lag1, step=100.0)
            lag_2 = st.number_input("Load at t-2 (2 Hours Ago)", value=max(500.0, st.session_state.preset_lag1 - 50.0), step=100.0)
        with lc2:
            lag_24 = st.number_input("Load at t-24 (Yesterday Same Hour)", value=st.session_state.preset_lag24, step=100.0)
            lag_168 = st.number_input("Load at t-168 (Last Week Same Hour)", value=max(500.0, st.session_state.preset_lag24 - 50.0), step=100.0)
            
        roll_mean_6 = (lag_1 + lag_2) / 2.0
        roll_mean_24 = lag_24 * 0.95
        roll_std_24 = 150.0

    with col_out:
        st.subheader("🔮 Forecast Inference Output")
        
        # Build feature vector dictionary matching FEATURE_COLUMNS
        dt = pd.to_datetime(input_date)
        month = dt.month
        dayofweek = dt.dayofweek
        dayofyear = dt.dayofyear

        feat_dict = {
            'population': population,
            'is_mnc_zone': is_mnc,
            'transformer_capacity': transformer_cap,
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
            'is_extreme_temp': int((temp_c > 35) or (temp_c < 10)),
            'pop_temp_idx': population * temp_c / 10000.0,
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
            
        max_capacity = transformer_cap
        peak_threshold = max_capacity * 0.90
        load_pct = (pred_kwh / max_capacity) * 100.0
        
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 14px; padding: 24px; text-align: center; border: 1px solid #334155;">
            <div style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;">Predicted 1-Hour Electricity Load</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: #38bdf8; margin: 8px 0;">
                {pred_kwh:,.1f} <span style="font-size: 1.4rem;">kWh</span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.95rem;">Transformer Load Utilization: <b>{load_pct:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🚨 Transformer Strain & Load Balancing")
        
        if pred_kwh >= max_capacity:
            shortage = pred_kwh - max_capacity
            st.markdown(f"""
            <div class="status-critical">
                💥 CRITICAL: TRANSFORMER BLAST RISK ({pred_kwh:,.0f} kWh)<br>
                Demand exceeds physical capacity of {max_capacity:,.0f} kWh! Risk of short circuit or explosion.<br>
                <b>ACTION REQUIRED: Initiate targeted Load Shedding of at least {shortage:,.0f} kWh to balance supply equitably.</b>
            </div>
            """, unsafe_allow_html=True)
        elif pred_kwh >= max_capacity * 0.90:
            st.markdown(f"""
            <div class="status-warning">
                ⚡ OVERLOAD WARNING ({pred_kwh:,.0f} kWh)<br>
                Transformer operating at over 90% capacity. High risk of localized voltage drops and heating. Monitor {zone_type} usage.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-normal">
                ✅ NORMAL OPERATING LOAD ({pred_kwh:,.0f} kWh)<br>
                Grid operating well within safety reserves.
            </div>
            """, unsafe_allow_html=True)

        # Gauge Chart with ample margin and crisp font formatting to prevent overlap
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        gauge_max = round(max(max_capacity * 1.25, pred_kwh * 1.15), -1)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_kwh,
            number = {
                'font': {'size': 26, 'color': '#38bdf8', 'family': 'Space Grotesk, sans-serif'},
                'suffix': ' kWh',
                'valueformat': ',.1f'
            },
            domain = {'x': [0.05, 0.95], 'y': [0.0, 0.78]},
            title = {
                'text': "<b>Transformer Load vs Blast Limit</b>",
                'font': {'size': 14, 'color': '#94a3b8'}
            },
            gauge = {
                'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': '#64748b'},
                'bar': {'color': "#38bdf8", 'thickness': 0.28},
                'bgcolor': "rgba(15, 23, 42, 0.4)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, max_capacity * 0.85], 'color': "rgba(16, 185, 129, 0.25)"},
                    {'range': [max_capacity * 0.85, max_capacity], 'color': "rgba(245, 158, 11, 0.25)"},
                    {'range': [max_capacity, gauge_max], 'color': "rgba(239, 68, 68, 0.35)"}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 4},
                    'thickness': 0.8,
                    'value': max_capacity
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=30, r=30, t=55, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature 1: Transformer Thermal Health & Accelerated Aging Rate (IEEE C57 Standard Principle)
        load_ratio = pred_kwh / max_capacity
        if load_ratio <= 0.75 and temp_c <= 32:
            aging_factor = round(0.5 + 0.5 * load_ratio, 1)
            health_badge = "OPTIMAL"
            health_desc = "Normal winding temperature (~65°C). Minimal thermal wear."
            health_color = "#34d399"
            badge_bg = "rgba(16, 185, 129, 0.2)"
        elif load_ratio <= 0.95:
            temp_penalty = max(0.0, (temp_c - 30.0) * 0.15)
            aging_factor = round(1.0 + (load_ratio - 0.75) * 8.0 + temp_penalty, 1)
            health_badge = "MODERATE WEAR"
            health_desc = "Accelerated oil insulation heating (~85°C). Maintenance window shortens."
            health_color = "#fbbf24"
            badge_bg = "rgba(245, 158, 11, 0.2)"
        else:
            temp_penalty = max(0.0, (temp_c - 30.0) * 0.35)
            aging_factor = round(4.0 + (load_ratio - 0.95) * 18.0 + temp_penalty, 1)
            health_badge = "CRITICAL THERMAL STRESS"
            health_desc = "Extreme coil winding breakdown risk (~115°C+). Oil breakdown imminent!"
            health_color = "#f87171"
            badge_bg = "rgba(239, 68, 68, 0.2)"

        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 12px; padding: 18px; border: 1px solid #334155; margin-top: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: 700; color: #cbd5e1; font-size: 0.95rem;">
                    🌡️ Transformer Health & Accelerated Aging Rate
                </div>
                <div style="background: {badge_bg}; color: {health_color}; border: 1px solid {health_color}; font-weight: 800; padding: 4px 10px; border-radius: 8px; font-size: 0.85rem;">
                    {aging_factor:.1f}x Wear Rate ({health_badge})
                </div>
            </div>
            <div style="color: {health_color}; font-size: 0.85rem; margin-top: 8px; font-weight: 600;">
                {health_desc}
            </div>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 6px; line-height: 1.4;">
                <b>Predictive Maintenance Insight:</b> At current load ({load_pct:.1f}%) and ambient temperature ({temp_c}°C), transformer insulation degrades <b>{aging_factor:.1f} times faster</b> than manufacturer baseline.
            </div>
        </div>
        """, unsafe_allow_html=True)

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
        - **Dominance of Lagged Memory**: `load_lag_1` (t-1) and `load_lag_24` (t-24 yesterday) account for strong short-term inertia and diurnal patterns.
        - **Socio-Demographic Scaling**: `population`, `is_mnc_zone`, and `pop_temp_idx` empower the model to differentiate between residential vs. high-intensity commercial peak hours.
        - **Diurnal Cycles**: Cyclic sine/cosine features (`sin_hour`, `cos_hour`) capture smooth 24-hour transitions without arbitrary boundary discontinuities.
        - **HVAC Non-Linearity**: `temp_squared` captures the exponential spike in electricity demand during extreme summer heatwaves (>35°C).
        """)
        st.dataframe(df_imp.sort_values('Importance', ascending=False), use_container_width=True, hide_index=True)


# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b;'>⚡ Smart Energy Grid Forecaster | 3rd Year AIML Capstone Project</div>", unsafe_allow_html=True)
