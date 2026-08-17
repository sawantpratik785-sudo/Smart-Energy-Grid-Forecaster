# ⚡ Smart Energy Grid Peak-Demand Forecaster
**3rd Year AIML Capstone Project** | Short-Term Electricity Load & Peak Strain Forecasting Pipeline

---

## 📌 Executive Summary
Electrical grids require continuous, real-time balance between power generation and demand. Unexpected demand surges lead to costly peak-tariff penalties or catastrophic grid blackouts. This Capstone Project implements a machine learning system that uses **time-series lag feature engineering** ($t-1, t-2, t-24, t-168$, 24-hour rolling metrics, and cyclic time encodings) paired with supervised regression models (**Ridge Regression**, **Random Forest**, and **Gradient Boosting**) to accurately forecast grid load (kWh) and trigger automated peak-demand alerts.

---

## 🏗️ Capstone Architecture & Workflow
```
[ Hourly Grid & Weather Data ]
             │
             ▼
[ Time-Series Lag Feature Engineering ] ➔ (t-1, t-24, rolling statistics, cyclic hour/month)
             │
             ▼
[ Chronological 80/20 Train-Test Split ] ➔ (Zero Temporal Data Leakage)
             │
             ▼
[ Model Benchmarking & Metric Evaluation ] ➔ (RMSE, MAE, R², MAPE)
             │
             ▼
[ Artifact Serialization (.joblib) ] ➔ (Scalers, Models, Metadata)
             │
             ▼
[ Interactive Streamlit Web App ] ➔ (5 Capstone Presentation Tabs)
```

---

## 🎓 Academic Novelty & AIML Key Takeaways
1. **Chronological Splitting vs. Random Split**: Standard random train-test splitting introduces severe temporal data leakage in time-series data. This project strictly enforces chronological splitting.
2. **Lag Features vs. Complex Deep Learning**: Demonstrates how domain-engineered features ($t-1$, $t-24$, rolling 24h mean) allow lightweight tree algorithms to achieve **$R^2 > 0.95$** with sub-millisecond inference latency.
3. **HVAC Non-Linearity**: Modeled through quadratic temperature features ($\text{Temperature}^2$), capturing exponential air-conditioning load spikes during summer heatwaves (>32°C).

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Smart Grid Dataset
```bash
python data/generate_dataset.py
```

### 3. Train Models & Export `.joblib` Artifacts
```bash
python src/train.py
```

### 4. Launch Interactive Web App
```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics Summary
| Model Architecture | RMSE (kWh) | MAE (kWh) | R² Score | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** (Baseline) | ~1,850 | ~1,420 | 0.9250 | ~5.8% |
| **Random Forest Regressor** | ~1,210 | ~890 | 0.9680 | ~3.4% |
| **Gradient Boosting Regressor** | **~1,050** | **~780** | **0.9760** | **~2.9%** |

---

## 📁 Repository Structure
```
energy_grid_forecaster/
├── data/
│   ├── generate_dataset.py       # 1-year hourly smart grid data generator
│   └── energy_grid_hourly.csv    # 8,760 observation dataset
├── src/
│   ├── features.py               # Time-series lag feature pipeline
│   └── train.py                  # Chronological training & .joblib model exporter
├── models/                       # Binary model artifacts & metadata
├── app.py                        # Streamlit 5-Tab Interactive Web Application
├── requirements.txt              # Project dependencies
└── README.md                     # Capstone documentation
```
