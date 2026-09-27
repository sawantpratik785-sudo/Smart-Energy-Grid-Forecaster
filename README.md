# ⚡ Smart Energy Grid Forecaster
### 🏆 3rd Year AIML Capstone Project: Transformer Overload Prediction & Automated Load Balancing

---

## 📌 Project Overview & Significance
In rapidly developing power networks (such as India's regional grids), localized electricity demand surges—intensified by dense residential populations, commercial/MNC hubs, and severe summer heatwaves—routinely exceed physical transformer capacity. This triggers catastrophic transformer fires, short circuits, and unscheduled blackouts.

Traditional distribution management is reactive: power is cut *after* a transformer trips or fails. 

**Our AIML Solution**:
We built an end-to-end Machine Learning pipeline using **Time-Series Lag Feature Engineering** coupled with **Socio-Demographic & Zoning Indicators** (Population Density, MNC commercial zones, Population-Temperature interactions). Supervised regression models (**Ridge Regression**, **Random Forest**, **XGBoost**, and **HistGradientBoosting**) predict short-term localized demand with single-digit millisecond latency (sub-millisecond 0.10 ms for Ridge, 6.6 ms for XGBoost, 7.1 ms for HistGradientBoosting), enabling proactive early warnings and automated, targeted **Load Shedding recommendations** before physical equipment damage occurs.

---

## 🏗️ System Architecture & Workflow
```
[ 3-Zone Grid Data (26,280 obs: Residential, MNC Hub, Industrial) ]
                                    │
                                    ▼
[ Socio-Temporal Feature Engineering ] ➔ (t-1, t-24, rolling statistics, Pop-Temp Index)
                                    │
                                    ▼
[ Chronological 80/20 Train-Test Split ] ➔ (Strictly Prevents Temporal Data Leakage)
                                    │
                                    ▼
[ Multi-Model Benchmarking & Financial Impact ] ➔ (RMSE, MAE, R², MAPE, Annual Cost Savings)
                                    │
                                    ▼
[ Artifact Serialization (.joblib) ] ➔ (Fitted Scalers, Models, Pipeline Metadata)
                                    │
                                    ▼
[ Streamlit Real-Time Decision Support Dashboard ] ➔ (Blast Risk Warnings & Load Shedding UI)
```

---

## 🎓 Academic Novelty & AIML Key Highlights
1. **Chronological Splitting vs. Random Shuffle**: Standard random splitting in time series causes severe temporal data leakage. We enforce strict chronological partitioning (first 80% train, last 20% unseen test).
2. **Socio-Demographic Feature Injection**: Beyond pure time and weather, incorporating `population`, `is_mnc_zone`, and `pop_temp_idx` allows tree ensembles to capture differing peak schedules (daytime commercial vs. evening residential).
3. **Actionable Decision Support**: Rather than outputting raw kWh numbers alone, the pipeline computes the gap between predicted demand and transformer capacity, prescribing exact targeted load shedding amounts to prevent transformer fires.
4. **Lightweight Edge Inference**: Real-time single-digit millisecond latency suitable for deployment on low-cost SCADA / IoT substation controllers without requiring heavy GPUs (0.10 ms for Ridge, 6.6 ms for XGBoost, 7.1 ms for HistGradientBoosting).

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Multi-Zone Smart Grid Dataset
```bash
python data/generate_dataset.py
```

### 3. Train Models & Export `.joblib` Artifacts
```bash
python src/train.py
```

### 4. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
*(Runs locally at `http://localhost:8501`)*

---

## 📊 Evaluation Metrics Summary (Test Set)
| Model Architecture | Test RMSE (kWh) | Test MAE (kWh) | Test R² Score | Test MAPE (%) | Overload Recall | Overload Precision | Est. Annual Penalty ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naïve Baseline ($t-24$)** | 494.00 | 303.19 | 0.4386 | 16.71% | 85.6% | 85.6% | $212,475 |
| **Holt-Winters Statistical** | 592.93 | 431.05 | 0.1913 | 23.40% | 0.0% | 0.0% | $302,080 |
| **Ridge Regression** | 307.19 | 196.71 | 0.7829 | 11.15% | 70.3% | 98.1% | $137,854 |
| **Random Forest Regressor** | 242.39 | 129.81 | 0.8648 | 7.02% | 87.6% | 95.4% | $90,971 |
| **HistGradientBoosting (🏆)** | **234.00** | **122.56** | **0.8740** | **6.74%** | **85.4%** | **98.7%** | **$85,892** |
| **XGBoost Regressor** | 238.10 | 123.50 | 0.8696 | 6.65% | 82.0% | 98.9% | $86,549 |

> 💰 **Financial Impact (CERC DSM Basis)**: Champion HistGradientBoosting delivers **$126,582.52 / year in operational savings** over the naïve rule per substation feeder at the official $0.08/kWh deviation penalty rate.
> 
> 🌐 **Empirical Utility Benchmark (Real PJM & ERA5 Weather)**: Achieves **77.14% error reduction** (110.69 kWh vs. Naïve 484.10 kWh) with an **$R^2$ of 0.9858** and **90.16% overload recall** across 8,614 hours.

---

## 📁 Repository Structure
```
energy_grid_forecaster/
├── data/
│   ├── generate_dataset.py       # Multi-zone demographic & meteorological generator
│   └── energy_grid_hourly.csv    # 26,280 observation dataset (3 zones)
├── src/
│   ├── features.py               # Socio-temporal lag engineering module
│   └── train.py                  # Chronological training & .joblib exporter
├── models/                       # Serialized models (.joblib) and metadata JSON
├── app.py                        # Streamlit 4-Tab Decision Support Web Application
├── style.css                     # Custom glassmorphic stylesheet & animations
├── requirements.txt              # Project dependencies
├── CAPSTONE_FINAL_PRESENTATION.md# Complete presentation & viva blueprint
├── CAPSTONE_EXPLAINER.md         # Plain-English viva Q&A guide
└── README.md                     # Master project documentation
```

---

## 👥 Project Contributors
- **Pratik Sawant** (20240802324)
- **Pranav Karne** (20240802356)
- **Amar Nimbalkar** (20240802392)
