# ⚡ Smart Energy Grid Forecaster
### 🏆 3rd Year AIML Capstone Project: Transformer Overload Prediction & Automated Load Balancing

---

## 📌 Project Overview & Significance
In rapidly developing power networks (such as India's regional grids), localized electricity demand surges—intensified by dense residential populations, commercial/MNC hubs, and severe summer heatwaves—routinely exceed physical transformer capacity. This triggers catastrophic transformer fires, short circuits, and unscheduled blackouts.

Traditional distribution management is reactive: power is cut *after* a transformer trips or fails. 

**Our AIML Solution**:
We built an end-to-end Machine Learning pipeline using **Time-Series Lag Feature Engineering** coupled with **Socio-Demographic & Zoning Indicators** (Population Density, MNC commercial zones, Population-Temperature interactions). Supervised regression models (**Ridge Regression**, **Random Forest**, and **HistGradientBoosting**) predict short-term localized demand with sub-millisecond inference latency, enabling proactive early warnings and automated, targeted **Load Shedding recommendations** before physical equipment damage occurs.

---

## 🏗️ System Architecture & Workflow
```
[ 3-Zone Synthetic Grid Data (26,280 obs: Residential, MNC Hub, Industrial) ]
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
4. **Lightweight Edge Inference**: Sub-millisecond prediction latency suitable for deployment on low-cost SCADA / IoT substation controllers without requiring heavy GPUs.

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
| Model Architecture | Test RMSE (kWh) | Test MAE (kWh) | Test R² Score | Test MAPE (%) | Est. Annual Penalty ($) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naïve Baseline ($t-24$)** | 687.05 | 447.11 | -0.1396 | 21.79% | $313,337 |
| **Ridge Regression** | 340.61 | 207.73 | 0.7199 | 9.25% | $145,579 |
| **Random Forest Regressor** | 294.62 | 155.63 | 0.7904 | 6.70% | $109,062 |
| **Gradient Boosting (Champion)** | **284.81** | **146.13** | **0.8042** | **6.22%** | **$102,406** |

> 💰 **Financial Impact**: The Gradient Boosting model delivers **$210,930 / year in cost savings** over the naïve rule and **$43,172 / year** over linear ridge regression.

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
