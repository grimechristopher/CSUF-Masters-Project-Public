# Aerospace Component Demand Forecasting with LSTM

M.S. Computer Science capstone project — California State University, Fullerton (2025)

Predicts weekly maintenance demand for aerospace components using a deep learning pipeline built on historical repair records, fleet composition, and flight operational data. The model achieves **94.5% precision** on binary maintenance event prediction (will a part need repair in the next week?), enabling smarter inventory positioning and reducing both stockouts and overstock.

---

## Problem

Aerospace MRO (Maintenance, Repair & Overhaul) operations carry large inventories of line-replaceable units (LRUs) to avoid aircraft-on-ground situations. Demand is irregular — some parts go months without failures, then spike. Traditional min/max reorder rules overstock low-demand parts and still miss sudden demand spikes.

This project frames demand forecasting as a time-series prediction problem: given 8 weeks of history for a part across the active fleet, predict the number of maintenance events in the following week.

---

## Approach

### Data Pipeline (7 notebooks)

| Notebook | Description |
|---|---|
| `1_Extract.ipynb` | Pulls raw data from multiple sources via SQL and REST APIs — equipment installations, RMA repair orders, product catalog, MTBF records, flight events, and passenger counts |
| `2_Transform.ipynb` | Joins and normalizes sources into a unified part-level weekly time series |
| `3_Preprocess.ipynb` | Cleans outliers, handles sparse parts, aligns fleet counts to repair windows |
| `4_Feature_Eng.ipynb` | Engineers lag features, rolling statistics, cyclical time encoding, and maintenance rate |
| `5_Training.ipynb` | Trains the LSTM model on time-based train/val/test splits |
| `6_Evaluation.ipynb` | Evaluates model performance with regression and binary classification metrics |
| `7_Example.ipynb` | Demonstrates predictions on held-out parts |

### Feature Engineering

- **Lag features**: `ReceivedCount_lag1w`, 4-week and 12-week rolling means and standard deviations
- **Fleet context**: `RunningFleetTotal`, `InService`, `ShippedCount`
- **Cyclical time encoding**: `month_sin`, `month_cos`, `quarter_sin`, `quarter_cos` — prevents the model treating Dec→Jan as a large discontinuity
- **Maintenance rate**: `ReceivedCount / RunningFleetTotal` — normalizes demand by fleet size
- **Target transformation**: log(x+1) applied to `ReceivedCount` for training stability; inverse-transformed for evaluation

### Model Architecture

LSTM network implemented in TensorFlow/Keras:

```
LSTM(32, return_sequences=False)   ← 8-week lookback window
BatchNormalization + Dropout(0.3)
Dense(16, relu) + L2 regularization
BatchNormalization + Dropout(0.2)
Dense(1, linear)                   ← predicted maintenance event count
```

**Training details:**
- Optimizer: Adam (lr=0.0005, gradient clipping clipnorm=0.5)
- Loss: MSE on log-transformed target
- Normalization: RobustScaler (handles skewed, outlier-heavy distributions)
- Splits: 70% train / 15% val / 15% test — time-based, not random
- Callbacks: EarlyStopping (patience=15), ReduceLROnPlateau (patience=5), ModelCheckpoint
- Filtered to parts with ≥ 10 historical maintenance events

---

## Results

| Metric | Value |
|---|---|
| Precision | **94.5%** |
| Binary classification task | Will this part have ≥1 maintenance event next week? |
| Target metric | Precision (minimizes false "no maintenance needed" predictions that cause stockouts) |

Optimized for precision over recall: in MRO contexts, a false negative (missed demand) grounds aircraft; a false positive (unnecessary stock) is costly but manageable.

**Visualizations included:**

| File | Description |
|---|---|
| `prediction_accuracy.png` | Actual vs. predicted maintenance event counts |
| `weekly_parts_forecast.png` | Weekly demand forecast across the fleet |
| `inventory_comparison.png` | Model-driven inventory levels vs. baseline |
| `inventory_reduction.png` | Reduction in excess inventory from demand-guided stocking |
| `stockout_risk_reduction.png` | Reduction in stockout events vs. min/max rules |

---

## Stack

- **Deep learning**: TensorFlow 2.19, Keras 3.9
- **Data**: pandas, NumPy, PyArrow (Parquet), psycopg2, pyodbc
- **ML utilities**: scikit-learn (RobustScaler, LinearRegression), SciPy (KDE)
- **Visualization**: Matplotlib, Seaborn
- **Auth**: MSAL (Microsoft identity platform for REST API access)

---

## Setup

```bash
pip install -r requirements.txt
```

Database credentials and API endpoints are loaded from a `.env` file. See `.env.example` for required variables. Source data is stored in `private/data/` (excluded from this repo).

Run notebooks in order: `1_Extract` → `2_Transform` → `3_Preprocess` → `4_Feature_Eng` → `5_Training` → `6_Evaluation`.

---

## Notes

- Source data is proprietary and not included in this repository
- Sensitive identifiers (customer names, airline codes, serial numbers) have been anonymized in the extraction notebooks
- The `private/` directory is gitignored; all data files remain local
