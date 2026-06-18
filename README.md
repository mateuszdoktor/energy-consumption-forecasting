<div align="center">

# Hourly Energy Consumption Forecasting

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14+-green.svg)](https://www.statsmodels.org/)
[![Pmdarima](https://img.shields.io/badge/Pmdarima-2.0+-orange.svg)](https://alkaline-ml.com/pmdarima/)

_Hourly electricity load forecasting on PJM East data_

</div>

## About The Project

This repository is an applied machine learning project focused on **time series forecasting**.
The objective is to predict hourly energy consumption for the **PJM East (PJME)** region and compare statistical, gradient-boosting, and deep learning approaches under a consistent evaluation setup.

Beyond model training, the project demonstrates an end-to-end and reproducible workflow:

- exploratory analysis in `notebooks/01_eda.ipynb`
- feature engineering and chronological train/validation/test splits in `notebooks/02_preprocessing.ipynb`
- DHR + ARIMAX modeling and single-origin evaluation in `notebooks/03_dhr_arimax.ipynb`
- reusable preprocessing and modeling utilities in `src/`

### Dataset

The project uses hourly load data from **PJM Interconnection**:

- **Primary series:** `data/raw/PJME_hourly.csv` — PJM East (`PJME_MW`), Jan 2002 – Aug 2018
- **Secondary series (available):** `data/raw/PJMW_hourly.csv` — PJM West

The target variable is hourly energy consumption in **MW** (`energy` after preprocessing).

The preprocessing pipeline includes:

- datetime parsing and hourly reindexing
- linear interpolation of missing load values
- calendar features (`hour`, `dayofweek`, `month`, US holidays, weekends)
- weather features from Meteostat (cooling/heating degree hours across 5 cities)
- Fourier terms for multiple seasonal periods (daily, weekly, annual)
- lag and rolling features for ML models
- chronological split:
  - **train:** before 2016
  - **validation:** 2016
  - **test:** from 2017 onward

Processed splits are exported to `data/processed/`.

### Implemented Model: DHR + ARIMAX

The first forecasting baseline combines:

- **Dynamic Harmonic Regression (DHR):** Fourier terms for periods `[24, 168, 4406, 8766]` hours, plus calendar and weather regressors
- **ARIMAX:** autoregressive component fitted on the differenced series (`d=1`) with exogenous DHR features

Training and inference are handled via `pmdarima.auto_arima` (order search on the train split) and cached in `models/arimax_dhr.joblib`.

Evaluation uses a **single-origin** setup: one multi-step forecast from the end of train, compared against a **seasonal naive** baseline that uses only train history (lag aligned with forecast horizon).

### Key Highlights

- **Multi-Seasonality Handling:** Fourier harmonics for intraday, weekly, and annual patterns.
- **Exogenous Features:** US holidays, weekends, and regional temperature-driven degree-hour features.
- **Leakage-Safe Splits:** Strict chronological partitioning with features computed before splitting where required.
- **Reusable Pipeline:** Shared utilities for preprocessing, MSTL decomposition, metrics, and model caching.
- **Fair Baseline Comparison:** Seasonal naive restricted to information available at forecast origin.

### Planned Models

The current DHR + ARIMAX setup is the first completed forecasting baseline. Planned extensions under the same data and evaluation workflow:

- **Prophet / TBATS** — classical decomposition and Bayesian structural time series baselines
- **XGBoost / LightGBM** — gradient boosting on engineered lag, rolling, calendar, and weather features
- **LSTM** — sequence-based deep learning model for multi-step hourly forecasting

---

## Tech Stack

- **Time Series / Statistics:** Statsmodels, Pmdarima
- **Data Processing:** NumPy, Pandas, Holidays, Meteostat
- **Visualization:** Matplotlib, Seaborn
- **Notebooks:** Jupyter
