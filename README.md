# Temporary README — GNSS Clock & Ephemeris Error Prediction

> **Project:** Predicting time-varying errors between uploaded (broadcast) and modeled satellite clock & ephemeris values
>
> **Status:** Temporary README — use as a working project summary and bootstrap for the repository.

---

## 1. Project Overview

This project develops AI/ML models to predict the time-varying differences (errors) between the **uploaded/broadcast** and **ICD-based modeled** values of GNSS satellite clock biases and ephemeris (orbit) parameters. The objective is to predict these errors at 15-minute intervals for an unseen 8th day, using the provided seven-day training dataset. Improvements in these predictions increase the accuracy and reliability of GNSS positioning and timing.

## 2. Dataset

* **Provided:** 7 days of timestamped error measurements (clock biases and ephemeris parameter differences) for GNSS satellites in GEO/GSO and MEO.
* **Prediction target:** Errors for day 8, at 15-minute resolution.
* **Submission/evaluation timescales:** predictions evaluated at horizons of 15 min, 30 min, 1 hr, 2 hr, ... up to 24 hr from the last observed point.

> Note: this README assumes the dataset directory is `data/` and that each satellite has a CSV time series or a unified timeseries file.

## 3. Goals and Success Criteria

* Produce accurate forecasts of clock and ephemeris errors at 15-minute intervals for the 8th (unseen) day.
* Provide probabilistic or interval estimates (uncertainty) where possible.
* Residual error distribution should be close to normal (bell-shaped) — a metric used for final evaluation.
* Demonstrate robust performance across short (15–30 min) and long horizons (hours to 24 hr).

## 4. Approaches to Try

Suggested model classes (not exhaustive):

* **Sequence models:** LSTM, GRU, stacked/bi-directional variants.
* **Transformer-based models:** for longer-range dependency capture.
* **Generative models:** GANs for realistic synthetic sequences (useful for data augmentation or scenario generation).
* **Probabilistic models:** Gaussian Processes for uncertainty estimates and smooth interpolation.
* **Ensembles:** Combine models to improve robustness and calibrated uncertainty.

## 5. Baselines

Always implement and report against simple baselines:

* **Persistence (naive):** predict last observed value.
* **Moving average:** short-window average.
* **Linear autoregressive models (ARIMA / SARIMAX)** for quick benchmarks.

Beating these baselines is the first requirement.

## 6. Data preprocessing / feature ideas

* Resample and align timestamps to **15-minute** intervals.
* Handle missing values (interpolation, forward-fill, or mask-based models).
* Add time features: time-of-day, day-of-week, satellite type (GEO/MEO), leap seconds flags, etc.
* Compute short-term derivatives/rolling stats: recent slope, rolling mean/std for multiple windows (15m, 1h, 6h).
* If available, include satellite health flags, age of ephemeris, or other meta-data as features.

## 7. Training & Evaluation protocol

* **Train / Validation split:** use a sliding-window or block-validation across the 7 days (e.g., last 12–24 hours as validation) to mimic prediction on day 8.
* **Metrics:** MAE and RMSE per horizon (15m, 30m, 1h, 2h, ... 24h). Track mean & std across satellites.
* **Residual analysis:** evaluate distribution of residuals; compute skewness and kurtosis and perform a normality test (e.g., Shapiro–Wilk or Kolmogorov–Smirnov) as part of diagnostics.
* **Calibration / Uncertainty:** if producing predictive intervals, evaluate interval coverage (e.g., 95% interval coverage).

## 8. Suggested evaluation table (example)

|  Horizon | MAE (s) | RMSE (s) | 95% CI coverage (%) | Notes |
| -------: | ------: | -------: | ------------------: | ----- |
|   15 min |         |          |                     |       |
|   30 min |         |          |                     |       |
|   1 hour |         |          |                     |       |
|  2 hours |         |          |                     |       |
| 24 hours |         |          |                     |       |

## 9. Repo structure (recommended)

```
gnss-error-prediction/
├─ data/                 # raw and processed data
│  ├─ raw/               # original CSVs (DO NOT EDIT)
│  └─ processed/         # resampled / cleaned files used for training
├─ notebooks/            # EDA and experiments (notebooks)
├─ src/
│  ├─ data.py            # data loading & preprocessing functions
│  ├─ features.py        # feature engineering helpers
│  ├─ models/
│  │  ├─ lstm.py
│  │  ├─ transformer.py
│  │  └─ gp.py
│  ├─ train.py
│  └─ evaluate.py
├─ experiments/          # saved configs and checkpoints
├─ results/              # evaluation outputs and plots
├─ requirements.txt
└─ README.md             # this file (temporary)
```

## 10. How to run (example commands)

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Preprocess data (example):

```bash
python src/data.py --input_dir data/raw --output_dir data/processed --resample 15min
```

3. Train a baseline LSTM (example):

```bash
python src/train.py --config configs/lstm_baseline.yaml
```

4. Evaluate on held-out day (produce horizon-wise metrics):

```bash
python src/evaluate.py --checkpt experiments/lstm_baseline/last.ckpt --test_data data/processed/day8.csv
```

## 11. Dependencies (suggested)

* Python 3.10+
* numpy, pandas
* scikit-learn
* torch (PyTorch) or TensorFlow (choose one)
* gpytorch (optional, for Gaussian Processes)
* matplotlib (for plots)


