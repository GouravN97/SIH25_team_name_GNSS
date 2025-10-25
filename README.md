# 🛰️ GNSS Clock & Ephemeris Error Prediction

> **Project:** Predicting time-varying errors between uploaded (broadcast) and modeled GNSS clock & ephemeris values using PatchTST, a transformer-based model.

---

## 🚀 How to Run 

### 1. Required Files

Download the `PatchTST_model.zip` and extract it.

### 2. Create Virtual Environment

```bash
# On Linux/WSL
python3 -m venv patchtst_env
source patchtst_env/bin/activate  
```

```cmd
# command prompt
py -m venv patchtst_env
patchtst_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Model

Now the model is set to run!
Just modify the commands as per the name of the dataset(e.g., Data_GEO_Train.csv) and the specific column name(e.g., z_error (m)) in that dataset.

#### (i) For testing against 7th day (shows Shapiro-Wilk Test Results and Double Line Graph)

```bash
#Linux/WSL
python3 run_full.py --input_file Data_GEO_Train.csv --target_column "z_error (m)"
```

```cmd
# command prompt
python run_full.py --input_file Data_GEO_Train.csv --target_column "z_error (m)"
```

#### (ii) For testing against 8th day (unspecified)

```bash
#Linux/WSL
python3 run_8thday.py --input_file Data_GEO_Train.csv --target_column "z_error (m)"
```

```cmd
# command prompt
python run_8thday.py --input_file Data_GEO_Train.csv --target_column "z_error (m)"
```

### 5. View Results

Results for the next day are automatically saved in the **`results`** folder and plots are automatically saved in the **`graphs`** folder.

---

## 📋 Project Overview

This project develops ML models using **PatchTST**, a transformer-based model, to predict the time-varying differences (errors) between the **uploaded/broadcast** and **ICD-based modeled** values of GNSS satellite clock biases and ephemeris (orbit) parameters. 

The objective is to predict these errors at 15-minute intervals for an unseen 8th day, using the provided seven-day training dataset. Improvements in these predictions increase the accuracy and reliability of GNSS positioning and timing.

---

## 📊 Dataset
**Provided:** 7 days of timestamped error measurements (clock biases and ephemeris parameter differences) for GNSS satellites in GEO and MEO.

🔗 [Download Dataset](https://www.sac.gov.in/files/sih/SIH_Data_PS-08.zip)
**NOTE:** THE DATA_MEO_Train2.csv file is only modified in our folder to remove duplicate data and make it easier for the program to process

---

## 🎯 Goal

To show that **PatchTST predicts errors for smaller datasets (as the one provided) faster and more accurately** (as evaluated by the Shapiro-Wilk test for normalisation) **compared to an LSTM.**

---

## ⚙️ What Our Model Does

- The training data is normalised (PatchTST does this automatically).
- We remove some outlying points, i.e., those outside the boundary of ± 1.5 σ (standard deviation).
- Train and test.

---

## 📈 How We Evaluated It

- We trained both the PatchTST model and LSTM model on the data of first 6 days.
- Then tested it against the data of the 7th day.
- The Shapiro-Wilk test for normalisation passed in 3 out of 4 cases, with the 4th case's score close passing.
- The mse, rse and rae values were within reasonable bounds.

---

## 📦 Dependencies (suggested)

- Python
- numpy, pandas
- scikit-learn
- torch
- matplotlib (for plots)
- _optional:_ darts (for seeing how LSTM performs on the same dataset, download the lstm_model.py file from the lstm folder and run).

---

## 📁 Project Structure

```
├── run_full.py              # Main pipeline for 7th day prediction
├── run_8thday.py            # Pipeline for 8th day prediction
├── scripts/
│   └── PatchTST/
│       ├── geodata.sh       # Training script for 7th day
│       └── geodata8thday.sh # Training script for 8th day
├── results/                 # Model outputs and predictions
├── graphs/                  # Generated visualization plots
├── Data_GEO_Train.csv       # Input training data
└── requirements.txt         # Python dependencies
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!


---



---


