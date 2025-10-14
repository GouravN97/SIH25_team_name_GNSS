#  README — GNSS Clock & Ephemeris Error Prediction

> **Project:** Predicting time-varying errors between uploaded (broadcast) and modeled satellite clock & ephemeris values
>
> 
---

##  Project Overview

This project develops ML models using PatchTST, a transformer-based model, to predict the time-varying differences (errors) between the **uploaded/broadcast** and **ICD-based modeled** values of GNSS satellite clock biases and ephemeris (orbit) parameters. The objective is to predict these errors at 15-minute intervals for an unseen 8th day, using the provided seven-day training dataset. Improvements in these predictions increase the accuracy and reliability of GNSS positioning and timing.

##  Dataset

* **Provided:** 7 days of timestamped error measurements (clock biases and ephemeris parameter differences) for GNSS satellites in GEO and MEO.
https://www.sac.gov.in/files/sih/SIH_Data_PS-08.zip

##  Goal

To show that **PatchTST predicts errors for smaller datasets (as the one provided) faster and more accurately** (as evaluated by the Shapiro-Wilk test for normalisation) **compared to an LSTM.** 


## What Our Model Does

* The training data is normalised (PatchTST does this automatically).
* We remove some outlying points, i.e., those outside the boundary of ± 1.5 σ (standard deviation).
* Train and test.

## How We Evaluated It

* We trained both the PatchTST model and LSTM model on the data of first 6 days.
* Then tested it against the data of the 7th day.
* 


## 11. Dependencies (suggested)

* Python 3.10+
* numpy, pandas
* scikit-learn
* torch (PyTorch) or TensorFlow (choose one)
* gpytorch (optional, for Gaussian Processes)
* matplotlib (for plots)


