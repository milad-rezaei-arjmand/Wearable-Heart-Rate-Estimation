# Wearable Heart Rate Estimation Using PPG Signals and Machine Learning

## Overview

This repository presents a machine learning framework for estimating
heart rate from photoplethysmography (PPG) signals collected from
commercial wearable devices.

The project investigates the impact of signal preprocessing, filtering
techniques, and regression-based models for improving wearable heart
rate estimation compared with ECG reference measurements.

The implemented framework includes:

-   PPG signal preprocessing
-   Noise reduction and signal filtering
-   Kalman filtering
-   Regression-based heart rate estimation
-   Neural network-based estimation
-   Quantitative performance evaluation

------------------------------------------------------------------------

# Research Objective

Wearable devices provide convenient physiological monitoring; however,
their heart rate measurements can be affected by motion artifacts,
sensor limitations, and environmental noise.

This project evaluates methods for improving heart rate estimation
accuracy from wearable PPG signals using advanced filtering and machine
learning approaches.

------------------------------------------------------------------------

# Dataset

The experiments were conducted using the BigIdeasLab STEP dataset
available through PhysioNet.

The dataset contains synchronized wearable PPG recordings and ECG
reference signals collected from multiple commercial wearable devices.

Raw dataset files are not included in this repository due to dataset
distribution restrictions and privacy considerations.

Dataset source:

PhysioNet - BigIdeasLab STEP Heart Rate Smartwatch Dataset

------------------------------------------------------------------------

# Wearable Devices

The evaluated devices include:

-   Apple Watch
-   Fitbit
-   Garmin
-   Xiaomi Mi Band
-   Empatica E4
-   Biovotion Everion

------------------------------------------------------------------------

# Methodology

The complete workflow:

    Wearable PPG Signal

            ↓

    Signal Preprocessing

            ↓

    Filtering Methods

    (Kalman Filtering / Butterworth Filtering)

            ↓

    Heart Rate Estimation Models

            ↓

    Performance Evaluation

------------------------------------------------------------------------

# Signal Processing

The preprocessing pipeline includes:

-   Signal cleaning
-   Noise reduction
-   Filtering approaches
-   Reference ECG comparison

Filtering methods evaluated:

-   Raw signal
-   Kalman filter
-   Butterworth filter

------------------------------------------------------------------------

# Machine Learning Models

The framework evaluates multiple estimation approaches:

## Linear Regression

A baseline regression model for heart rate estimation.

## Polynomial Regression

Polynomial regression with higher-order feature mapping.

## Deep Neural Network

A neural network model for nonlinear heart rate estimation.

------------------------------------------------------------------------

# Evaluation Metrics

Performance is evaluated using:

-   Mean Absolute Error (MAE)
-   Root Mean Square Error (RMSE)
-   Coefficient of Determination (R²)
-   Bland-Altman analysis

------------------------------------------------------------------------

# Results

The repository contains experimental results for each wearable device.

Included results:

-   Device-level performance tables
-   Model comparison
-   Filtering comparison
-   Visualization of estimation performance

Results are provided in:

    results/

------------------------------------------------------------------------

# Repository Structure

    Wearable-Heart-Rate-Estimation/

    ├── src/
    │   └── final_code.py
    │
    ├── results/
    │   ├── Device_results.xlsx
    │   ├── figures/
    │   └── summary_results.md
    │
    ├── data/
    │   └── README.md
    │
    ├── requirements.txt
    │
    ├── README.md
    │
    └── LICENSE

------------------------------------------------------------------------

# Installation

Install dependencies:

``` bash
pip install -r requirements.txt
```

Required libraries:

    numpy
    pandas
    scikit-learn
    scipy
    tensorflow
    matplotlib
    openpyxl

------------------------------------------------------------------------

# Usage

After downloading the required dataset:

1.  Prepare the dataset according to the expected format.
2.  Update dataset paths in the source code.
3.  Run the analysis pipeline:

``` bash
python src/final_code.py
```

The pipeline performs:

    Data Loading

    ↓

    Signal Processing

    ↓

    Filtering

    ↓

    Model Training

    ↓

    Evaluation

    ↓

    Result Generation

------------------------------------------------------------------------

# Publication

This project is based on:

**Improving Heart Rate Estimation in Commercial Wearable Devices Using
Kalman Filtering and Regression Models**

Presented at the 32nd National and 10th International Iranian Conference
on Biomedical Engineering.

------------------------------------------------------------------------

# Future Work

Future improvements include:

-   Deep learning based PPG waveform modeling
-   Transformer-based physiological signal analysis
-   Real-time wearable inference
-   Personalized heart rate estimation models

------------------------------------------------------------------------

# Author

**Milad Rezaei Arjmand**

M.Sc. Student in Biomedical Engineering (Bioelectric)

Research Interests:

-   Medical Artificial Intelligence
-   Biomedical Signal Processing
-   Wearable Health Monitoring
-   Machine Learning for Healthcare
