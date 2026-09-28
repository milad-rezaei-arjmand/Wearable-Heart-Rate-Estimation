# Improving Heart Rate Estimation in Commercial Wearable Devices Using Kalman Filtering and Regression Models

## Overview

This repository accompanies a biomedical engineering study on improving and evaluating heart-rate measurements reported by commercial wearable devices against an ECG-derived reference.

The analysis uses the **BigIdeasLab_STEP: Heart rate measurements captured by smartwatches for differing skin tones** dataset from PhysioNet.

> **Important data note:** the public/restricted dataset contains **heart-rate values reported by the wearables** and the corresponding **ECG-derived heart-rate reference**. It does **not** provide raw PPG waveform signals. The wearable HR measurements originate from optical heart-rate sensors, but this repository analyzes the reported HR values rather than raw photoplethysmography waveforms.

---

## Research Objective

Commercial wearable heart-rate measurements can vary across devices, activities, participants, and acquisition conditions.

This project evaluates:

- raw wearable-reported heart rate,
- Kalman-filtered heart rate,
- Butterworth-filtered heart rate,
- linear regression,
- polynomial regression,
- and an optional dense neural-network regressor,

using ECG-derived heart rate as the reference target.

---

## Dataset

The project uses:

**BigIdeasLab_STEP: Heart rate measurements captured by smartwatches for differing skin tones, version 1.0**
PhysioNet DOI: https://doi.org/10.13026/cqfy-d860

The official dataset contains time-synchronized observations for **53 participants** and includes:

- ECG-derived heart rate,
- Apple Watch heart rate,
- Empatica E4 heart rate,
- Garmin heart rate,
- Fitbit heart rate,
- Xiaomi Miband heart rate,
- Biovotion Everion heart rate,
- participant ID,
- Fitzpatrick skin tone,
- activity label.

The study protocol includes rest, deep breathing, walking/activity, and typing conditions.

The dataset is restricted-access and is **not redistributed in this repository**.

See [`data/README.md`](data/README.md) for access and local file-layout instructions.

---

## Evaluated Devices

- Apple Watch 4
- Fitbit Charge 2
- Garmin Vivosmart 3
- Xiaomi Miband 3
- Empatica E4
- Biovotion Everion

---

## Repository Structure

```text
Wearable-Heart-Rate-Estimation/
├── data/
│   └── README.md
├── paper/
│   └── README.md
├── results/
│   ├── Apple_Watch_results.xlsx
│   ├── Biovotion_Everion_results.xlsx
│   ├── Empatica_E4_results.xlsx
│   ├── Fitbit_results.xlsx
│   ├── Garmin_results.xlsx
│   ├── Xiaomi_MiBand_results.xlsx
│   ├── figures/
│   └── summary_results.md
├── src/
│   ├── legacy_original_analysis.py
│   ├── run_analysis.py
│   └── summarize_archived_results.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-dnn.txt
```

---

## Installation

Create and activate a virtual environment, then install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The core pipeline supports the classical filtering and regression analyses without TensorFlow.

To enable the optional dense neural-network model:

```bash
pip install -r requirements-dnn.txt
```

---

## Analysis Code

### Recommended cleaned pipeline

`src/run_analysis.py` reads the official single CSV structure directly and supports two validation modes:

- `subject` — **recommended**, using `GroupKFold` with participant ID so train and test subjects are disjoint.
- `row` — shuffled row-wise `KFold`, provided only for comparison with the earlier analysis style.

Example:

```bash
python src/run_analysis.py \
  --dataset data/bigideaslab_step_hr.csv \
  --validation subject
```

To include the optional TensorFlow DNN model:

```bash
python src/run_analysis.py \
  --dataset data/bigideaslab_step_hr.csv \
  --validation subject \
  --include-dnn
```

Generated files are written under `outputs/`, which is ignored by Git.

### Legacy analysis

`src/legacy_original_analysis.py` preserves the original analysis script associated with the previously generated conference-project outputs.

The legacy script uses row-wise shuffled K-fold validation. Because rows from the same participant may occur in both training and test folds, those historical results should **not** be described as subject-independent validation.

The cleaned pipeline adds a participant-disjoint validation option without rewriting or retroactively changing the archived conference outputs.

---

## Filtering Methods

The analysis compares:

- Raw wearable-reported HR
- Kalman filtering
- Butterworth low-pass filtering

Filtering is applied separately within participant subsets in the cleaned pipeline to avoid filtering across participant boundaries.

---

## Regression Models

### Linear Regression

A linear mapping from wearable-reported HR to ECG-derived reference HR.

### Polynomial Regression

Third-degree polynomial feature mapping followed by linear regression.

### Dense Neural Network

An optional TensorFlow/Keras regressor with two hidden layers.

The DNN is not required for the paper title or the core filtering/regression comparison and can be disabled for faster reproducibility.

---

## Evaluation Metrics

The analysis reports:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Coefficient of Determination (R²)

The cleaned pipeline also creates a Bland-Altman overview using the raw wearable-vs-ECG measurements.

---

## Results

Archived device-level spreadsheets and figures are available in [`results/`](results/).

The six archived workbooks contain **216 activity/filter/model result rows** in total:

- 6 wearable devices
- 4 activity labels
- 3 filtering conditions
- 3 regression/model variants

A descriptive audit of the archived workbooks found no missing values or duplicated Device/Activity/Filter/Model rows.

For each device, the table below shows the configuration with the **lowest unweighted mean MAE across the four archived activity-specific summaries**:

| Device | Filter | Model | Mean MAE across activities | Mean RMSE across activities | Mean R² across activities |
|---|---|---|---:|---:|---:|
| Apple Watch | Kalman | Deep | 2.380 | 3.603 | 0.921 |
| Biovotion | Kalman | Deep | 3.447 | 6.119 | 0.799 |
| Empatica | Kalman | Polynomial | 6.773 | 9.724 | 0.517 |
| Fitbit | Kalman | Linear | 4.906 | 6.897 | 0.730 |
| Garmin | Kalman | Deep | 4.321 | 6.182 | 0.798 |
| Miband | Kalman | Polynomial | 6.605 | 9.049 | 0.555 |

These values are **descriptive summaries of the archived row-wise cross-validation outputs**. They are not participant-disjoint reanalysis results and should not be interpreted as subject-independent validation.

A complete machine-readable audit is available in:

- `results/archive_results_combined.csv`
- `results/archive_results_activity_mean.csv`

Additional interpretation is provided in [`results/summary_results.md`](results/summary_results.md).

New participant-disjoint results generated with:

```bash
--validation subject
```

should be reported separately rather than silently replacing the archived conference outputs.

---

## Publication

This repository is associated with the conference work:

**Improving Heart Rate Estimation in Commercial Wearable Devices Using Kalman Filtering and Regression Models**

Presented at the **32nd National and 10th International Iranian Conference on Biomedical Engineering** and included in the conference proceedings/booklet.

See [`paper/README.md`](paper/README.md).

---

## Limitations

Important limitations include:

- the dataset contains reported HR values rather than raw PPG waveforms,
- the data were collected in 2019 and device firmware/hardware may have changed,
- the current modeling does not explicitly use skin tone as a predictor,
- archived conference outputs were generated using the earlier row-wise validation workflow,
- and external validation on a separate cohort has not been performed.

This repository is a research/engineering project and is not intended for clinical use.

---

## Dataset Citation

Bent, B., & Dunn, J. (2021). *BigIdeasLab_STEP: Heart rate measurements captured by smartwatches for differing skin tones* (Version 1.0). PhysioNet.
https://doi.org/10.13026/cqfy-d860

The associated PhysioNet page also requests citation of the original wearable-sensor accuracy publication.

---

## Author

**Milad Rezaei Arjmand**
M.Sc. Student in Biomedical Engineering (Bioelectric)

Research interests include Biomedical Signal Processing, Medical AI, Wearable Health Monitoring, and Machine Learning for Healthcare.

---

## License

Code and repository documentation are released under the MIT License.

The PhysioNet dataset is **not** covered by this repository license and remains subject to the original PhysioNet access agreement and data-use terms.
