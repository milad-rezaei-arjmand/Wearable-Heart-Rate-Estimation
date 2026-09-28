# Dataset

## Official Source

This project uses:

**BigIdeasLab_STEP: Heart rate measurements captured by smartwatches for differing skin tones, version 1.0**

PhysioNet: https://physionet.org/content/bigideaslab-step-hr-smartwatch/1.0/
DOI: https://doi.org/10.13026/cqfy-d860

The resource is restricted-access. Users must satisfy the PhysioNet access requirements and data-use agreement before downloading the dataset.

---

## What the Dataset Contains

The dataset consists of time-synchronized **heart-rate values**, not raw PPG waveform recordings.

Columns documented by PhysioNet include:

```text
ECG
Apple Watch
Empatica
Garmin
Fitbit
Miband
Biovotion
ID
Skin Tone
Activity
```

`ECG` is the heart rate reported by the Bittium Faros 180 ECG patch and is used as the reference.

The wearable columns contain heart rate reported by the corresponding devices.

Wearable HR is produced by optical sensors internally, but the downloadable dataset does not expose the raw PPG waveform used by those devices.

---

## Participants and Activities

The official resource reports:

- 53 participants
- study data collected in 2019
- activities including rest, paced deep breathing, walking/activity, and typing

The six evaluated wearable devices are:

- Apple Watch 4
- Fitbit Charge 2
- Garmin Vivosmart 3
- Xiaomi Miband 3
- Empatica E4
- Biovotion Everion

---

## Local Layout

Do not commit the restricted dataset to GitHub.

Place the downloaded CSV locally as:

```text
data/
├── README.md
└── bigideaslab_step_hr.csv
```

The repository `.gitignore` excludes dataset files under `data/` while retaining this README.

Run the cleaned analysis with:

```bash
python src/run_analysis.py \
  --dataset data/bigideaslab_step_hr.csv \
  --validation subject
```

---

## Citation

Bent, B., & Dunn, J. (2021). *BigIdeasLab_STEP: Heart rate measurements captured by smartwatches for differing skin tones* (Version 1.0). PhysioNet.
https://doi.org/10.13026/cqfy-d860

Please also follow the citation instructions on the official PhysioNet resource page.
