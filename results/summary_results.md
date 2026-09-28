# Results Summary

This directory contains archived outputs from the wearable heart-rate project associated with the conference study.

## Workbook Audit

The six archived device workbooks were audited programmatically.

Across the archive there are:

- **6 wearable devices**
- **4 activity labels:** `Activity`, `Breathe`, `Rest`, `Type`
- **3 filtering conditions:** `Raw`, `Kalman`, `Butterworth`
- **3 model variants:** `Linear`, `Polynomial`, `Deep`
- **36 rows per device workbook**
- **216 result rows in total**

All six workbooks use the same 10-column result schema:

```text
Device
Activity
Filter
Model
MAE_mean
MAE_std
RMSE_mean
RMSE_std
R2_mean
R2_std
```

The audit found:

- no missing values,
- no duplicated Device/Activity/Filter/Model rows,
- and no schema mismatches.

## Archived Descriptive Overview

For each device, the following row has the **lowest unweighted mean MAE across the four archived activity-specific result summaries**:

| Device | Filter | Model | Mean MAE across activities | Mean RMSE across activities | Mean R² across activities |
|---|---|---|---:|---:|---:|
| Apple Watch | Kalman | Deep | 2.379575 | 3.603129 | 0.920810 |
| Biovotion | Kalman | Deep | 3.447387 | 6.118758 | 0.799168 |
| Empatica | Kalman | Polynomial | 6.773099 | 9.723760 | 0.516841 |
| Fitbit | Kalman | Linear | 4.906310 | 6.896872 | 0.730325 |
| Garmin | Kalman | Deep | 4.321279 | 6.181793 | 0.798137 |
| Miband | Kalman | Polynomial | 6.605118 | 9.048701 | 0.555072 |

### Interpretation

The table above is a **descriptive archive summary only**.

It is calculated by giving equal weight to the four activity-specific result rows for each Device/Filter/Model combination. It is **not**:

- a pooled participant-level estimate,
- a weighted estimate based on sample count,
- a subject-independent validation result,
- or a replacement for the original activity-specific workbook results.

The fact that the lowest descriptive activity-mean MAE row for all six devices uses the Kalman condition should therefore be interpreted only within the archived row-wise analysis framework.

## Machine-Readable Audit Files

The audit script generates:

```text
archive_results_combined.csv
archive_results_activity_mean.csv
```

`archive_results_combined.csv` concatenates all 216 archived result rows and records the source workbook and source sheet.

`archive_results_activity_mean.csv` contains the descriptive, unweighted across-activity summaries used for the table above.

The source workbooks remain unchanged.

## Device-Level Result Files

```text
Apple_Watch_results.xlsx
Biovotion_Everion_results.xlsx
Empatica_E4_results.xlsx
Fitbit_results.xlsx
Garmin_results.xlsx
Xiaomi_MiBand_results.xlsx
```

## Figures

```text
figures/apple_watch_raw_vs_kalman.png
figures/bland_altman_analysis.png
```

## Validation Note

The archived result files were generated using the earlier analysis script, which uses shuffled row-wise K-fold validation.

Because multiple observations can belong to the same participant, row-wise splitting does not guarantee participant-disjoint train/test folds.

For stricter generalization analysis, the cleaned pipeline in:

```text
src/run_analysis.py
```

supports:

```text
--validation subject
```

which uses participant ID with `GroupKFold`.

Results from the subject-wise pipeline should be reported as a **new validation analysis** and should not be silently substituted for the archived conference outputs.

## Data Interpretation

The BigIdeasLab_STEP dataset provides wearable-reported heart-rate values and an ECG-derived heart-rate reference. The repository does not contain or analyze raw PPG waveforms from the commercial devices.

## Clinical Scope

These outputs are research results and are not intended to establish clinical diagnostic performance.
