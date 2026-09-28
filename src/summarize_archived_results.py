#!/usr/bin/env python3
"""
Audit and combine the six archived wearable-device result workbooks.

This script does not modify the original .xlsx files.
It validates their schema, concatenates them into one CSV, and creates
an explicitly descriptive (unweighted across activities) overview.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

EXPECTED_COLUMNS = [
    "Device",
    "Activity",
    "Filter",
    "Model",
    "MAE_mean",
    "MAE_std",
    "RMSE_mean",
    "RMSE_std",
    "R2_mean",
    "R2_std",
]

EXPECTED_FILES = [
    "Apple_Watch_results.xlsx",
    "Biovotion_Everion_results.xlsx",
    "Empatica_E4_results.xlsx",
    "Fitbit_results.xlsx",
    "Garmin_results.xlsx",
    "Xiaomi_MiBand_results.xlsx",
]

COMBINED_PATH = RESULTS_DIR / "archive_results_combined.csv"
ACTIVITY_MEAN_PATH = RESULTS_DIR / "archive_results_activity_mean.csv"


def main() -> None:
    frames: list[pd.DataFrame] = []

    print("=" * 90)
    print("ARCHIVED RESULT WORKBOOK AUDIT")
    print("=" * 90)

    for filename in EXPECTED_FILES:
        path = RESULTS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing expected workbook: {path}")

        book = pd.ExcelFile(path)

        if len(book.sheet_names) != 1:
            raise ValueError(
                f"{filename}: expected exactly 1 sheet, found {book.sheet_names}"
            )

        sheet = book.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)

        if list(df.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"{filename}: unexpected columns.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {list(df.columns)}"
            )

        if len(df) != 36:
            raise ValueError(
                f"{filename}: expected 36 rows, found {len(df)}"
            )

        if df.isna().any().any():
            raise ValueError(
                f"{filename}: contains missing values"
            )

        duplicate_key = ["Device", "Activity", "Filter", "Model"]
        duplicates = int(df.duplicated(duplicate_key).sum())
        if duplicates:
            raise ValueError(
                f"{filename}: found {duplicates} duplicated "
                f"Device/Activity/Filter/Model rows"
            )

        numeric_columns = EXPECTED_COLUMNS[4:]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

        df = df.copy()
        df.insert(0, "Source_Workbook", filename)
        df.insert(1, "Source_Sheet", sheet)
        frames.append(df)

        print(f"\n{filename}")
        print(f"  sheet:      {sheet}")
        print(f"  rows:       {len(df)}")
        print(f"  activities: {sorted(df['Activity'].astype(str).unique().tolist())}")
        print(f"  filters:    {sorted(df['Filter'].astype(str).unique().tolist())}")
        print(f"  models:     {sorted(df['Model'].astype(str).unique().tolist())}")
        print(f"  missing:    0")
        print(f"  duplicates: 0")

    combined = pd.concat(frames, ignore_index=True)

    if len(combined) != 216:
        raise ValueError(
            f"Expected 216 total archived rows, found {len(combined)}"
        )

    combined.to_csv(COMBINED_PATH, index=False, encoding="utf-8")

    # Descriptive only: equal weight to each activity-specific summary.
    activity_mean = (
        combined.groupby(
            ["Device", "Filter", "Model"],
            as_index=False,
        )
        .agg(
            n_activity_rows=("Activity", "size"),
            MAE_mean_across_activities=("MAE_mean", "mean"),
            RMSE_mean_across_activities=("RMSE_mean", "mean"),
            R2_mean_across_activities=("R2_mean", "mean"),
        )
        .sort_values(
            ["Device", "MAE_mean_across_activities", "Filter", "Model"]
        )
        .reset_index(drop=True)
    )

    activity_mean.to_csv(
        ACTIVITY_MEAN_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("GLOBAL CHECK")
    print("=" * 90)
    print(f"Total rows: {len(combined)}")
    print(
        "Devices:",
        sorted(combined["Device"].astype(str).unique().tolist()),
    )
    print(
        "Activities:",
        sorted(combined["Activity"].astype(str).unique().tolist()),
    )
    print(
        "Filters:",
        sorted(combined["Filter"].astype(str).unique().tolist()),
    )
    print(
        "Models:",
        sorted(combined["Model"].astype(str).unique().tolist()),
    )

    print("\nDescriptive lowest activity-mean MAE row per device")
    print(
        "(UNWEIGHTED mean of the archived activity-specific MAE summaries; "
        "not a subject-wise reanalysis):"
    )

    best_rows = (
        activity_mean.sort_values(
            ["Device", "MAE_mean_across_activities"]
        )
        .groupby("Device", as_index=False)
        .first()
    )

    print(
        best_rows[
            [
                "Device",
                "Filter",
                "Model",
                "MAE_mean_across_activities",
                "RMSE_mean_across_activities",
                "R2_mean_across_activities",
            ]
        ].to_string(index=False)
    )

    print("\nCreated:")
    print(f"  {COMBINED_PATH}")
    print(f"  {ACTIVITY_MEAN_PATH}")
    print("\nAUDIT PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        raise
