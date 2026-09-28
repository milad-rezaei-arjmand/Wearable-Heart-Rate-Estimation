#!/usr/bin/env python3
"""
Clean analysis entry point for the BigIdeasLab_STEP wearable heart-rate dataset.

Important:
- The PhysioNet dataset contains heart-rate values reported by commercial
  wearables plus an ECG-derived heart-rate reference; it does NOT provide
  raw PPG waveforms.
- Subject-wise validation is recommended to keep participant IDs disjoint
  between training and test folds.
- The legacy/original conference-analysis script can be kept separately for
  historical reproducibility of previously generated result files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DEVICE_COLUMNS = [
    "Apple Watch",
    "Biovotion",
    "Empatica",
    "Fitbit",
    "Garmin",
    "Miband",
]

REQUIRED_COLUMNS = {
    "ECG",
    "Apple Watch",
    "Empatica",
    "Garmin",
    "Fitbit",
    "Miband",
    "Biovotion",
    "ID",
    "Skin Tone",
    "Activity",
}

FILTER_TYPES = ["Raw", "Kalman", "Butterworth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate commercial wearable-reported heart rate against "
            "ECG-derived reference heart rate."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/bigideaslab_step_hr.csv"),
        help="Path to the official BigIdeasLab_STEP CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated metrics and figures.",
    )
    parser.add_argument(
        "--validation",
        choices=["subject", "row"],
        default="subject",
        help=(
            "'subject' uses GroupKFold with participant ID and is recommended. "
            "'row' uses shuffled KFold for comparison with legacy row-wise analysis."
        ),
    )
    parser.add_argument(
        "--include-dnn",
        action="store_true",
        help="Also train the TensorFlow dense neural-network regressor.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the restricted-access dataset from PhysioNet and place "
            "the CSV at the requested path."
        )

    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.copy()
    data["Activity"] = data["Activity"].astype("string").str.strip()
    data = data[data["Activity"].notna()]
    data = data[data["Activity"].str.len() > 0]

    return data


def kalman_filter(values: np.ndarray, q: float = 1e-4, r: float = 1e-2) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values

    estimate = np.zeros(values.size, dtype=np.float32)
    covariance = np.zeros(values.size, dtype=np.float32)
    estimate[0] = values[0]
    covariance[0] = 1.0

    for index in range(1, values.size):
        predicted_estimate = estimate[index - 1]
        predicted_covariance = covariance[index - 1] + q
        gain = predicted_covariance / (predicted_covariance + r)

        estimate[index] = (
            predicted_estimate
            + gain * (values[index] - predicted_estimate)
        )
        covariance[index] = (1.0 - gain) * predicted_covariance

    return estimate


def butterworth_filter(
    values: np.ndarray,
    order: int = 3,
    cutoff: float = 0.1,
    zero_phase: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values

    b, a = butter(order, cutoff, btype="low", analog=False)

    if not zero_phase:
        return lfilter(b, a, values).astype(np.float32)

    padlen = 3 * (max(len(a), len(b)) - 1)
    if values.size <= padlen:
        return lfilter(b, a, values).astype(np.float32)

    try:
        return filtfilt(b, a, values).astype(np.float32)
    except ValueError:
        return lfilter(b, a, values).astype(np.float32)


def apply_filter(values: np.ndarray, filter_type: str) -> np.ndarray:
    if filter_type == "Raw":
        return np.asarray(values, dtype=np.float32)
    if filter_type == "Kalman":
        return kalman_filter(values)
    if filter_type == "Butterworth":
        return butterworth_filter(values)
    raise ValueError(f"Unknown filter type: {filter_type}")


def filter_subset_by_subject(
    subset: pd.DataFrame,
    device: str,
    filter_type: str,
) -> np.ndarray:
    """
    Filter each participant separately and restore the subset's original row order.

    This avoids applying a temporal filter across participant boundaries.
    """
    working = subset[["ID", device]].copy()
    output = pd.Series(index=working.index, dtype=np.float32)

    for _, group in working.groupby("ID", sort=False):
        filtered = apply_filter(
            group[device].to_numpy(dtype=np.float32),
            filter_type,
        )
        output.loc[group.index] = filtered

    return output.loc[working.index].to_numpy(dtype=np.float32)


def make_splitter(
    validation: str,
    folds: int,
    random_state: int,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if validation == "subject":
        unique_groups = np.unique(groups)
        if unique_groups.size < folds:
            raise ValueError(
                f"Need at least {folds} unique subjects; found {unique_groups.size}."
            )
        splitter = GroupKFold(n_splits=folds)
        return splitter.split(x, y, groups=groups)

    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    return splitter.split(x)


def build_dnn(random_state: int):
    # TensorFlow is imported lazily so the classical models can run without it.
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    tf.keras.utils.set_random_seed(random_state)

    model = Sequential(
        [
            Input(shape=(1,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, dtype="float32"),
        ]
    )
    model.compile(optimizer=Adam(0.01), loss="mse")
    return model


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    return float(mae), rmse, float(r2)


def summarize_fold_metrics(
    device: str,
    activity: str,
    filter_type: str,
    model_name: str,
    fold_metrics: list[tuple[float, float, float]],
    validation: str,
) -> dict[str, object]:
    values = np.asarray(fold_metrics, dtype=float)

    return {
        "Device": device,
        "Activity": activity,
        "Filter": filter_type,
        "Model": model_name,
        "Validation": validation,
        "MAE_mean": float(values[:, 0].mean()),
        "MAE_std": float(values[:, 0].std(ddof=1)),
        "RMSE_mean": float(values[:, 1].mean()),
        "RMSE_std": float(values[:, 1].std(ddof=1)),
        "R2_mean": float(values[:, 2].mean()),
        "R2_std": float(values[:, 2].std(ddof=1)),
    }


def save_bland_altman_overview(
    data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create a compact raw-measurement Bland-Altman overview for all devices."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    for axis, device in zip(axes.flat, DEVICE_COLUMNS):
        subset = data[["ECG", device]].dropna()
        reference = subset["ECG"].to_numpy(dtype=float)
        wearable = subset[device].to_numpy(dtype=float)

        means = (reference + wearable) / 2.0
        differences = wearable - reference

        mean_difference = float(np.mean(differences))
        sd_difference = float(np.std(differences, ddof=1))
        upper = mean_difference + 1.96 * sd_difference
        lower = mean_difference - 1.96 * sd_difference

        axis.scatter(means, differences, s=6, alpha=0.25)
        axis.axhline(mean_difference, linewidth=1)
        axis.axhline(upper, linestyle="--", linewidth=1)
        axis.axhline(lower, linestyle="--", linewidth=1)
        axis.set_title(device)
        axis.set_xlabel("Mean HR (bpm)")
        axis.set_ylabel("Wearable - ECG (bpm)")

    figure.suptitle("Bland-Altman overview: wearable HR vs ECG reference")
    figure.savefig(
        output_dir / "bland_altman_overview.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset)

    if args.validation == "row":
        print(
            "WARNING: row-wise KFold is not participant-disjoint. "
            "Use --validation subject for the recommended evaluation."
        )

    activities = sorted(data["Activity"].dropna().unique().tolist())
    results: list[dict[str, object]] = []

    start_all = perf_counter()

    for filter_type in FILTER_TYPES:
        for device in DEVICE_COLUMNS:
            for activity in activities:
                subset = (
                    data.loc[
                        data["Activity"].eq(activity),
                        ["ECG", "ID", device],
                    ]
                    .dropna()
                    .copy()
                )

                if len(subset) < max(10, args.folds):
                    continue

                x_raw = subset[device].to_numpy(dtype=np.float32).reshape(-1, 1)
                y = subset["ECG"].to_numpy(dtype=np.float32)
                groups = subset["ID"].astype(str).to_numpy()

                split_iterator = make_splitter(
                    validation=args.validation,
                    folds=args.folds,
                    random_state=args.random_state,
                    x=x_raw,
                    y=y,
                    groups=groups,
                )

                linear_metrics: list[tuple[float, float, float]] = []
                polynomial_metrics: list[tuple[float, float, float]] = []
                dnn_metrics: list[tuple[float, float, float]] = []

                for fold_number, (train_idx, test_idx) in enumerate(
                    split_iterator,
                    start=1,
                ):
                    if args.validation == "subject":
                        train_groups = set(groups[train_idx])
                        test_groups = set(groups[test_idx])
                        if train_groups.intersection(test_groups):
                            raise RuntimeError(
                                "Participant leakage detected in subject-wise validation."
                            )

                    train_subset = subset.iloc[train_idx]
                    test_subset = subset.iloc[test_idx]

                    x_train = filter_subset_by_subject(
                        train_subset,
                        device,
                        filter_type,
                    ).reshape(-1, 1)
                    x_test = filter_subset_by_subject(
                        test_subset,
                        device,
                        filter_type,
                    ).reshape(-1, 1)

                    y_train = y[train_idx]
                    y_test = y[test_idx]

                    linear = LinearRegression()
                    linear.fit(x_train, y_train)
                    pred_linear = linear.predict(x_test)
                    linear_metrics.append(
                        calculate_metrics(y_test, pred_linear)
                    )

                    polynomial = PolynomialFeatures(degree=3)
                    x_train_poly = polynomial.fit_transform(x_train)
                    x_test_poly = polynomial.transform(x_test)

                    polynomial_regressor = LinearRegression()
                    polynomial_regressor.fit(x_train_poly, y_train)
                    pred_polynomial = polynomial_regressor.predict(x_test_poly)
                    polynomial_metrics.append(
                        calculate_metrics(y_test, pred_polynomial)
                    )

                    if args.include_dnn:
                        scaler = StandardScaler().fit(x_train)
                        x_train_scaled = scaler.transform(x_train)
                        x_test_scaled = scaler.transform(x_test)

                        model = build_dnn(
                            args.random_state + fold_number
                        )
                        model.fit(
                            x_train_scaled,
                            y_train,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            verbose=0,
                        )
                        pred_dnn = model.predict(
                            x_test_scaled,
                            verbose=0,
                        ).reshape(-1)
                        dnn_metrics.append(
                            calculate_metrics(y_test, pred_dnn)
                        )

                results.append(
                    summarize_fold_metrics(
                        device,
                        activity,
                        filter_type,
                        "Linear",
                        linear_metrics,
                        args.validation,
                    )
                )
                results.append(
                    summarize_fold_metrics(
                        device,
                        activity,
                        filter_type,
                        "Polynomial",
                        polynomial_metrics,
                        args.validation,
                    )
                )

                if args.include_dnn:
                    results.append(
                        summarize_fold_metrics(
                            device,
                            activity,
                            filter_type,
                            "DNN",
                            dnn_metrics,
                            args.validation,
                        )
                    )

    results_table = pd.DataFrame(results)
    results_table = results_table.sort_values(
        ["Device", "Activity", "Filter", "Model"]
    ).reset_index(drop=True)

    results_path = output_dir / "final_results_by_filter.csv"
    results_table.to_csv(results_path, index=False)

    save_bland_altman_overview(data, output_dir)

    metadata = {
        "dataset": str(args.dataset),
        "validation": args.validation,
        "folds": args.folds,
        "include_dnn": args.include_dnn,
        "random_state": args.random_state,
        "devices": DEVICE_COLUMNS,
        "filters": FILTER_TYPES,
        "activities": activities,
        "elapsed_seconds": perf_counter() - start_all,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {results_path}")
    print(f"Saved: {output_dir / 'bland_altman_overview.png'}")
    print(f"Saved: {output_dir / 'run_metadata.json'}")
    print(f"Completed in {metadata['elapsed_seconds']:.1f} seconds.")


if __name__ == "__main__":
    main()
