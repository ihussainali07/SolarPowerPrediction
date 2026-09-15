import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regression import (
    cost,
    fit_batch_gd,
    fit_normal,
    fit_sgd,
    hypothesis,
    rmse,
)


# =====================================================================
# DIRECTORIES
# =====================================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
WEIGHTS_DIR = RESULTS_DIR / "weights"

for directory in (
    RESULTS_DIR,
    FIG_DIR,
    WEIGHTS_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =====================================================================
# DATE SPLIT
# =====================================================================

TRAIN_START = pd.Timestamp("2020-05-15 00:00:00")
TRAIN_END = pd.Timestamp("2020-06-10 23:59:59")

TEST_START = pd.Timestamp("2020-06-11 00:00:00")
TEST_END = pd.Timestamp("2020-06-17 23:59:59")


# =====================================================================
# FEATURE SETS
# =====================================================================

SET_A_COLS = [
    "irradiation",
    "module_temp",
    "ambient_temp",
]

SET_B_COLS = [
    "sw_radiation",
    "temp_2m",
    "cloud_cover",
]


# =====================================================================
# FINAL TRAINING SETTINGS
# =====================================================================

# Task 4.7 uses exactly:
#
# Batch:
#   500 iterations
#
# SGD:
#   50 epochs
#
# These are used to choose alpha.
#
# For the final Batch GD model, we use many more iterations because
# Task 4.8 requires Batch GD to reach approximately the same theta
# as the normal equation.

FINAL_BATCH_ITERS = 20000


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def add_time_features(df):
    """
    Add hour, sin_hour and cos_hour.

    Time features:

        sin(2*pi*h/24)
        cos(2*pi*h/24)

    where h is hour of day.
    """

    df = df.copy()

    df["hour"] = df["datetime"].dt.hour

    df["sin_hour"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["cos_hour"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    return df


# =====================================================================
# COLUMN NORMALIZATION
# =====================================================================

def normalize_column_name(column):
    """
    Convert different possible CSV column spellings into a
    consistent format.

    Examples:

        'SW Radiation' -> 'sw_radiation'
        'sw radiation' -> 'sw_radiation'
        'SW_RADIATION' -> 'sw_radiation'
        'Temp 2m'      -> 'temp_2m'
        'Cloud Cover'  -> 'cloud_cover'
    """

    column = str(column).strip().lower()

    # Replace spaces and common separators
    column = column.replace(" ", "_")
    column = column.replace("-", "_")
    column = column.replace("/", "_")

    # Remove repeated underscores
    while "__" in column:
        column = column.replace("__", "_")

    return column


def normalize_dataframe_columns(df):
    """
    Normalize all DataFrame column names.
    """

    df = df.copy()

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    return df


# =====================================================================
# LOAD SET A
# =====================================================================

def load_set_a():
    """
    Load prepared hourly Plant 1 data.

    Expected file:

        data/plant1_hourly.csv
    """

    path = DATA_DIR / "plant1_hourly.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file:\n{path}\n\n"
            "Run prepare.py first."
        )

    df = pd.read_csv(path)

    df = normalize_dataframe_columns(df)

    if "datetime" not in df.columns:
        raise ValueError(
            "Set A file does not contain a 'datetime' column."
        )

    if "ac_power" not in df.columns:
        raise ValueError(
            "Set A file does not contain an 'ac_power' column."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    df = add_time_features(df)

    df = df.sort_values("datetime").reset_index(
        drop=True
    )

    return df


# =====================================================================
# LOAD SET B
# =====================================================================

def load_set_b():
    """
    Load Open-Meteo hourly weather data.

    Supports either:

        data/plant1_openmeteo.csv

    or:

        data/plant1_hourly_openmeteo.csv

    The assignment's required Set B features are:

        sw_radiation
        temp_2m
        cloud_cover

    The target variable remains:

        ac_power

    IMPORTANT:
    Open-Meteo normally does not contain the on-site AC power target or
    the on-site irradiation sensor value.

    Therefore, both AC power and irradiation are merged from:

        data/plant1_hourly.csv

    Irradiation is used ONLY to identify daytime test rows:

        irradiation > 0

    It is NOT used as a Set B prediction feature.

    AC power is the prediction target and is NOT included as a Set B
    input feature.
    """

    possible_files = [
        DATA_DIR / "plant1_openmeteo.csv",
        DATA_DIR / "plant1_hourly_openmeteo.csv",
    ]

    weather_path = None

    for path in possible_files:
        if path.exists():
            weather_path = path
            break

    if weather_path is None:
        return None

    print(
        f"\nLoading Set B weather data from:\n"
        f"{weather_path}"
    )

    weather = pd.read_csv(weather_path)
    weather = normalize_dataframe_columns(weather)

    print("\nSet B columns found in file:")
    print(list(weather.columns))

    if "datetime" not in weather.columns:
        raise ValueError(
            "Set B file must contain a datetime column."
        )

    weather["datetime"] = pd.to_datetime(
        weather["datetime"],
        errors="coerce",
    )

    if weather["datetime"].isna().any():
        bad_rows = int(weather["datetime"].isna().sum())
        raise ValueError(
            f"Set B contains {bad_rows} invalid datetime value(s)."
        )

    # ---------------------------------------------------------------
    # Rename possible alternative spellings
    # ---------------------------------------------------------------

    rename_map = {}

    possible_sw = [
        "sw_radiation",
        "shortwave_radiation",
        "shortwave_rad",
        "sw_radiation_w_m2",
    ]

    possible_temp = [
        "temp_2m",
        "temperature_2m",
        "temperature_2_m",
        "temperature_2m_c",
    ]

    possible_cloud = [
        "cloud_cover",
        "cloudcover",
        "cloud_cover_percent",
    ]

    for name in possible_sw:
        if name in weather.columns:
            rename_map[name] = "sw_radiation"
            break

    for name in possible_temp:
        if name in weather.columns:
            rename_map[name] = "temp_2m"
            break

    for name in possible_cloud:
        if name in weather.columns:
            rename_map[name] = "cloud_cover"
            break

    weather = weather.rename(columns=rename_map)

    # ---------------------------------------------------------------
    # Check required public weather features
    # ---------------------------------------------------------------

    missing = [
        column
        for column in SET_B_COLS
        if column not in weather.columns
    ]

    if missing:
        raise ValueError(
            "\nSet B is missing required Open-Meteo columns:\n"
            f"{missing}\n\n"
            "Columns found:\n"
            f"{list(weather.columns)}\n\n"
            "The required Set B features are:\n"
            "  sw_radiation\n"
            "  temp_2m\n"
            "  cloud_cover"
        )

    # ---------------------------------------------------------------
    # Keep only public-weather inputs before merging the plant target.
    # Some prepared Open-Meteo files already contain ac_power and
    # irradiation; retaining those columns would create ac_power_x/y
    # during the merge and break the target lookup below.
    # ---------------------------------------------------------------

    weather = weather[[
        "datetime",
        "sw_radiation",
        "temp_2m",
        "cloud_cover",
    ]].copy()

    # ---------------------------------------------------------------
    # Load sensor target + irradiation for the same timestamps
    # ---------------------------------------------------------------

    sensor_path = DATA_DIR / "plant1_hourly.csv"

    if not sensor_path.exists():
        raise FileNotFoundError(
            f"Cannot load Set B target/daytime information because:\n"
            f"{sensor_path}\n"
            "does not exist."
        )

    sensor = pd.read_csv(sensor_path)
    sensor = normalize_dataframe_columns(sensor)

    required_sensor_columns = [
        "datetime",
        "irradiation",
        "ac_power",
    ]

    missing_sensor = [
        column
        for column in required_sensor_columns
        if column not in sensor.columns
    ]

    if missing_sensor:
        raise ValueError(
            "\nplant1_hourly.csv is missing required Set B columns:\n"
            f"{missing_sensor}\n\n"
            f"Columns found:\n{list(sensor.columns)}"
        )

    sensor["datetime"] = pd.to_datetime(
        sensor["datetime"],
        errors="coerce",
    )

    if sensor["datetime"].isna().any():
        bad_rows = int(sensor["datetime"].isna().sum())
        raise ValueError(
            f"plant1_hourly.csv contains {bad_rows} invalid datetime value(s)."
        )

    sensor_daytime = sensor[
        [
            "datetime",
            "irradiation",
            "ac_power",
        ]
    ].copy()

    # Check that the sensor table has one row per timestamp.
    duplicate_sensor_times = int(
        sensor_daytime["datetime"].duplicated().sum()
    )

    if duplicate_sensor_times > 0:
        raise ValueError(
            "plant1_hourly.csv contains duplicate datetime values. "
            "Set B requires one sensor row per datetime."
        )

    # ---------------------------------------------------------------
    # Merge weather with AC power + irradiation
    # ---------------------------------------------------------------

    weather = weather.merge(
        sensor_daytime,
        on="datetime",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Check merge
    # ---------------------------------------------------------------

    missing_target = int(
        weather["ac_power"].isna().sum()
    )

    missing_irradiation = int(
        weather["irradiation"].isna().sum()
    )

    if missing_target > 0:
        print(
            "\nWARNING:"
            f" {missing_target} Set B rows have no matching "
            "AC power target."
        )

    if missing_irradiation > 0:
        print(
            "\nWARNING:"
            f" {missing_irradiation} Set B rows have no matching "
            "sensor irradiation."
        )

    # ---------------------------------------------------------------
    # Add time features
    # ---------------------------------------------------------------

    weather = add_time_features(weather)

    weather = weather.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return weather


# =====================================================================
# CHRONOLOGICAL SPLIT
# =====================================================================

def chronological_split(df):
    """
    Split data strictly by date.

    Training:
        15 May - 10 June 2020

    Testing:
        11 June - 17 June 2020

    No shuffling.
    """

    train = df[
        (df["datetime"] >= TRAIN_START)
        &
        (df["datetime"] <= TRAIN_END)
    ].copy()

    test = df[
        (df["datetime"] >= TEST_START)
        &
        (df["datetime"] <= TEST_END)
    ].copy()

    train = train.sort_values(
        "datetime"
    ).reset_index(drop=True)

    test = test.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return train, test


# =====================================================================
# BUILD DESIGN MATRIX
# =====================================================================

def build_design_matrix(
    train_df,
    test_df,
    cont_cols
):
    """
    Build X for linear regression.

    Steps:

    1. Calculate mean from TRAINING data only.
    2. Calculate standard deviation from TRAINING data only.
    3. Scale training features.
    4. Apply SAME statistics to test features.
    5. Add sin_hour and cos_hour.
    6. Add intercept column x0 = 1.

    Final feature order:

        [1,
         feature_1,
         feature_2,
         feature_3,
         sin_hour,
         cos_hour]

    Therefore d = 5 and X has 6 columns.
    """

    # ---------------------------------------------------------------
    # Verify columns
    # ---------------------------------------------------------------

    missing_train = [
        column
        for column in cont_cols
        if column not in train_df.columns
    ]

    missing_test = [
        column
        for column in cont_cols
        if column not in test_df.columns
    ]

    if missing_train:
        raise KeyError(
            "Missing Set feature(s) from training data: "
            f"{missing_train}\n\n"
            f"Available columns:\n{list(train_df.columns)}"
        )

    if missing_test:
        raise KeyError(
            "Missing Set feature(s) from test data: "
            f"{missing_test}\n\n"
            f"Available columns:\n{list(test_df.columns)}"
        )

    # ---------------------------------------------------------------
    # Training statistics ONLY
    # ---------------------------------------------------------------

    means = train_df[
        cont_cols
    ].mean()

    stds = train_df[
        cont_cols
    ].std(ddof=0)

    # Prevent division by zero
    zero_std = stds[
        stds == 0
    ]

    if len(zero_std) > 0:
        raise ValueError(
            "The following feature(s) have zero training "
            f"standard deviation:\n{list(zero_std.index)}"
        )

    # ---------------------------------------------------------------
    # Build matrix
    # ---------------------------------------------------------------

    def make_X(df):

        scaled = (
            df[cont_cols] - means
        ) / stds

        time_features = df[
            [
                "sin_hour",
                "cos_hour",
            ]
        ].copy()

        # Reset both indexes so concatenation is safe
        scaled = scaled.reset_index(drop=True)

        time_features = time_features.reset_index(
            drop=True
        )

        features = pd.concat(
            [
                scaled,
                time_features,
            ],
            axis=1,
        )

        X = np.column_stack(
            [
                np.ones(len(features)),
                features.to_numpy(dtype=float),
            ]
        )

        return X

    X_train = make_X(train_df)
    X_test = make_X(test_df)

    # ---------------------------------------------------------------
    # Targets
    # ---------------------------------------------------------------

    if "ac_power" not in train_df.columns:
        raise ValueError(
            "Training data does not contain ac_power."
        )

    if "ac_power" not in test_df.columns:
        raise ValueError(
            "Test data does not contain ac_power."
        )

    y_train = train_df[
        "ac_power"
    ].to_numpy(dtype=float)

    y_test = test_df[
        "ac_power"
    ].to_numpy(dtype=float)

    # ---------------------------------------------------------------
    # Final numeric/finite-data check
    # ---------------------------------------------------------------
    # Linear regression cannot train correctly with NaN or infinite
    # values. Raise a clear error instead of producing misleading
    # results.
    if not np.all(np.isfinite(X_train)):
        raise ValueError(
            "X_train contains NaN or infinite values. "
            "Check the input CSV and feature columns."
        )

    if not np.all(np.isfinite(X_test)):
        raise ValueError(
            "X_test contains NaN or infinite values. "
            "Check the input CSV and feature columns."
        )

    if not np.all(np.isfinite(y_train)):
        raise ValueError(
            "y_train contains NaN or infinite values. "
            "Check the ac_power column in the training data."
        )

    if not np.all(np.isfinite(y_test)):
        raise ValueError(
            "y_test contains NaN or infinite values. "
            "Check the ac_power column in the test data."
        )

    # ---------------------------------------------------------------
    # Save scaler information
    # ---------------------------------------------------------------

    scaler = {
        "cols": list(cont_cols),
        "mean": means.to_dict(),
        "std": stds.to_dict(),
    }

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
    )


# =====================================================================
# FEATURE NAMES
# =====================================================================

def feature_names_for(cont_cols):

    return [
        "intercept",
        *cont_cols,
        "sin_hour",
        "cos_hour",
    ]


# =====================================================================
# TASK 4.7 LEARNING RATE EXPERIMENTS
# =====================================================================

def learning_rate_experiments(
    X_train,
    y_train
):
    """
    Task 4.7.

    Batch GD:
        alpha = 10^-5
        alpha = 10^-4
        alpha = 10^-3

        500 iterations

    SGD:
        alpha = 10^-4
        alpha = 10^-3
        alpha = 10^-2

        50 epochs

    Set A is used for selecting alpha.
    """

    # ================================================================
    # BATCH GD
    # ================================================================

    batch_alphas = [
        1e-5,
        1e-4,
        1e-3,
    ]

    batch_iters = 500

    batch_histories = {}

    print(
        "\n=================================================="
    )
    print(
        "Task 4.7 - Batch GD learning-rate experiment"
    )
    print(
        "=================================================="
    )

    for alpha in batch_alphas:

        print(
            f"Running Batch GD with alpha={alpha}..."
        )

        _, history = fit_batch_gd(
            X_train,
            y_train,
            alpha=alpha,
            n_iters=batch_iters,
        )

        batch_histories[
            alpha
        ] = history

        print(
            f"  Final cost = {history[-1]:.6e}"
        )

    # ---------------------------------------------------------------
    # Plot Batch GD
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for alpha, history in batch_histories.items():

        plt.plot(
            range(
                1,
                batch_iters + 1
            ),
            history,
            label=f"alpha={alpha}",
        )

    plt.xlabel(
        "Iteration"
    )

    plt.ylabel(
        "J(theta)"
    )

    plt.yscale(
        "log"
    )

    plt.title(
        "Task 4.7 - Batch GD Learning Rates (Set A)"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    batch_plot = (
        FIG_DIR /
        "4_7_batch_gd_learning_rates.png"
    )

    plt.savefig(
        batch_plot,
        dpi=150
    )

    plt.close()

    print(
        f"Saved {batch_plot}"
    )

    # ================================================================
    # SGD
    # ================================================================

    sgd_alphas = [
        1e-4,
        1e-3,
        1e-2,
    ]

    sgd_epochs = 50

    sgd_histories = {}

    print(
        "\n=================================================="
    )
    print(
        "Task 4.7 - SGD learning-rate experiment"
    )
    print(
        "=================================================="
    )

    for alpha in sgd_alphas:

        print(
            f"Running SGD with alpha={alpha}..."
        )

        _, history = fit_sgd(
            X_train,
            y_train,
            alpha=alpha,
            n_epochs=sgd_epochs,
        )

        sgd_histories[
            alpha
        ] = history

        print(
            f"  Final cost = {history[-1]:.6e}"
        )

    # ---------------------------------------------------------------
    # Plot SGD
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for alpha, history in sgd_histories.items():

        plt.plot(
            range(
                1,
                sgd_epochs + 1
            ),
            history,
            label=f"alpha={alpha}",
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "J(theta)"
    )

    plt.yscale(
        "log"
    )

    plt.title(
        "Task 4.7 - SGD Learning Rates (Set A)"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    sgd_plot = (
        FIG_DIR /
        "4_7_sgd_learning_rates.png"
    )

    plt.savefig(
        sgd_plot,
        dpi=150
    )

    plt.close()

    print(
        f"Saved {sgd_plot}"
    )

    # ================================================================
    # CHOOSE ALPHAS
    # ================================================================

    def choose_alpha(histories):

        valid = {}

        for alpha, history in histories.items():

            if np.all(
                np.isfinite(history)
            ):
                valid[alpha] = history

        if not valid:
            raise RuntimeError(
                "All learning-rate experiments diverged."
            )

        return min(
            valid,
            key=lambda alpha:
            valid[alpha][-1]
        )

    best_batch_alpha = choose_alpha(
        batch_histories
    )

    best_sgd_alpha = choose_alpha(
        sgd_histories
    )

    print(
        "\nSelected Batch GD alpha:"
        f" {best_batch_alpha}"
    )

    print(
        "Selected SGD alpha:"
        f" {best_sgd_alpha}"
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    report = {
        "batch_alphas_tested": batch_alphas,
        "batch_iters": batch_iters,

        "batch_final_costs": {
            str(alpha): float(history[-1])
            for alpha, history
            in batch_histories.items()
        },

        "batch_alpha_chosen": best_batch_alpha,

        "sgd_alphas_tested": sgd_alphas,
        "sgd_epochs": sgd_epochs,

        "sgd_final_costs": {
            str(alpha): float(history[-1])
            for alpha, history
            in sgd_histories.items()
        },

        "sgd_alpha_chosen": best_sgd_alpha,
    }

    return (
        best_batch_alpha,
        batch_iters,
        best_sgd_alpha,
        sgd_epochs,
        report,
    )


# =====================================================================
# FIT + EVALUATE ONE FEATURE SET
# =====================================================================

def fit_and_evaluate_set(
    name,
    train_df,
    test_df,
    cont_cols,
    batch_alpha,
    batch_iters,
    sgd_alpha,
    sgd_epochs,
):
    """
    Fit all three regression methods for one feature set.

    Methods:
        1. Normal equation
        2. Batch GD
        3. SGD

    Then calculate:

        RMSE - all test hours
        RMSE - daytime test hours

    Negative predictions are clipped to zero.
    """

    print(
        "\n=================================================="
    )

    print(
        f"Fitting Set {name}"
    )

    print(
        "Features:"
    )

    print(
        cont_cols
    )

    print(
        "=================================================="
    )

    # ---------------------------------------------------------------
    # Build X
    # ---------------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
    ) = build_design_matrix(
        train_df,
        test_df,
        cont_cols,
    )

    names = feature_names_for(
        cont_cols
    )

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"X_test shape:  {X_test.shape}"
    )

    # ---------------------------------------------------------------
    # Normal equation
    # ---------------------------------------------------------------

    print(
        "\nTraining Normal Equation..."
    )

    theta_normal = fit_normal(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------------
    # Batch GD
    # ---------------------------------------------------------------

    print(
        "Training Batch GD..."
    )

    theta_bgd, bgd_hist = fit_batch_gd(
        X_train,
        y_train,
        alpha=batch_alpha,
        n_iters=batch_iters,
    )

    # ---------------------------------------------------------------
    # SGD
    # ---------------------------------------------------------------

    print(
        "Training SGD..."
    )

    theta_sgd, sgd_hist = fit_sgd(
        X_train,
        y_train,
        alpha=sgd_alpha,
        n_epochs=sgd_epochs,
    )

    # ---------------------------------------------------------------
    # Compare Batch GD and Normal Equation
    # ---------------------------------------------------------------

    max_diff = float(
        np.max(
            np.abs(
                theta_bgd -
                theta_normal
            )
        )
    )

    # ---------------------------------------------------------------
    # Daytime mask
    # ---------------------------------------------------------------

    if "irradiation" not in test_df.columns:

        raise RuntimeError(
            "irradiation column is required to define "
            "daytime test hours."
        )

    daytime_mask = (
        test_df[
            "irradiation"
        ].to_numpy(dtype=float)
        > 0
    )

    daytime_count = int(
        np.sum(daytime_mask)
    )

    print(
        f"Daytime test rows: {daytime_count}"
    )

    if daytime_count == 0:

        raise RuntimeError(
            "No daytime test rows found."
        )

    # ---------------------------------------------------------------
    # Evaluate all three models
    # ---------------------------------------------------------------

    results = {}

    model_list = [
        (
            "normal_equation",
            theta_normal,
        ),
        (
            "batch_gd",
            theta_bgd,
        ),
        (
            "sgd",
            theta_sgd,
        ),
    ]

    for solver_name, theta in model_list:

        # Prediction
        y_pred = hypothesis(
            X_test,
            theta,
        )

        # Assignment requirement:
        # negative AC power is physically impossible
        y_pred_clipped = np.clip(
            y_pred,
            a_min=0,
            a_max=None,
        )

        # RMSE for all test hours
        rmse_all = rmse(
            y_test,
            y_pred_clipped,
        )

        # RMSE for daytime only
        rmse_day = rmse(
            y_test[daytime_mask],
            y_pred_clipped[
                daytime_mask
            ],
        )

        results[
            solver_name
        ] = {
            "theta": theta.tolist(),

            "rmse_all_hours": float(
                rmse_all
            ),

            "rmse_daytime": float(
                rmse_day
            ),

            "y_pred_clipped":
                y_pred_clipped,
        }

        print(
            f"\n{solver_name}:"
        )

        print(
            f"  All-hours RMSE = "
            f"{rmse_all:.3f} kW"
        )

        print(
            f"  Daytime RMSE   = "
            f"{rmse_day:.3f} kW"
        )

    # ---------------------------------------------------------------
    # Save weights
    # ---------------------------------------------------------------

    weight_path = (
        WEIGHTS_DIR /
        f"set_{name}_weights.npz"
    )

    np.savez(
        weight_path,

        theta_normal=theta_normal,

        theta_batch_gd=theta_bgd,

        theta_sgd=theta_sgd,

        feature_names=np.array(
            names,
            dtype=str
        ),

        scaler_mean=np.array(
            [
                scaler["mean"][column]
                for column in cont_cols
            ],
            dtype=float,
        ),

        scaler_std=np.array(
            [
                scaler["std"][column]
                for column in cont_cols
            ],
            dtype=float,
        ),

        cont_cols=np.array(
            cont_cols,
            dtype=str
        ),

        # Metadata used by the frontend to document the saved model.
        target_unit=np.array("kW", dtype=str),
        model_type=np.array("linear_regression", dtype=str),
        training_split=np.array("2020-05-15 through 2020-06-10", dtype=str),
        testing_split=np.array("2020-06-11 through 2020-06-17", dtype=str),
    )

    print(
        f"\nSaved weights: {weight_path}"
    )

    return {
        "feature_names": names,

        "scaler": scaler,

        "results": results,

        "bgd_cost_history":
            bgd_hist,

        "sgd_cost_history":
            sgd_hist,

        "max_abs_diff_bgd_vs_normal":
            max_diff,

        "batch_alpha":
            batch_alpha,

        "batch_iters":
            batch_iters,

        "sgd_alpha":
            sgd_alpha,

        "sgd_epochs":
            sgd_epochs,

        "X_test":
            X_test,

        "y_test":
            y_test,

        "test_df":
            test_df,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "\n=================================================="
    )

    print(
        "Solar Power Prediction - Task 4"
    )

    print(
        "=================================================="
    )

    # ================================================================
    # SET A
    # ================================================================

    set_a = load_set_a()

    train_a, test_a = chronological_split(
        set_a
    )

    print(
        f"\nSet A: "
        f"{len(train_a)} train rows, "
        f"{len(test_a)} test rows"
    )

    # ---------------------------------------------------------------
    # Chronological correctness checks
    # ---------------------------------------------------------------

    assert (
        train_a["datetime"].max()
        <= TRAIN_END
    )

    assert (
        test_a["datetime"].min()
        >= TEST_START
    )

    assert (
        train_a["datetime"].max()
        <
        test_a["datetime"].min()
    )

    print(
        "Chronological split verified: "
        "no overlap between train and test."
    )

    # ================================================================
    # TASK 4.7 - LEARNING RATE
    # ================================================================

    (
        Xa_train,
        ya_train,
        _,
        _,
        _,
    ) = build_design_matrix(
        train_a,
        test_a,
        SET_A_COLS,
    )

    (
        best_batch_alpha,
        batch_iters,
        best_sgd_alpha,
        sgd_epochs,
        lr_report,
    ) = learning_rate_experiments(
        Xa_train,
        ya_train,
    )

    # ================================================================
    # SET A - ALL THREE METHODS
    # ================================================================

    out_a = fit_and_evaluate_set(
        "A",
        train_a,
        test_a,
        SET_A_COLS,
        best_batch_alpha,
        FINAL_BATCH_ITERS,
        best_sgd_alpha,
        sgd_epochs,
    )

    # ================================================================
    # CORRECTNESS CHECK
    # ================================================================

    bgd_hist = out_a[
        "bgd_cost_history"
    ]

    monotone = bool(
        np.all(
            np.diff(bgd_hist)
            <= 1e-6
        )
    )

    print(
        "\n=================================================="
    )

    print(
        "Correctness checks - Set A"
    )

    print(
        "=================================================="
    )

    print(
        "Batch GD cost decreases every "
        f"iteration: {monotone}"
    )

    print(
        "Max |theta_bgd - theta_normal|: "
        f"{out_a['max_abs_diff_bgd_vs_normal']:.6f}"
    )

    # ================================================================
    # SET B
    # ================================================================

    set_b_data = load_set_b()

    out_b = None

    if set_b_data is None:

        print(
            "\n=================================================="
        )

        print(
            "Set B skipped"
        )

        print(
            "=================================================="
        )

        print(
            "No Open-Meteo file was found."
        )

        print(
            "Run fetch_weather.py first."
        )

    else:

        train_b, test_b = chronological_split(
            set_b_data
        )

        print(
            f"\nSet B: "
            f"{len(train_b)} train rows, "
            f"{len(test_b)} test rows"
        )

        # ------------------------------------------------------------
        # Verify target exists
        # ------------------------------------------------------------

        if "ac_power" not in train_b.columns:
            raise ValueError(
                "Set B does not contain ac_power. "
                "The Set B loader should merge ac_power "
                "from plant1_hourly.csv."
            )

        # ------------------------------------------------------------
        # Final checks
        # ------------------------------------------------------------

        required_b = (
            SET_B_COLS
            +
            [
                "ac_power",
                "irradiation",
                "sin_hour",
                "cos_hour",
            ]
        )

        missing_b = [
            column
            for column in required_b
            if column not in train_b.columns
            or column not in test_b.columns
        ]

        if missing_b:

            raise ValueError(
                "\nSet B is missing required columns:\n"
                f"{missing_b}\n\n"
                f"Set B training columns:\n"
                f"{list(train_b.columns)}"
            )

        # ------------------------------------------------------------
        # Fit Set B
        # ------------------------------------------------------------

        out_b = fit_and_evaluate_set(
            "B",
            train_b,
            test_b,
            SET_B_COLS,
            best_batch_alpha,
            FINAL_BATCH_ITERS,
            best_sgd_alpha,
            sgd_epochs,
        )

        print(
            "\nMax |theta_bgd - theta_normal| "
            f"(Set B): "
            f"{out_b['max_abs_diff_bgd_vs_normal']:.6f}"
        )

    # ================================================================
    # TABLE 2 - TEST RMSE
    # ================================================================

    rows = []

    solver_information = [
        (
            "normal_equation",
            "Normal equation",
        ),
        (
            "batch_gd",
            (
                "Batch GD "
                f"(alpha={best_batch_alpha}, "
                f"iters={FINAL_BATCH_ITERS})"
            ),
        ),
        (
            "sgd",
            (
                "Stochastic GD "
                f"(alpha={best_sgd_alpha}, "
                f"epochs={sgd_epochs})"
            ),
        ),
    ]

    # ---------------------------------------------------------------
    # Set A
    # ---------------------------------------------------------------

    for solver_key, solver_label in solver_information:

        result = out_a[
            "results"
        ][solver_key]

        rows.append(
            {
                "Solver": solver_label,
                "Features": "Set A",
                "All hours RMSE (kW)": round(
                    result[
                        "rmse_all_hours"
                    ],
                    3,
                ),
                "Daytime RMSE (kW)": round(
                    result[
                        "rmse_daytime"
                    ],
                    3,
                ),
            }
        )

    # ---------------------------------------------------------------
    # Set B
    # ---------------------------------------------------------------

    if out_b is not None:

        for solver_key, solver_label in solver_information:

            result = out_b[
                "results"
            ][solver_key]

            rows.append(
                {
                    "Solver": solver_label,
                    "Features": "Set B",
                    "All hours RMSE (kW)": round(
                        result[
                            "rmse_all_hours"
                        ],
                        3,
                    ),
                    "Daytime RMSE (kW)": round(
                        result[
                            "rmse_daytime"
                        ],
                        3,
                    ),
                }
            )

    table2 = pd.DataFrame(rows)

    table2_path = (
        RESULTS_DIR /
        "table2_test_rmse.csv"
    )

    table2.to_csv(
        table2_path,
        index=False,
    )

    print(
        "\n=================================================="
    )

    print(
        "Table 2 - Test RMSE"
    )

    print(
        "=================================================="
    )

    print(
        table2.to_string(
            index=False
        )
    )

    print(
        f"\nSaved {table2_path}"
    )

    # ================================================================
    # TABLE 3 - LEARNED THETA SET A
    # ================================================================

    names_a = out_a[
        "feature_names"
    ]

    table3 = pd.DataFrame(
        {
            "feature": names_a,

            "normal_equation":
                out_a[
                    "results"
                ][
                    "normal_equation"
                ][
                    "theta"
                ],

            "batch_gd":
                out_a[
                    "results"
                ][
                    "batch_gd"
                ][
                    "theta"
                ],

            "sgd":
                out_a[
                    "results"
                ][
                    "sgd"
                ][
                    "theta"
                ],
        }
    )

    table3_path = (
        RESULTS_DIR /
        "table3_theta_set_a.csv"
    )

    table3.to_csv(
        table3_path,
        index=False,
    )

    print(
        "\n=================================================="
    )

    print(
        "Table 3 - Learned theta (Set A)"
    )

    print(
        "=================================================="
    )

    print(
        table3.to_string(
            index=False
        )
    )

    print(
        "\nMax |theta_bgd - theta_normal|: "
        f"{out_a['max_abs_diff_bgd_vs_normal']:.6f}"
    )

    print(
        f"\nSaved {table3_path}"
    )

    # ================================================================
    # FIGURE 5 - ACTUAL VS PREDICTED
    # ================================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        test_a["datetime"],
        out_a["y_test"],
        label="Actual",
        linewidth=1.5,
    )

    plt.plot(
        test_a["datetime"],
        out_a[
            "results"
        ][
            "normal_equation"
        ][
            "y_pred_clipped"
        ],
        label="Predicted - Set A",
        linewidth=1.2,
        alpha=0.85,
    )

    if out_b is not None:

        # Set B test dates should match Set A,
        # but we use Set B's own datetime column.
        plt.plot(
            out_b[
                "test_df"
            ]["datetime"],
            out_b[
                "results"
            ][
                "normal_equation"
            ][
                "y_pred_clipped"
            ],
            label="Predicted - Set B",
            linewidth=1.2,
            alpha=0.85,
        )

    plt.xlabel(
        "Datetime"
    )

    plt.ylabel(
        "AC Power (kW)"
    )

    plt.title(
        "Actual vs Predicted AC Power - Test Week"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    actual_prediction_path = (
        FIG_DIR /
        "5_actual_vs_predicted_test_week.png"
    )

    plt.savefig(
        actual_prediction_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved {actual_prediction_path}"
    )

    # ================================================================
    # FIGURE 6 - RESIDUALS VS HOUR
    # ================================================================

    residuals = (
        out_a["y_test"]
        -
        out_a[
            "results"
        ][
            "normal_equation"
        ][
            "y_pred_clipped"
        ]
    )

    hours = test_a[
        "hour"
    ].to_numpy()

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        hours,
        residuals,
        alpha=0.5,
        s=14,
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1,
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Residual = Actual - Predicted (kW)"
    )

    plt.title(
        "Task 5.5 - Residuals vs Hour of Day "
        "(Set A, Normal Equation)"
    )

    plt.xticks(
        range(24)
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    residual_path = (
        FIG_DIR /
        "6_residuals_vs_hour.png"
    )

    plt.savefig(
        residual_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved {residual_path}"
    )

    # ================================================================
    # RESIDUAL STATISTICS
    # ================================================================

    residual_df = pd.DataFrame(
        {
            "hour": hours,
            "residual": residuals,
        }
    )

    resid_by_hour = (
        residual_df
        .groupby("hour")[
            "residual"
        ]
        .agg(
            [
                "mean",
                "std",
                "count",
            ]
        )
    )

    print(
        "\nResidual mean/std by hour of day:"
    )

    print(
        resid_by_hour.to_string()
    )

    # ================================================================
    # TASK 5.1 - LARGEST SET A WEIGHT
    # ================================================================

    theta_a_normal = np.asarray(
        out_a[
            "results"
        ][
            "normal_equation"
        ][
            "theta"
        ],
        dtype=float,
    )

    # Ignore intercept when determining the largest feature weight
    feature_weights = theta_a_normal[1:]

    largest_index = int(
        np.argmax(
            np.abs(feature_weights)
        )
    )

    largest_feature = names_a[
        largest_index + 1
    ]

    largest_weight = feature_weights[
        largest_index
    ]

    print(
        "\n=================================================="
    )

    print(
        "Task 5.1"
    )

    print(
        "=================================================="
    )

    print(
        "Largest absolute feature weight:"
    )

    print(
        f"  {largest_feature} = "
        f"{largest_weight:.6f}"
    )

    # ================================================================
    # TASK 5.2 - SET B VS SET A
    # ================================================================

    summary = {
        "train_rows": int(
            len(train_a)
        ),

        "test_rows": int(
            len(test_a)
        ),

        "set_b_available":
            out_b is not None,

        "learning_rate_experiments":
            lr_report,

        "set_a_max_abs_diff_bgd_vs_normal":
            out_a[
                "max_abs_diff_bgd_vs_normal"
            ],

        "set_a_batch_gd_monotone_decrease":
            monotone,

        "peak_hourly_ac_power_train_kw":
            float(
                train_a[
                    "ac_power"
                ].max()
            ),

        "largest_set_a_feature_weight":
            largest_feature,

        "largest_set_a_feature_weight_value":
            float(largest_weight),

        "residual_by_hour_mean":
            resid_by_hour[
                "mean"
            ].round(3).to_dict(),
    }

    if out_b is not None:

        rmse_a_day = out_a[
            "results"
        ][
            "normal_equation"
        ][
            "rmse_daytime"
        ]

        rmse_b_day = out_b[
            "results"
        ][
            "normal_equation"
        ][
            "rmse_daytime"
        ]

        peak = summary[
            "peak_hourly_ac_power_train_kw"
        ]

        difference_kw = (
            rmse_b_day -
            rmse_a_day
        )

        difference_pct = (
            100.0 *
            difference_kw /
            peak
        )

        summary[
            "set_a_daytime_rmse_kw"
        ] = float(
            rmse_a_day
        )

        summary[
            "set_b_daytime_rmse_kw"
        ] = float(
            rmse_b_day
        )

        summary[
            "set_b_minus_set_a_daytime_rmse_kw"
        ] = float(
            difference_kw
        )

        summary[
            "set_b_vs_set_a_daytime_rmse_diff_pct_of_peak"
        ] = float(
            difference_pct
        )

        summary[
            "set_b_max_abs_diff_bgd_vs_normal"
        ] = out_b[
            "max_abs_diff_bgd_vs_normal"
        ]

        print(
            "\n=================================================="
        )

        print(
            "Task 5.2 - Set B vs Set A"
        )

        print(
            "=================================================="
        )

        print(
            f"Set A daytime RMSE: "
            f"{rmse_a_day:.3f} kW"
        )

        print(
            f"Set B daytime RMSE: "
            f"{rmse_b_day:.3f} kW"
        )

        print(
            f"Difference: "
            f"{difference_kw:.3f} kW"
        )

        print(
            f"Difference as % of peak: "
            f"{difference_pct:.2f}%"
        )

    # ================================================================
    # SAVE SUMMARY JSON
    # ================================================================

    summary_path = (
        RESULTS_DIR /
        "train_eval_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    print(
        f"\nSaved {summary_path}"
    )

    # ================================================================
    # FINAL SUMMARY
    # ================================================================

    print(
        "\n=================================================="
    )

    print(
        "TASK 4 COMPLETE"
    )

    print(
        "=================================================="
    )

    print(
        f"Training rows: {len(train_a)}"
    )

    print(
        f"Testing rows:  {len(test_a)}"
    )

    print(
        f"Selected Batch GD alpha: "
        f"{best_batch_alpha}"
    )

    print(
        f"Selected SGD alpha: "
        f"{best_sgd_alpha}"
    )

    print(
        f"Final Batch GD iterations: "
        f"{FINAL_BATCH_ITERS}"
    )

    print(
        f"SGD epochs: "
        f"{sgd_epochs}"
    )

    print(
        f"Batch GD monotone: "
        f"{monotone}"
    )

    print(
        f"Set A theta difference: "
        f"{out_a['max_abs_diff_bgd_vs_normal']:.6f}"
    )

    if out_b is not None:

        print(
            f"Set B theta difference: "
            f"{out_b['max_abs_diff_bgd_vs_normal']:.6f}"
        )

    print(
        "\nResults saved in:"
    )

    print(
        f"  {RESULTS_DIR}"
    )

    print(
        f"  {FIG_DIR}"
    )

    print(
        f"  {WEIGHTS_DIR}"
    )


# =====================================================================
# RUN
# =====================================================================

if __name__ == "__main__":
    main()