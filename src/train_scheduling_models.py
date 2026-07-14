import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =========================================================
# PATHS AND SETTINGS
# =========================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

METRICS_PATH = MODEL_DIR / "scheduling_metrics.txt"
METRICS_CSV_PATH = MODEL_DIR / "scheduling_metrics.csv"
IMPORTANCE_PATH = MODEL_DIR / "scheduling_feature_importance.csv"

# A real schedule-delay model is trained only if the timestamps contain
# enough non-zero variation. The Alibaba 2020 job/task timestamps often do not.
MIN_POSITIVE_DELAY_ROWS = 100
MIN_CLASS_ROWS = 50

# The readiness classifier uses the 90th percentile of ready_delay_sec.
READY_DELAY_QUANTILE = 0.90


# =========================================================
# ALIBABA TABLE SCHEMAS
# =========================================================

JOB_COLUMNS = [
    "job_name",
    "inst_id",
    "user",
    "status",
    "start_time",
    "end_time",
]

TASK_COLUMNS = [
    "job_name",
    "task_name",
    "inst_num",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "plan_gpu",
    "gpu_type",
]


# =========================================================
# FILE AND DATA HELPERS
# =========================================================

def find_file(filename: str) -> Path:
    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} anywhere inside {DATA_DIR.resolve()}"
        )

    print(f"Found {filename}: {matches[0]}")
    return matches[0]


def read_alibaba_csv(
    path: Path,
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    Read a CSV that may contain headers or may be headerless.
    """

    print(f"Reading {path}...")

    first_row = pd.read_csv(
        path,
        nrows=1,
        header=None,
        low_memory=False,
    )
    first_value = str(first_row.iloc[0, 0]).strip()

    if first_value == expected_columns[0]:
        dataframe = pd.read_csv(path, low_memory=False)
    else:
        dataframe = pd.read_csv(
            path,
            names=expected_columns,
            header=None,
            low_memory=False,
        )

    missing = set(expected_columns) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}\n"
            f"Columns found: {dataframe.columns.tolist()}"
        )

    return dataframe[expected_columns]


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    minimum_denominator: float = 1e-6,
) -> pd.Series:
    return numerator / denominator.clip(lower=minimum_denominator)


def combine_unique_strings(
    values: pd.Series,
    missing_value: str = "UNKNOWN",
) -> str:
    """
    Safely combine unique categorical values without mixed-type sort errors.
    """

    cleaned = {
        str(value)
        for value in values.tolist()
        if pd.notna(value)
    }

    cleaned.discard("")
    cleaned.discard("nan")
    cleaned.discard("None")

    if not cleaned:
        return missing_value

    return "|".join(sorted(cleaned, key=str))


# =========================================================
# DATA PREPARATION
# =========================================================

def load_and_prepare_data() -> tuple[pd.DataFrame, bool, float | None]:
    job_path = find_file("pai_job_table.csv")
    task_path = find_file("pai_task_table.csv")

    jobs = read_alibaba_csv(job_path, JOB_COLUMNS)
    tasks = read_alibaba_csv(task_path, TASK_COLUMNS)

    print(f"Raw job rows:  {len(jobs):,}")
    print(f"Raw task rows: {len(tasks):,}")

    # -----------------------------------------------------
    # JOB TABLE
    # -----------------------------------------------------

    for column in ["start_time", "end_time"]:
        jobs[column] = pd.to_numeric(
            jobs[column],
            errors="coerce",
        )

    jobs = jobs[jobs["start_time"].notna()].copy()

    jobs["submission_hour"] = (
        (jobs["start_time"] % 86400) // 3600
    ).astype("int16")

    jobs["submission_day"] = (
        (jobs["start_time"] // 86400) % 7
    ).astype("int16")

    jobs["submission_hour_sin"] = np.sin(
        2 * np.pi * jobs["submission_hour"] / 24
    )
    jobs["submission_hour_cos"] = np.cos(
        2 * np.pi * jobs["submission_hour"] / 24
    )
    jobs["submission_day_sin"] = np.sin(
        2 * np.pi * jobs["submission_day"] / 7
    )
    jobs["submission_day_cos"] = np.cos(
        2 * np.pi * jobs["submission_day"] / 7
    )

    # -----------------------------------------------------
    # TASK TABLE
    # -----------------------------------------------------

    numeric_task_columns = [
        "inst_num",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
        "plan_gpu",
    ]

    for column in numeric_task_columns:
        tasks[column] = pd.to_numeric(
            tasks[column],
            errors="coerce",
        )

    tasks["inst_num"] = (
        tasks["inst_num"]
        .fillna(1)
        .clip(lower=1)
    )
    tasks["plan_cpu"] = (
        tasks["plan_cpu"]
        .fillna(0)
        .clip(lower=0)
    )
    tasks["plan_mem"] = (
        tasks["plan_mem"]
        .fillna(0)
        .clip(lower=0)
    )
    tasks["plan_gpu"] = (
        tasks["plan_gpu"]
        .fillna(0)
        .clip(lower=0)
    )

    tasks["task_name"] = (
        tasks["task_name"]
        .fillna("UNKNOWN")
        .map(str)
    )
    tasks["gpu_type"] = (
        tasks["gpu_type"]
        .fillna("NONE")
        .map(str)
    )

    task_name_lower = tasks["task_name"].str.lower()

    tasks["worker_instances"] = np.where(
        task_name_lower.str.contains("worker", na=False),
        tasks["inst_num"],
        0,
    )

    # Non-capturing groups avoid the pandas regex warning.
    tasks["ps_instances"] = np.where(
        task_name_lower.str.contains(
            r"(?:^|[-_])ps(?:$|[-_0-9])|parameter",
            regex=True,
            na=False,
        ),
        tasks["inst_num"],
        0,
    )

    tasks["chief_instances"] = np.where(
        task_name_lower.str.contains(
            r"chief|master",
            regex=True,
            na=False,
        ),
        tasks["inst_num"],
        0,
    )

    tasks["evaluator_instances"] = np.where(
        task_name_lower.str.contains(
            r"eval|evaluator",
            regex=True,
            na=False,
        ),
        tasks["inst_num"],
        0,
    )

    tasks["requested_cpu_total"] = (
        tasks["plan_cpu"] * tasks["inst_num"]
    )
    tasks["requested_mem_total"] = (
        tasks["plan_mem"] * tasks["inst_num"]
    )
    tasks["requested_gpu_total"] = (
        tasks["plan_gpu"] * tasks["inst_num"]
    )

    numeric_summary = (
        tasks.groupby("job_name", as_index=False)
        .agg(
            earliest_task_start=("start_time", "min"),
            latest_task_start=("start_time", "max"),
            task_count=("task_name", "count"),
            unique_task_type_count=("task_name", "nunique"),
            total_instances=("inst_num", "sum"),
            total_cpu_request=("requested_cpu_total", "sum"),
            total_memory_request=("requested_mem_total", "sum"),
            total_gpu_request=("requested_gpu_total", "sum"),
            max_cpu_per_instance=("plan_cpu", "max"),
            max_memory_per_instance=("plan_mem", "max"),
            max_gpu_per_instance=("plan_gpu", "max"),
            mean_cpu_per_instance=("plan_cpu", "mean"),
            mean_memory_per_instance=("plan_mem", "mean"),
            mean_gpu_per_instance=("plan_gpu", "mean"),
            worker_count=("worker_instances", "sum"),
            ps_count=("ps_instances", "sum"),
            chief_count=("chief_instances", "sum"),
            evaluator_count=("evaluator_instances", "sum"),
        )
    )

    category_summary = (
        tasks.groupby("job_name", as_index=False)
        .agg(
            task_types=(
                "task_name",
                lambda values: combine_unique_strings(
                    values,
                    "UNKNOWN",
                ),
            ),
            gpu_types=(
                "gpu_type",
                lambda values: combine_unique_strings(
                    values,
                    "NONE",
                ),
            ),
        )
    )

    task_summary = numeric_summary.merge(
        category_summary,
        on="job_name",
        how="left",
    )

    # -----------------------------------------------------
    # JOIN AND DERIVE TARGETS
    # -----------------------------------------------------

    dataset = jobs.merge(
        task_summary,
        on="job_name",
        how="inner",
    )

    # In this release, job.start_time and earliest task start are usually
    # identical. We compute the field but train it only if it has variation.
    dataset["schedule_delay_sec"] = (
        dataset["earliest_task_start"]
        - dataset["start_time"]
    )

    # Readiness delay measures the spread between first and last task launch.
    dataset["ready_delay_sec"] = (
        dataset["latest_task_start"]
        - dataset["earliest_task_start"]
    )

    dataset["requested_cpu_cores"] = (
        dataset["total_cpu_request"] / 100.0
    )
    dataset["requested_gpu_count"] = (
        dataset["total_gpu_request"] / 100.0
    )
    dataset["max_cpu_cores_per_instance"] = (
        dataset["max_cpu_per_instance"] / 100.0
    )
    dataset["max_gpu_count_per_instance"] = (
        dataset["max_gpu_per_instance"] / 100.0
    )
    dataset["mean_cpu_cores_per_instance"] = (
        dataset["mean_cpu_per_instance"] / 100.0
    )
    dataset["mean_gpu_count_per_instance"] = (
        dataset["mean_gpu_per_instance"] / 100.0
    )

    dataset["cpu_per_gpu"] = safe_divide(
        dataset["requested_cpu_cores"],
        dataset["requested_gpu_count"],
        0.01,
    )
    dataset["memory_per_gpu"] = safe_divide(
        dataset["total_memory_request"],
        dataset["requested_gpu_count"],
        0.01,
    )
    dataset["cpu_per_instance"] = safe_divide(
        dataset["requested_cpu_cores"],
        dataset["total_instances"],
        1,
    )
    dataset["memory_per_instance"] = safe_divide(
        dataset["total_memory_request"],
        dataset["total_instances"],
        1,
    )
    dataset["gpu_per_instance"] = safe_divide(
        dataset["requested_gpu_count"],
        dataset["total_instances"],
        1,
    )
    dataset["worker_fraction"] = safe_divide(
        dataset["worker_count"],
        dataset["total_instances"],
        1,
    )
    dataset["tasks_per_instance"] = safe_divide(
        dataset["task_count"],
        dataset["total_instances"],
        1,
    )

    dataset["is_multi_gpu"] = (
        dataset["requested_gpu_count"] > 1
    ).astype("int8")
    dataset["is_distributed"] = (
        dataset["total_instances"] > 1
    ).astype("int8")
    dataset["has_parameter_server"] = (
        dataset["ps_count"] > 0
    ).astype("int8")
    dataset["has_chief"] = (
        dataset["chief_count"] > 0
    ).astype("int8")
    dataset["has_evaluator"] = (
        dataset["evaluator_count"] > 0
    ).astype("int8")

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    dataset = dataset.dropna(
        subset=[
            "schedule_delay_sec",
            "ready_delay_sec",
        ]
    ).copy()

    dataset = dataset[
        (dataset["schedule_delay_sec"] >= 0)
        & (dataset["ready_delay_sec"] >= 0)
    ].copy()

    dataset = dataset.reset_index(drop=True)

    dataset["log_schedule_delay_sec"] = np.log1p(
        dataset["schedule_delay_sec"]
    )
    dataset["log_ready_delay_sec"] = np.log1p(
        dataset["ready_delay_sec"]
    )

    # -----------------------------------------------------
    # DETERMINE WHICH TARGETS ARE VALID
    # -----------------------------------------------------

    positive_schedule_rows = int(
        (dataset["schedule_delay_sec"] > 0).sum()
    )
    unique_schedule_values = int(
        dataset["schedule_delay_sec"].nunique()
    )

    train_schedule_model = (
        positive_schedule_rows >= MIN_POSITIVE_DELAY_ROWS
        and unique_schedule_values > 1
    )

    if train_schedule_model:
        print(
            "Schedule-delay timestamps contain enough variation; "
            "the schedule-delay regressor will be trained."
        )
    else:
        print(
            "Schedule-delay timestamps do not contain enough useful "
            "variation. The schedule-delay regressor will be skipped."
        )

    # Use the 90th percentile as a data-driven readiness-risk threshold.
    ready_threshold = float(
        dataset["ready_delay_sec"].quantile(
            READY_DELAY_QUANTILE
        )
    )

    if not np.isfinite(ready_threshold):
        ready_threshold = 0.0

    dataset["high_ready_delay"] = (
        dataset["ready_delay_sec"] > ready_threshold
    ).astype("int8")

    class_counts = dataset["high_ready_delay"].value_counts()

    classifier_valid = (
        dataset["high_ready_delay"].nunique() == 2
        and class_counts.min() >= MIN_CLASS_ROWS
    )

    if not classifier_valid:
        # Try a threshold based on positive values only.
        positive_ready = dataset.loc[
            dataset["ready_delay_sec"] > 0,
            "ready_delay_sec",
        ]

        if len(positive_ready) >= 2 * MIN_CLASS_ROWS:
            ready_threshold = float(
                positive_ready.quantile(0.75)
            )
            dataset["high_ready_delay"] = (
                dataset["ready_delay_sec"] > ready_threshold
            ).astype("int8")
            class_counts = dataset[
                "high_ready_delay"
            ].value_counts()

            classifier_valid = (
                dataset["high_ready_delay"].nunique() == 2
                and class_counts.min() >= MIN_CLASS_ROWS
            )

    print(f"Usable scheduling rows: {len(dataset):,}")
    print(f"Positive schedule delays: {positive_schedule_rows:,}")
    print(
        f"Readiness-risk threshold: {ready_threshold:.2f} seconds"
    )

    if classifier_valid:
        print(
            "High-readiness-delay rate: "
            f"{dataset['high_ready_delay'].mean() * 100:.2f}%"
        )
    else:
        print(
            "The readiness classifier will be skipped because a stable "
            "two-class target could not be created."
        )

    if len(dataset) < 100:
        raise ValueError(
            "Too few usable rows for scheduling-model training."
        )

    return dataset, train_schedule_model, (
        ready_threshold if classifier_valid else None
    )


# =========================================================
# MODEL FEATURES
# =========================================================

def get_features() -> tuple[list[str], list[str]]:
    features = [
        "task_count",
        "unique_task_type_count",
        "total_instances",
        "requested_cpu_cores",
        "total_memory_request",
        "requested_gpu_count",
        "max_cpu_cores_per_instance",
        "max_memory_per_instance",
        "max_gpu_count_per_instance",
        "mean_cpu_cores_per_instance",
        "mean_memory_per_instance",
        "mean_gpu_count_per_instance",
        "cpu_per_gpu",
        "memory_per_gpu",
        "cpu_per_instance",
        "memory_per_instance",
        "gpu_per_instance",
        "tasks_per_instance",
        "worker_count",
        "ps_count",
        "chief_count",
        "evaluator_count",
        "worker_fraction",
        "is_multi_gpu",
        "is_distributed",
        "has_parameter_server",
        "has_chief",
        "has_evaluator",
        "submission_hour",
        "submission_day",
        "submission_hour_sin",
        "submission_hour_cos",
        "submission_day_sin",
        "submission_day_cos",
        "task_types",
        "gpu_types",
    ]

    categorical_features = [
        "task_types",
        "gpu_types",
    ]

    return features, categorical_features


def clean_model_inputs(
    dataframe: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    X = dataframe[features].copy()

    for column in categorical_features:
        X[column] = (
            X[column]
            .fillna("UNKNOWN")
            .map(str)
        )

    numeric_features = [
        column
        for column in features
        if column not in categorical_features
    ]

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    for column in numeric_features:
        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        X[column] = X[column].fillna(median_value)

    return X


# =========================================================
# TRAINING FUNCTIONS
# =========================================================

def train_regressor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: pd.Series,
    y_test_log: pd.Series,
    categorical_features: list[str],
    target_name: str,
    model_path: Path,
) -> tuple[dict, pd.DataFrame]:
    print(f"\nTraining regressor for: {target_name}")

    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=200,
        early_stopping_rounds=150,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train_log,
        cat_features=categorical_features,
        eval_set=(X_test, y_test_log),
        use_best_model=True,
    )

    prediction_log = model.predict(X_test)
    predictions = np.maximum(
        0,
        np.expm1(prediction_log),
    )
    actual = np.expm1(y_test_log.to_numpy())

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )
    r2 = r2_score(actual, predictions)

    median_value = float(
        np.median(
            np.expm1(y_train_log.to_numpy())
        )
    )
    baseline_predictions = np.full(
        len(actual),
        median_value,
    )

    baseline_mae = mean_absolute_error(
        actual,
        baseline_predictions,
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(
            actual,
            baseline_predictions,
        )
    )
    baseline_r2 = r2_score(
        actual,
        baseline_predictions,
    )

    model.save_model(model_path)
    print(f"Saved model to: {model_path}")

    metrics = {
        "model": target_name,
        "mae_seconds": mae,
        "rmse_seconds": rmse,
        "r2": r2,
        "baseline_mae_seconds": baseline_mae,
        "baseline_rmse_seconds": baseline_rmse,
        "baseline_r2": baseline_r2,
        "best_iteration": model.get_best_iteration(),
    }

    importance = pd.DataFrame(
        {
            "model": target_name,
            "feature": X_train.columns,
            "importance": model.get_feature_importance(),
        }
    )

    return metrics, importance


def train_classifier(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_features: list[str],
    model_path: Path,
) -> tuple[dict, pd.DataFrame, str]:
    print("\nTraining high-readiness-delay classifier...")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    positive_weight = negative_count / max(
        positive_count,
        1,
    )

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=200,
        early_stopping_rounds=150,
        class_weights=[
            1.0,
            positive_weight,
        ],
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )
    auc = roc_auc_score(
        y_test,
        probabilities,
    )
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    report = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    model.save_model(model_path)
    print(f"Saved model to: {model_path}")

    metrics = {
        "model": "high_ready_delay_classifier",
        "accuracy": accuracy,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_iteration": model.get_best_iteration(),
    }

    importance = pd.DataFrame(
        {
            "model": "high_ready_delay_classifier",
            "feature": X_train.columns,
            "importance": model.get_feature_importance(),
        }
    )

    report_text = (
        "Readiness classification report\n"
        "-------------------------------\n"
        f"{report}\n"
        "Confusion matrix\n"
        "----------------\n"
        f"{matrix}\n"
    )

    return metrics, importance, report_text


# =========================================================
# TRAIN ALL VALID MODELS
# =========================================================

def train_models(
    dataframe: pd.DataFrame,
    train_schedule_model: bool,
    ready_threshold: float | None,
) -> None:
    dataframe = dataframe.reset_index(drop=True)

    features, categorical_features = get_features()
    X = clean_model_inputs(
        dataframe,
        features,
        categorical_features,
    )

    classifier_valid = ready_threshold is not None

    if classifier_valid:
        stratify_values = dataframe["high_ready_delay"]
    else:
        stratify_values = None

    train_indices, test_indices = train_test_split(
        np.arange(len(dataframe)),
        test_size=0.20,
        random_state=42,
        stratify=stratify_values,
    )

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics: list[dict] = []
    all_importance: list[pd.DataFrame] = []
    report_sections: list[str] = []

    # Only train this if the source timestamps genuinely contain queue delay.
    if train_schedule_model:
        schedule_metrics, schedule_importance = (
            train_regressor(
                X_train,
                X_test,
                dataframe[
                    "log_schedule_delay_sec"
                ].iloc[train_indices],
                dataframe[
                    "log_schedule_delay_sec"
                ].iloc[test_indices],
                categorical_features,
                "schedule_delay",
                MODEL_DIR / "schedule_delay_model.cbm",
            )
        )

        all_metrics.append(schedule_metrics)
        all_importance.append(schedule_importance)
    else:
        report_sections.append(
            "Schedule-delay regressor skipped: job.start_time and "
            "earliest task start do not contain enough non-zero variation."
        )

    # The readiness-delay regression target is valid.
    ready_metrics, ready_importance = train_regressor(
        X_train,
        X_test,
        dataframe[
            "log_ready_delay_sec"
        ].iloc[train_indices],
        dataframe[
            "log_ready_delay_sec"
        ].iloc[test_indices],
        categorical_features,
        "ready_delay",
        MODEL_DIR / "ready_delay_model.cbm",
    )

    all_metrics.append(ready_metrics)
    all_importance.append(ready_importance)

    # Train a readiness-risk classifier only when two stable classes exist.
    if classifier_valid:
        (
            class_metrics,
            class_importance,
            class_report,
        ) = train_classifier(
            X_train,
            X_test,
            dataframe[
                "high_ready_delay"
            ].iloc[train_indices],
            dataframe[
                "high_ready_delay"
            ].iloc[test_indices],
            categorical_features,
            MODEL_DIR
            / "high_ready_delay_classifier.cbm",
        )

        all_metrics.append(class_metrics)
        all_importance.append(class_importance)
        report_sections.append(class_report)
    else:
        report_sections.append(
            "Readiness classifier skipped: a stable two-class target "
            "could not be created from ready_delay_sec."
        )

    metrics_df = pd.DataFrame(all_metrics)

    if all_importance:
        importance_df = pd.concat(
            all_importance,
            ignore_index=True,
        )
    else:
        importance_df = pd.DataFrame(
            columns=[
                "model",
                "feature",
                "importance",
            ]
        )

    metrics_df.to_csv(
        METRICS_CSV_PATH,
        index=False,
    )
    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    lines = [
        "Alibaba Readiness and Scheduling Model Results",
        "==============================================",
        f"Rows: {len(dataframe):,}",
        f"Training rows: {len(train_indices):,}",
        f"Testing rows: {len(test_indices):,}",
        f"Schedule-delay model trained: {train_schedule_model}",
        (
            f"High-readiness-delay threshold: "
            f"{ready_threshold:.4f} seconds"
            if ready_threshold is not None
            else "High-readiness-delay classifier: skipped"
        ),
        "",
        metrics_df.to_string(index=False),
        "",
        *report_sections,
    ]

    METRICS_PATH.write_text(
        "\n".join(lines)
    )

    print("\nScheduling/readiness model summary")
    print("--------------------------------")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(
        f"Saved metrics CSV to: "
        f"{METRICS_CSV_PATH}"
    )
    print(
        f"Saved feature importance to: "
        f"{IMPORTANCE_PATH}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    try:
        print("Preparing Alibaba scheduling/readiness data...")

        (
            dataset,
            train_schedule_model,
            ready_threshold,
        ) = load_and_prepare_data()

        print("\nSchedule-delay summary")
        print("--------------------------------")
        print(
            dataset["schedule_delay_sec"].describe()
        )
        print(
            dataset["schedule_delay_sec"].quantile(
                [0, 0.50, 0.90, 0.95, 0.99, 0.999, 1.0]
            )
        )

        print("\nReady-delay summary")
        print("--------------------------------")
        print(
            dataset["ready_delay_sec"].describe()
        )
        print(
            dataset["ready_delay_sec"].quantile(
                [0, 0.50, 0.90, 0.95, 0.99, 0.999, 1.0]
            )
        )

        train_models(
            dataset,
            train_schedule_model,
            ready_threshold,
        )

    except Exception:
        print("\nERROR: training did not complete.")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
