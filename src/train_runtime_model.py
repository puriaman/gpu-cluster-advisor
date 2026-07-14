import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path("data")
MODEL_PATH = Path("models/alibaba_runtime_model.cbm")
METRICS_PATH = Path("models/alibaba_runtime_metrics.txt")
IMPORTANCE_PATH = Path("models/alibaba_feature_importance.csv")


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
# FILE SEARCH
# =========================================================

def find_file(filename: str) -> Path:
    """
    Search recursively inside data/ for the requested filename.
    """

    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} anywhere inside "
            f"{DATA_DIR.resolve()}"
        )

    print(f"Found {filename}: {matches[0]}")
    return matches[0]


# =========================================================
# CSV LOADING
# =========================================================

def read_alibaba_csv(
    path: Path,
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    Read an Alibaba CSV whether it contains headers or is headerless.
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
        df = pd.read_csv(
            path,
            low_memory=False,
        )
    else:
        df = pd.read_csv(
            path,
            names=expected_columns,
            header=None,
            low_memory=False,
        )

    missing = set(expected_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    return df[expected_columns]


# =========================================================
# SAFE DIVISION
# =========================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    minimum_denominator: float = 1e-6,
) -> pd.Series:
    """
    Divide two pandas Series while preventing division by zero.
    """

    safe_denominator = denominator.clip(lower=minimum_denominator)
    return numerator / safe_denominator


# =========================================================
# DATA PREPARATION
# =========================================================

def load_and_prepare_data() -> pd.DataFrame:
    job_path = find_file("pai_job_table.csv")
    task_path = find_file("pai_task_table.csv")

    jobs = read_alibaba_csv(job_path, JOB_COLUMNS)
    tasks = read_alibaba_csv(task_path, TASK_COLUMNS)

    print(f"Raw job rows:  {len(jobs):,}")
    print(f"Raw task rows: {len(tasks):,}")

    # -----------------------------------------------------
    # CLEAN JOB TABLE
    # -----------------------------------------------------

    jobs["start_time"] = pd.to_numeric(
        jobs["start_time"],
        errors="coerce",
    )

    jobs["end_time"] = pd.to_numeric(
        jobs["end_time"],
        errors="coerce",
    )

    jobs = jobs[
        jobs["status"].astype(str).str.lower() == "terminated"
    ].copy()

    jobs["duration_hours"] = (
        jobs["end_time"] - jobs["start_time"]
    ) / 3600.0

    jobs = jobs[
        jobs["duration_hours"].notna()
        & (jobs["duration_hours"] > 0)
    ].copy()

    # Actual execution-start time features
    jobs["start_hour"] = (
        (jobs["start_time"] % 86400) // 3600
    ).astype("int16")

    jobs["start_day"] = (
        (jobs["start_time"] // 86400) % 7
    ).astype("int16")

    # Cyclic encoding:
    # hour 23 is close to hour 0;
    # day 6 is close to day 0.
    jobs["start_hour_sin"] = np.sin(
        2 * np.pi * jobs["start_hour"] / 24
    )

    jobs["start_hour_cos"] = np.cos(
        2 * np.pi * jobs["start_hour"] / 24
    )

    jobs["start_day_sin"] = np.sin(
        2 * np.pi * jobs["start_day"] / 7
    )

    jobs["start_day_cos"] = np.cos(
        2 * np.pi * jobs["start_day"] / 7
    )

    # -----------------------------------------------------
    # CLEAN TASK TABLE
    # -----------------------------------------------------

    numeric_task_columns = [
        "inst_num",
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
        .astype(str)
    )

    tasks["gpu_type"] = (
        tasks["gpu_type"]
        .fillna("NONE")
        .astype(str)
    )

    # -----------------------------------------------------
    # TASK-ROLE FEATURES
    # -----------------------------------------------------

    task_name_lower = tasks["task_name"].str.lower()

    tasks["worker_instances"] = np.where(
        task_name_lower.str.contains("worker", na=False),
        tasks["inst_num"],
        0,
    )

    tasks["ps_instances"] = np.where(
        task_name_lower.str.contains(
            r"(^|[-_])ps($|[-_0-9])|parameter",
            regex=True,
            na=False,
        ),
        tasks["inst_num"],
        0,
    )

    tasks["chief_instances"] = np.where(
        task_name_lower.str.contains(
            "chief|master",
            regex=True,
            na=False,
        ),
        tasks["inst_num"],
        0,
    )

    tasks["evaluator_instances"] = np.where(
        task_name_lower.str.contains(
            "eval|evaluator",
            regex=True,
            na=False,
        ),
        tasks["inst_num"],
        0,
    )

    # -----------------------------------------------------
    # TOTAL REQUESTED RESOURCES PER TASK
    # -----------------------------------------------------

    tasks["requested_cpu_total"] = (
        tasks["plan_cpu"] * tasks["inst_num"]
    )

    tasks["requested_mem_total"] = (
        tasks["plan_mem"] * tasks["inst_num"]
    )

    tasks["requested_gpu_total"] = (
        tasks["plan_gpu"] * tasks["inst_num"]
    )

    # -----------------------------------------------------
    # AGGREGATE TASKS TO ONE ROW PER JOB
    # -----------------------------------------------------

    numerical_task_features = (
        tasks.groupby("job_name", as_index=False)
        .agg(
            task_count=("task_name", "count"),
            total_instances=("inst_num", "sum"),

            total_cpu_request=(
                "requested_cpu_total",
                "sum",
            ),
            total_memory_request=(
                "requested_mem_total",
                "sum",
            ),
            total_gpu_request=(
                "requested_gpu_total",
                "sum",
            ),

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

    categorical_task_features = (
        tasks.groupby("job_name", as_index=False)
        .agg(
            task_types=(
                "task_name",
                lambda values: "|".join(
                    sorted(set(values.astype(str)))
                ),
            ),
            gpu_types=(
                "gpu_type",
                lambda values: "|".join(
                    sorted(set(values.astype(str)))
                ),
            ),
        )
    )

    task_summary = numerical_task_features.merge(
        categorical_task_features,
        on="job_name",
        how="left",
    )

    # -----------------------------------------------------
    # JOIN JOBS AND TASK SUMMARY
    # -----------------------------------------------------

    dataset = jobs.merge(
        task_summary,
        on="job_name",
        how="inner",
    )

    # Trace units:
    # 100 CPU units = 1 CPU core
    # 100 GPU units = 1 full GPU
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

    # -----------------------------------------------------
    # ENGINEERED RESOURCE FEATURES
    # -----------------------------------------------------

    dataset["cpu_per_instance"] = safe_divide(
        dataset["requested_cpu_cores"],
        dataset["total_instances"],
        minimum_denominator=1,
    )

    dataset["memory_per_instance"] = safe_divide(
        dataset["total_memory_request"],
        dataset["total_instances"],
        minimum_denominator=1,
    )

    dataset["gpu_per_instance"] = safe_divide(
        dataset["requested_gpu_count"],
        dataset["total_instances"],
        minimum_denominator=1,
    )

    dataset["cpu_per_gpu"] = safe_divide(
        dataset["requested_cpu_cores"],
        dataset["requested_gpu_count"],
        minimum_denominator=0.01,
    )

    dataset["memory_per_gpu"] = safe_divide(
        dataset["total_memory_request"],
        dataset["requested_gpu_count"],
        minimum_denominator=0.01,
    )

    dataset["instances_per_gpu"] = safe_divide(
        dataset["total_instances"],
        dataset["requested_gpu_count"],
        minimum_denominator=0.01,
    )

    dataset["tasks_per_instance"] = safe_divide(
        dataset["task_count"],
        dataset["total_instances"],
        minimum_denominator=1,
    )

    dataset["worker_fraction"] = safe_divide(
        dataset["worker_count"],
        dataset["total_instances"],
        minimum_denominator=1,
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

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    dataset["log_duration_hours"] = np.log1p(
        dataset["duration_hours"]
    )

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    required_columns = [
        "task_count",
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
        "cpu_per_instance",
        "memory_per_instance",
        "gpu_per_instance",
        "cpu_per_gpu",
        "memory_per_gpu",
        "instances_per_gpu",
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
        "start_hour",
        "start_day",
        "start_hour_sin",
        "start_hour_cos",
        "start_day_sin",
        "start_day_cos",
        "task_types",
        "gpu_types",
        "duration_hours",
        "log_duration_hours",
    ]

    dataset = dataset.dropna(
        subset=required_columns
    ).copy()

    print(f"Usable joined jobs: {len(dataset):,}")

    if len(dataset) < 100:
        raise ValueError(
            "Too few usable joined rows were found. "
            "Check that the job and task files belong "
            "to the same Alibaba trace."
        )

    return dataset


# =========================================================
# MODEL TRAINING
# =========================================================

def train_model(df: pd.DataFrame) -> None:
    features = [
        # Original aggregate features
        "task_count",
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

        # Resource-ratio features
        "cpu_per_instance",
        "memory_per_instance",
        "gpu_per_instance",
        "cpu_per_gpu",
        "memory_per_gpu",
        "instances_per_gpu",
        "tasks_per_instance",

        # Distributed-training structure
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

        # Time features
        "start_hour",
        "start_day",
        "start_hour_sin",
        "start_hour_cos",
        "start_day_sin",
        "start_day_cos",

        # Categorical features
        "task_types",
        "gpu_types",
    ]

    categorical_features = [
        "task_types",
        "gpu_types",
    ]

    target = "log_duration_hours"

    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=150,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(MODEL_PATH)

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    predictions_log = model.predict(X_test)

    predictions_hours = np.maximum(
        0,
        np.expm1(predictions_log),
    )

    actual_hours = np.expm1(
        y_test.to_numpy()
    )

    # -----------------------------------------------------
    # ORIGINAL-HOUR METRICS
    # -----------------------------------------------------

    mae = mean_absolute_error(
        actual_hours,
        predictions_hours,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_hours,
            predictions_hours,
        )
    )

    r2 = r2_score(
        actual_hours,
        predictions_hours,
    )

    # -----------------------------------------------------
    # LOG-SPACE METRICS
    # -----------------------------------------------------

    log_mae = mean_absolute_error(
        y_test,
        predictions_log,
    )

    log_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions_log,
        )
    )

    log_r2 = r2_score(
        y_test,
        predictions_log,
    )

    # -----------------------------------------------------
    # MEDIAN BASELINE
    # -----------------------------------------------------

    median_prediction = np.median(
        np.expm1(y_train.to_numpy())
    )

    baseline_predictions = np.full(
        len(actual_hours),
        median_prediction,
    )

    baseline_mae = mean_absolute_error(
        actual_hours,
        baseline_predictions,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            actual_hours,
            baseline_predictions,
        )
    )

    baseline_r2 = r2_score(
        actual_hours,
        baseline_predictions,
    )

    # -----------------------------------------------------
    # TYPICAL-JOB METRICS BELOW 99TH PERCENTILE
    # -----------------------------------------------------

    p99 = np.quantile(actual_hours, 0.99)
    typical_mask = actual_hours <= p99

    typical_mae = mean_absolute_error(
        actual_hours[typical_mask],
        predictions_hours[typical_mask],
    )

    typical_rmse = np.sqrt(
        mean_squared_error(
            actual_hours[typical_mask],
            predictions_hours[typical_mask],
        )
    )

    typical_r2 = r2_score(
        actual_hours[typical_mask],
        predictions_hours[typical_mask],
    )

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\nAlibaba Runtime Model Results")
    print("--------------------------------")
    print(f"MAE:             {mae:.2f} hours")
    print(f"RMSE:            {rmse:.2f} hours")
    print(f"R²:              {r2:.3f}")

    print("\nLog-space results")
    print("--------------------------------")
    print(f"Log MAE:         {log_mae:.4f}")
    print(f"Log RMSE:        {log_rmse:.4f}")
    print(f"Log R²:          {log_r2:.3f}")

    print("\nMedian baseline")
    print("--------------------------------")
    print(f"Baseline MAE:    {baseline_mae:.2f} hours")
    print(f"Baseline RMSE:   {baseline_rmse:.2f} hours")
    print(f"Baseline R²:     {baseline_r2:.3f}")

    print("\nJobs below test-set 99th percentile")
    print("--------------------------------")
    print(f"99th percentile: {p99:.2f} hours")
    print(f"Typical MAE:     {typical_mae:.2f} hours")
    print(f"Typical RMSE:    {typical_rmse:.2f} hours")
    print(f"Typical R²:      {typical_r2:.3f}")

    print(f"\nBest iteration:  {model.get_best_iteration()}")
    print(f"Saved model to:  {MODEL_PATH}")

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics_text = (
        "Alibaba Runtime Model Results\n"
        "================================\n"
        f"Rows: {len(df):,}\n"
        f"Training rows: {len(X_train):,}\n"
        f"Testing rows: {len(X_test):,}\n"
        "\n"
        f"MAE: {mae:.4f} hours\n"
        f"RMSE: {rmse:.4f} hours\n"
        f"R2: {r2:.6f}\n"
        "\n"
        f"Log MAE: {log_mae:.6f}\n"
        f"Log RMSE: {log_rmse:.6f}\n"
        f"Log R2: {log_r2:.6f}\n"
        "\n"
        f"Baseline MAE: {baseline_mae:.4f} hours\n"
        f"Baseline RMSE: {baseline_rmse:.4f} hours\n"
        f"Baseline R2: {baseline_r2:.6f}\n"
        "\n"
        f"Test 99th percentile: {p99:.4f} hours\n"
        f"Typical-job MAE: {typical_mae:.4f} hours\n"
        f"Typical-job RMSE: {typical_rmse:.4f} hours\n"
        f"Typical-job R2: {typical_r2:.6f}\n"
        "\n"
        f"Best iteration: {model.get_best_iteration()}\n"
    )

    METRICS_PATH.write_text(metrics_text)

    print(f"Saved metrics to: {METRICS_PATH}")

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.get_feature_importance(),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    print("\nFeature importance")
    print("--------------------------------")
    print(importance.to_string(index=False))

    print(
        f"\nSaved feature importance to: "
        f"{IMPORTANCE_PATH}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    try:
        print("Preparing Alibaba GPU trace data...")

        dataset = load_and_prepare_data()

        print("\nRuntime summary")
        print("--------------------------------")
        print(dataset["duration_hours"].describe())

        print("\nRuntime percentiles")
        print("--------------------------------")
        print(
            dataset["duration_hours"].quantile(
                [
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                    0.999,
                ]
            )
        )

        print("\nTraining improved CatBoost model...")
        train_model(dataset)

    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()