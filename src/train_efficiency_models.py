import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

METRICS_PATH = MODEL_DIR / "efficiency_metrics_improved.txt"
METRICS_CSV_PATH = MODEL_DIR / "efficiency_metrics_improved.csv"
IMPORTANCE_PATH = MODEL_DIR / "efficiency_feature_importance_improved.csv"
PREDICTIONS_PATH = MODEL_DIR / "efficiency_test_predictions.csv"

RUNTIME_MODEL_PATH = MODEL_DIR / "alibaba_runtime_model.cbm"


# =========================================================
# OFFICIAL ALIBABA GPU 2020 SCHEMAS
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

INSTANCE_COLUMNS = [
    "job_name",
    "task_name",
    "inst_name",
    "worker_name",
    "inst_id",
    "status",
    "start_time",
    "end_time",
    "machine",
]

SENSOR_COLUMNS = [
    "job_name",
    "task_name",
    "worker_name",
    "inst_id",
    "machine",
    "gpu_name",
    "cpu_usage",
    "gpu_wrk_util",
    "avg_mem",
    "max_mem",
    "avg_gpu_wrk_mem",
    "max_gpu_wrk_mem",
    "read",
    "write",
    "read_count",
    "write_count",
]

GROUP_TAG_COLUMNS = [
    "inst_id",
    "user",
    "gpu_type_spec",
    "group",
    "workload",
]

MACHINE_SPEC_COLUMNS = [
    "machine",
    "gpu_type",
    "cap_cpu",
    "cap_mem",
    "cap_gpu",
]


# =========================================================
# FILE AND CSV HELPERS
# =========================================================

def find_file(filename: str, required: bool = True) -> Path | None:
    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        if required:
            raise FileNotFoundError(
                f"Could not find {filename} anywhere inside {DATA_DIR.resolve()}"
            )
        print(f"Optional file not found: {filename}")
        return None

    print(f"Found {filename}: {matches[0]}")
    return matches[0]


def read_alibaba_csv(
    path: Path,
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    Read an Alibaba CSV whether it contains a header row or is headerless.
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
        df = pd.read_csv(path, low_memory=False)
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


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    minimum_denominator: float = 1e-6,
) -> pd.Series:
    return numerator / denominator.clip(lower=minimum_denominator)


def clean_category(series: pd.Series, missing_value: str = "UNKNOWN") -> pd.Series:
    return (
        series.fillna(missing_value)
        .astype(str)
        .replace({"": missing_value, "nan": missing_value, "None": missing_value})
    )


def combine_unique_strings(
    values: pd.Series,
    missing_value: str = "UNKNOWN",
) -> str:
    """
    Safely combine unique values into a pipe-separated categorical value.
    This avoids mixed float/string comparison errors during sorting.
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
# REQUEST FEATURES
# =========================================================

def prepare_job_features(jobs: pd.DataFrame) -> pd.DataFrame:
    for column in ["start_time", "end_time"]:
        jobs[column] = pd.to_numeric(jobs[column], errors="coerce")

    jobs = jobs[jobs["start_time"].notna()].copy()

    jobs["start_hour"] = (
        (jobs["start_time"] % 86400) // 3600
    ).astype("int16")

    jobs["start_day"] = (
        (jobs["start_time"] // 86400) % 7
    ).astype("int16")

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

    jobs["job_status"] = clean_category(jobs["status"])

    # Actual runtime is retained only for diagnostics.
    # It is NOT used as an efficiency-model input.
    jobs["duration_hours"] = (
        jobs["end_time"] - jobs["start_time"]
    ) / 3600.0

    return jobs[
        [
            "job_name",
            "inst_id",
            "user",
            "job_status",
            "start_time",
            "duration_hours",
            "start_hour",
            "start_day",
            "start_hour_sin",
            "start_hour_cos",
            "start_day_sin",
            "start_day_cos",
        ]
    ]


def prepare_task_features(tasks: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "inst_num",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
        "plan_gpu",
    ]

    for column in numeric_columns:
        tasks[column] = pd.to_numeric(tasks[column], errors="coerce")

    tasks["inst_num"] = tasks["inst_num"].fillna(1).clip(lower=1)
    tasks["plan_cpu"] = tasks["plan_cpu"].fillna(0).clip(lower=0)
    tasks["plan_mem"] = tasks["plan_mem"].fillna(0).clip(lower=0)
    tasks["plan_gpu"] = tasks["plan_gpu"].fillna(0).clip(lower=0)

    tasks["task_name"] = clean_category(tasks["task_name"])
    tasks["gpu_type"] = clean_category(tasks["gpu_type"], "NONE")

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

    tasks["requested_cpu_total"] = tasks["plan_cpu"] * tasks["inst_num"]
    tasks["requested_mem_total"] = tasks["plan_mem"] * tasks["inst_num"]
    tasks["requested_gpu_total"] = tasks["plan_gpu"] * tasks["inst_num"]

    numeric_summary = (
        tasks.groupby("job_name", as_index=False)
        .agg(
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
                lambda values: combine_unique_strings(values),
            ),
            gpu_types=(
                "gpu_type",
                lambda values: combine_unique_strings(values),
            ),
        )
    )

    summary = numeric_summary.merge(
        category_summary,
        on="job_name",
        how="left",
    )

    # Official units:
    # plan_cpu 100 = 1 CPU core
    # plan_gpu 100 = 1 GPU
    # plan_mem is already GB
    summary["requested_cpu_cores"] = summary["total_cpu_request"] / 100.0
    summary["requested_gpu_count"] = summary["total_gpu_request"] / 100.0

    summary["max_cpu_cores_per_instance"] = (
        summary["max_cpu_per_instance"] / 100.0
    )
    summary["max_gpu_count_per_instance"] = (
        summary["max_gpu_per_instance"] / 100.0
    )
    summary["mean_cpu_cores_per_instance"] = (
        summary["mean_cpu_per_instance"] / 100.0
    )
    summary["mean_gpu_count_per_instance"] = (
        summary["mean_gpu_per_instance"] / 100.0
    )

    summary["cpu_per_gpu"] = safe_divide(
        summary["requested_cpu_cores"],
        summary["requested_gpu_count"],
        0.01,
    )
    summary["memory_per_gpu"] = safe_divide(
        summary["total_memory_request"],
        summary["requested_gpu_count"],
        0.01,
    )
    summary["cpu_per_instance"] = safe_divide(
        summary["requested_cpu_cores"],
        summary["total_instances"],
        1,
    )
    summary["memory_per_instance"] = safe_divide(
        summary["total_memory_request"],
        summary["total_instances"],
        1,
    )
    summary["gpu_per_instance"] = safe_divide(
        summary["requested_gpu_count"],
        summary["total_instances"],
        1,
    )
    summary["memory_request_per_task"] = safe_divide(
        summary["total_memory_request"],
        summary["task_count"],
        1,
    )
    summary["memory_request_per_worker"] = safe_divide(
        summary["total_memory_request"],
        summary["worker_count"],
        1,
    )
    summary["cpu_request_per_worker"] = safe_divide(
        summary["requested_cpu_cores"],
        summary["worker_count"],
        1,
    )
    summary["tasks_per_instance"] = safe_divide(
        summary["task_count"],
        summary["total_instances"],
        1,
    )
    summary["worker_fraction"] = safe_divide(
        summary["worker_count"],
        summary["total_instances"],
        1,
    )

    summary["is_multi_gpu"] = (
        summary["requested_gpu_count"] > 1
    ).astype("int8")
    summary["is_distributed"] = (
        summary["total_instances"] > 1
    ).astype("int8")
    summary["has_parameter_server"] = (
        summary["ps_count"] > 0
    ).astype("int8")
    summary["has_chief"] = (
        summary["chief_count"] > 0
    ).astype("int8")
    summary["has_evaluator"] = (
        summary["evaluator_count"] > 0
    ).astype("int8")

    summary["job_resource_class"] = np.select(
        [
            summary["requested_gpu_count"] <= 0,
            summary["requested_gpu_count"] <= 1,
        ],
        [
            "cpu_only",
            "single_gpu",
        ],
        default="multi_gpu",
    )

    return summary


# =========================================================
# OPTIONAL SEMANTIC, INSTANCE, AND MACHINE FEATURES
# =========================================================

def prepare_group_features(group_tags: pd.DataFrame) -> pd.DataFrame:
    group_tags["gpu_type_spec"] = clean_category(
        group_tags["gpu_type_spec"],
        "UNSPECIFIED",
    )
    group_tags["group"] = clean_category(
        group_tags["group"],
        "NO_GROUP",
    )
    group_tags["workload"] = clean_category(
        group_tags["workload"],
        "UNKNOWN_WORKLOAD",
    )

    # One row per job_id/inst_id in the official table.
    return (
        group_tags.groupby("inst_id", as_index=False)
        .agg(
            gpu_type_spec=("gpu_type_spec", "first"),
            semantic_group=("group", "first"),
            workload=("workload", "first"),
        )
    )


def prepare_instance_features(instances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for column in ["start_time", "end_time"]:
        instances[column] = pd.to_numeric(
            instances[column],
            errors="coerce",
        )

    instances["machine"] = clean_category(instances["machine"], "UNKNOWN_MACHINE")
    instances["worker_name"] = clean_category(
        instances["worker_name"],
        "UNKNOWN_WORKER",
    )

    instances["instance_duration_hours"] = (
        instances["end_time"] - instances["start_time"]
    ) / 3600.0

    summary = (
        instances.groupby("job_name", as_index=False)
        .agg(
            actual_instance_count=("worker_name", "nunique"),
            machine_count=("machine", "nunique"),
            earliest_instance_start=("start_time", "min"),
            latest_instance_start=("start_time", "max"),
            mean_instance_duration_hours=("instance_duration_hours", "mean"),
            max_instance_duration_hours=("instance_duration_hours", "max"),
        )
    )

    summary["instance_launch_spread_sec"] = (
        summary["latest_instance_start"]
        - summary["earliest_instance_start"]
    )
    summary["instances_per_machine"] = safe_divide(
        summary["actual_instance_count"],
        summary["machine_count"],
        1,
    )

    # Keep unique job-machine placements for machine-spec aggregation.
    placements = instances[["job_name", "machine"]].drop_duplicates()

    return summary, placements


def prepare_machine_features(
    machine_specs: pd.DataFrame,
    placements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate machine-capacity information to one row per job.

    This implementation explicitly converts all machine and GPU-type values
    to strings before grouping, preventing mixed float/string sort errors.
    """

    machine_specs = machine_specs.copy()
    placements = placements.copy()

    # Normalize join keys so pandas does not compare mixed types.
    machine_specs["machine"] = (
        machine_specs["machine"]
        .fillna("UNKNOWN_MACHINE")
        .map(str)
    )

    placements["machine"] = (
        placements["machine"]
        .fillna("UNKNOWN_MACHINE")
        .map(str)
    )

    # Convert machine capacities to numeric values.
    for column in ["cap_cpu", "cap_mem", "cap_gpu"]:
        machine_specs[column] = pd.to_numeric(
            machine_specs[column],
            errors="coerce",
        )

    # Convert every GPU-type value to a normal Python string.
    machine_specs["machine_gpu_type"] = (
        machine_specs["gpu_type"]
        .fillna("UNKNOWN_GPU")
        .map(str)
        .replace(
            {
                "": "UNKNOWN_GPU",
                "nan": "UNKNOWN_GPU",
                "None": "UNKNOWN_GPU",
            }
        )
    )

    placement_specs = placements.merge(
        machine_specs[
            [
                "machine",
                "machine_gpu_type",
                "cap_cpu",
                "cap_mem",
                "cap_gpu",
            ]
        ],
        on="machine",
        how="left",
    )

    placement_specs["machine_gpu_type"] = (
        placement_specs["machine_gpu_type"]
        .fillna("UNKNOWN_GPU")
        .map(str)
        .replace(
            {
                "": "UNKNOWN_GPU",
                "nan": "UNKNOWN_GPU",
                "None": "UNKNOWN_GPU",
            }
        )
    )

    # Numeric aggregation.
    numeric_summary = (
        placement_specs.groupby(
            "job_name",
            as_index=False,
            dropna=False,
        )
        .agg(
            assigned_machine_count=("machine", "nunique"),
            total_machine_cpu_capacity=("cap_cpu", "sum"),
            total_machine_memory_capacity=("cap_mem", "sum"),
            total_machine_gpu_capacity=("cap_gpu", "sum"),
            mean_machine_cpu_capacity=("cap_cpu", "mean"),
            mean_machine_memory_capacity=("cap_mem", "mean"),
            mean_machine_gpu_capacity=("cap_gpu", "mean"),
            max_machine_cpu_capacity=("cap_cpu", "max"),
            max_machine_memory_capacity=("cap_mem", "max"),
            max_machine_gpu_capacity=("cap_gpu", "max"),
        )
    )

    def combine_strings(values: pd.Series) -> str:
        """
        Safely combine unique values without comparing floats and strings.
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
            return "UNKNOWN_GPU"

        return "|".join(sorted(cleaned, key=str))

    category_summary = (
        placement_specs.groupby(
            "job_name",
            as_index=False,
            dropna=False,
        )
        .agg(
            assigned_machine_gpu_types=(
                "machine_gpu_type",
                combine_strings,
            )
        )
    )

    result = numeric_summary.merge(
        category_summary,
        on="job_name",
        how="left",
    )

    result["assigned_machine_gpu_types"] = (
        result["assigned_machine_gpu_types"]
        .fillna("UNKNOWN_GPU")
        .map(str)
    )

    return result


# =========================================================
# SENSOR TARGETS
# =========================================================

def prepare_sensor_targets(sensors: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "cpu_usage",
        "gpu_wrk_util",
        "avg_mem",
        "max_mem",
        "avg_gpu_wrk_mem",
        "max_gpu_wrk_mem",
        "read",
        "write",
        "read_count",
        "write_count",
    ]

    for column in numeric_columns:
        sensors[column] = pd.to_numeric(
            sensors[column],
            errors="coerce",
        )

    sensors = sensors.replace([np.inf, -np.inf], np.nan)

    # Official units:
    # cpu_usage 100 = 1 CPU core
    # gpu_wrk_util is GPU utilization percentage-style units
    sensors["cpu_usage_cores"] = sensors["cpu_usage"] / 100.0
    sensors["gpu_util_percent"] = sensors["gpu_wrk_util"]

    return (
        sensors.groupby("job_name", as_index=False)
        .agg(
            avg_gpu_util_percent=("gpu_util_percent", "mean"),
            avg_gpu_memory_gib=("avg_gpu_wrk_mem", "mean"),
            max_gpu_memory_gib=("max_gpu_wrk_mem", "max"),
            avg_cpu_usage_cores=("cpu_usage_cores", "mean"),
            avg_memory_used_gib=("avg_mem", "mean"),
            max_memory_used_gib=("max_mem", "max"),
            avg_network_read_bytes=("read", "mean"),
            avg_network_write_bytes=("write", "mean"),
            avg_network_read_count=("read_count", "mean"),
            avg_network_write_count=("write_count", "mean"),
            sensor_instance_count=("worker_name", "nunique"),
            sensor_machine_count=("machine", "nunique"),
        )
    )


# =========================================================
# OPTIONAL PREDICTED-RUNTIME FEATURE
# =========================================================

def add_predicted_runtime_feature(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """
    Add a prediction from the separately trained runtime model.

    This is safe because it uses the runtime MODEL'S prediction, not actual
    duration_hours. If model loading/prediction fails, the feature is skipped.
    """

    if not RUNTIME_MODEL_PATH.exists():
        print(
            f"Runtime model not found at {RUNTIME_MODEL_PATH}; "
            "continuing without predicted_runtime_hours."
        )
        return dataset, False

    runtime_features = [
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
    ]

    dataset["instances_per_gpu"] = safe_divide(
        dataset["total_instances"],
        dataset["requested_gpu_count"],
        0.01,
    )

    missing = [
        feature
        for feature in runtime_features
        if feature not in dataset.columns
    ]

    if missing:
        print(
            "Skipping predicted-runtime feature because these runtime "
            f"features are missing: {missing}"
        )
        return dataset, False

    try:
        model = CatBoostRegressor()
        model.load_model(RUNTIME_MODEL_PATH)

        predictions_log = model.predict(dataset[runtime_features])
        dataset["predicted_runtime_hours"] = np.maximum(
            0,
            np.expm1(predictions_log),
        )

        print("Added predicted_runtime_hours from the saved runtime model.")
        return dataset, True

    except Exception as error:
        print(
            "Could not add predicted_runtime_hours; continuing without it. "
            f"Reason: {error}"
        )
        return dataset, False


# =========================================================
# COMPLETE DATASET
# =========================================================

def load_and_prepare_data() -> tuple[pd.DataFrame, bool, bool]:
    job_path = find_file("pai_job_table.csv")
    task_path = find_file("pai_task_table.csv")
    sensor_path = find_file("pai_sensor_table.csv")

    instance_path = find_file("pai_instance_table.csv", required=False)
    group_path = find_file("pai_group_tag_table.csv", required=False)
    machine_spec_path = find_file("pai_machine_spec.csv", required=False)

    jobs = read_alibaba_csv(job_path, JOB_COLUMNS)
    tasks = read_alibaba_csv(task_path, TASK_COLUMNS)
    sensors = read_alibaba_csv(sensor_path, SENSOR_COLUMNS)

    print(f"Raw job rows:    {len(jobs):,}")
    print(f"Raw task rows:   {len(tasks):,}")
    print(f"Raw sensor rows: {len(sensors):,}")

    job_features = prepare_job_features(jobs)
    task_features = prepare_task_features(tasks)
    targets = prepare_sensor_targets(sensors)

    dataset = (
        job_features
        .merge(task_features, on="job_name", how="inner")
        .merge(targets, on="job_name", how="inner")
    )

    added_machine_features = False

    if group_path is not None:
        group_tags = read_alibaba_csv(
            group_path,
            GROUP_TAG_COLUMNS,
        )
        group_features = prepare_group_features(group_tags)
        dataset = dataset.merge(
            group_features,
            on="inst_id",
            how="left",
        )
    else:
        dataset["gpu_type_spec"] = "UNSPECIFIED"
        dataset["semantic_group"] = "NO_GROUP"
        dataset["workload"] = "UNKNOWN_WORKLOAD"

    if instance_path is not None:
        instances = read_alibaba_csv(
            instance_path,
            INSTANCE_COLUMNS,
        )
        instance_features, placements = prepare_instance_features(instances)

        dataset = dataset.merge(
            instance_features,
            on="job_name",
            how="left",
        )

        if machine_spec_path is not None:
            machine_specs = read_alibaba_csv(
                machine_spec_path,
                MACHINE_SPEC_COLUMNS,
            )
            machine_features = prepare_machine_features(
                machine_specs,
                placements,
            )
            dataset = dataset.merge(
                machine_features,
                on="job_name",
                how="left",
            )
            added_machine_features = True

    # Efficiency targets.
    dataset["cpu_request_utilization"] = safe_divide(
        dataset["avg_cpu_usage_cores"],
        dataset["requested_cpu_cores"],
        0.01,
    )
    dataset["memory_request_utilization"] = safe_divide(
        dataset["avg_memory_used_gib"],
        dataset["total_memory_request"],
        0.01,
    )

    # Log targets reduce domination by rare extreme jobs.
    dataset["log_avg_cpu_usage_cores"] = np.log1p(
        dataset["avg_cpu_usage_cores"]
    )
    dataset["log_avg_memory_used_gib"] = np.log1p(
        dataset["avg_memory_used_gib"]
    )
    dataset["log_cpu_request_utilization"] = np.log1p(
        dataset["cpu_request_utilization"]
    )
    dataset["log_memory_request_utilization"] = np.log1p(
        dataset["memory_request_utilization"]
    )

    dataset, added_runtime_feature = add_predicted_runtime_feature(dataset)

    categorical_defaults = {
        "task_types": "UNKNOWN",
        "gpu_types": "NONE",
        "job_resource_class": "UNKNOWN",
        "gpu_type_spec": "UNSPECIFIED",
        "semantic_group": "NO_GROUP",
        "workload": "UNKNOWN_WORKLOAD",
        "assigned_machine_gpu_types": "UNKNOWN_GPU",
    }

    for column, default in categorical_defaults.items():
        if column not in dataset.columns:
            dataset[column] = default
        dataset[column] = clean_category(dataset[column], default)

    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    required_targets = [
        "avg_gpu_util_percent",
        "avg_gpu_memory_gib",
        "avg_cpu_usage_cores",
        "avg_memory_used_gib",
        "cpu_request_utilization",
        "memory_request_utilization",
        "log_avg_cpu_usage_cores",
        "log_avg_memory_used_gib",
        "log_cpu_request_utilization",
        "log_memory_request_utilization",
    ]

    dataset = dataset.dropna(subset=required_targets).copy()

    # Remove physically invalid negative values only.
    for target in [
        "avg_gpu_util_percent",
        "avg_gpu_memory_gib",
        "avg_cpu_usage_cores",
        "avg_memory_used_gib",
        "cpu_request_utilization",
        "memory_request_utilization",
    ]:
        dataset = dataset[dataset[target] >= 0].copy()

    print(f"Usable efficiency rows: {len(dataset):,}")

    if len(dataset) < 100:
        raise ValueError("Too few usable rows for efficiency-model training.")

    return dataset, added_runtime_feature, added_machine_features


# =========================================================
# MODEL CONFIGURATION
# =========================================================

def get_feature_lists(
    added_runtime_feature: bool,
    added_machine_features: bool,
) -> tuple[list[str], list[str]]:
    features = [
        # Requested resources and job shape
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

        # Resource ratios
        "cpu_per_gpu",
        "memory_per_gpu",
        "cpu_per_instance",
        "memory_per_instance",
        "gpu_per_instance",
        "memory_request_per_task",
        "memory_request_per_worker",
        "cpu_request_per_worker",
        "tasks_per_instance",

        # Distributed structure
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

        # Time context
        "start_hour",
        "start_day",
        "start_hour_sin",
        "start_hour_cos",
        "start_day_sin",
        "start_day_cos",

        # Semantic/categorical context
        "task_types",
        "gpu_types",
        "job_resource_class",
        "gpu_type_spec",
        "semantic_group",
        "workload",
    ]

    categorical_features = [
        "task_types",
        "gpu_types",
        "job_resource_class",
        "gpu_type_spec",
        "semantic_group",
        "workload",
    ]

    instance_features = [
        "actual_instance_count",
        "machine_count",
        "instance_launch_spread_sec",
        "instances_per_machine",
    ]
    for feature in instance_features:
        if feature in features:
            continue

    # Only add instance fields that were actually joined.
    features.extend(
        [
            feature
            for feature in instance_features
            if feature not in features
        ]
    )

    if added_machine_features:
        machine_features = [
            "assigned_machine_count",
            "total_machine_cpu_capacity",
            "total_machine_memory_capacity",
            "total_machine_gpu_capacity",
            "mean_machine_cpu_capacity",
            "mean_machine_memory_capacity",
            "mean_machine_gpu_capacity",
            "max_machine_cpu_capacity",
            "max_machine_memory_capacity",
            "max_machine_gpu_capacity",
            "assigned_machine_gpu_types",
        ]
        features.extend(machine_features)
        categorical_features.append("assigned_machine_gpu_types")

    if added_runtime_feature:
        features.append("predicted_runtime_hours")

    return features, categorical_features


TARGET_CONFIGS = {
    # Existing GPU models
    "avg_gpu_util_percent": {
        "model_file": "gpu_utilization_model.cbm",
        "training_target": "avg_gpu_util_percent",
        "inverse": "identity",
        "iterations": 2500,
        "depth": 8,
        "learning_rate": 0.025,
        "loss_function": "Huber:delta=5",
        "eval_metric": "MAE",
    },
    "avg_gpu_memory_gib": {
        "model_file": "gpu_memory_model.cbm",
        "training_target": "avg_gpu_memory_gib",
        "inverse": "identity",
        "iterations": 2500,
        "depth": 8,
        "learning_rate": 0.025,
        "loss_function": "Huber:delta=1",
        "eval_metric": "MAE",
    },

    # Improved absolute CPU and memory models train in log space.
    "avg_cpu_usage_cores": {
        "model_file": "cpu_usage_model.cbm",
        "training_target": "log_avg_cpu_usage_cores",
        "inverse": "expm1",
        "iterations": 4000,
        "depth": 9,
        "learning_rate": 0.02,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
    },
    "avg_memory_used_gib": {
        "model_file": "memory_usage_model.cbm",
        "training_target": "log_avg_memory_used_gib",
        "inverse": "expm1",
        "iterations": 4000,
        "depth": 10,
        "learning_rate": 0.02,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
    },

    # New advisor-focused efficiency-ratio models.
    "cpu_request_utilization": {
        "model_file": "cpu_utilization_ratio_model.cbm",
        "training_target": "log_cpu_request_utilization",
        "inverse": "expm1",
        "iterations": 4000,
        "depth": 9,
        "learning_rate": 0.02,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
    },
    "memory_request_utilization": {
        "model_file": "memory_utilization_ratio_model.cbm",
        "training_target": "log_memory_request_utilization",
        "inverse": "expm1",
        "iterations": 4000,
        "depth": 10,
        "learning_rate": 0.02,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
    },
}


# =========================================================
# TRAINING AND EVALUATION
# =========================================================

def inverse_predictions(
    values: np.ndarray,
    inverse: str,
) -> np.ndarray:
    if inverse == "expm1":
        return np.maximum(0, np.expm1(values))
    return np.maximum(0, values)


def train_one_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    categorical_features: list[str],
    output_target: str,
    config: dict,
) -> tuple[CatBoostRegressor, dict, pd.DataFrame, pd.DataFrame]:
    training_target = config["training_target"]

    y_train_model = dataframe[training_target].iloc[train_indices]
    y_test_model = dataframe[training_target].iloc[test_indices]

    y_train_actual = dataframe[output_target].iloc[train_indices].to_numpy()
    y_test_actual = dataframe[output_target].iloc[test_indices].to_numpy()

    print(f"\nTraining model for: {output_target}")
    print(f"Training target: {training_target}")

    model = CatBoostRegressor(
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        loss_function=config["loss_function"],
        eval_metric=config["eval_metric"],
        l2_leaf_reg=8,
        random_seed=42,
        verbose=200,
        early_stopping_rounds=250,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train_model,
        cat_features=categorical_features,
        eval_set=(X_test, y_test_model),
        use_best_model=True,
    )

    raw_predictions = model.predict(X_test)
    predictions = inverse_predictions(
        raw_predictions,
        config["inverse"],
    )

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    r2 = r2_score(y_test_actual, predictions)

    baseline_value = float(np.median(y_train_actual))
    baseline_predictions = np.full(
        len(y_test_actual),
        baseline_value,
    )

    baseline_mae = mean_absolute_error(
        y_test_actual,
        baseline_predictions,
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test_actual,
            baseline_predictions,
        )
    )
    baseline_r2 = r2_score(
        y_test_actual,
        baseline_predictions,
    )

    absolute_errors = np.abs(y_test_actual - predictions)

    metrics = {
        "target": output_target,
        "training_target": training_target,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_absolute_error": float(np.median(absolute_errors)),
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "best_iteration": model.get_best_iteration(),
    }

    importance = pd.DataFrame(
        {
            "target": output_target,
            "feature": X_train.columns,
            "importance": model.get_feature_importance(),
        }
    ).sort_values(
        ["target", "importance"],
        ascending=[True, False],
    )

    predictions_frame = pd.DataFrame(
        {
            "target": output_target,
            "row_index": test_indices,
            "actual": y_test_actual,
            "predicted": predictions,
            "absolute_error": absolute_errors,
        }
    )

    return model, metrics, importance, predictions_frame


def print_segment_metrics(
    dataframe: pd.DataFrame,
    predictions_frame: pd.DataFrame,
) -> None:
    """
    Print CPU/memory performance by CPU-only, single-GPU, and multi-GPU jobs.
    """

    print("\nSegment metrics")
    print("--------------------------------")

    for target in [
        "avg_cpu_usage_cores",
        "avg_memory_used_gib",
        "cpu_request_utilization",
        "memory_request_utilization",
    ]:
        target_predictions = predictions_frame[
            predictions_frame["target"] == target
        ].copy()

        if target_predictions.empty:
            continue

        target_predictions["job_resource_class"] = (
            dataframe.loc[
                target_predictions["row_index"],
                "job_resource_class",
            ].to_numpy()
        )

        print(f"\nTarget: {target}")

        for category, group in target_predictions.groupby(
            "job_resource_class"
        ):
            if len(group) < 2:
                continue

            segment_mae = mean_absolute_error(
                group["actual"],
                group["predicted"],
            )
            segment_rmse = np.sqrt(
                mean_squared_error(
                    group["actual"],
                    group["predicted"],
                )
            )
            segment_r2 = r2_score(
                group["actual"],
                group["predicted"],
            )

            print(
                f"  {category:12s} rows={len(group):6,d} "
                f"MAE={segment_mae:8.3f} "
                f"RMSE={segment_rmse:8.3f} "
                f"R²={segment_r2:7.3f}"
            )


def train_models(
    dataframe: pd.DataFrame,
    added_runtime_feature: bool,
    added_machine_features: bool,
) -> None:
    features, categorical_features = get_feature_lists(
        added_runtime_feature,
        added_machine_features,
    )

    # Remove optional columns that were not available after merging.
    missing_features = [
        feature
        for feature in features
        if feature not in dataframe.columns
    ]
    if missing_features:
        print(
            "Skipping unavailable optional features: "
            f"{missing_features}"
        )
        features = [
            feature
            for feature in features
            if feature in dataframe.columns
        ]
        categorical_features = [
            feature
            for feature in categorical_features
            if feature in features
        ]

    X = dataframe[features].copy()

    # -----------------------------------------------------
    # FINAL TYPE CLEANUP
    # -----------------------------------------------------
    # CatBoost categorical columns must contain strings only.
    for column in categorical_features:
        X[column] = (
            X[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # All remaining input columns must be numeric.
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

    # Fill missing numeric inputs with each column's median.
    for column in numeric_features:
        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        X[column] = X[column].fillna(median_value)

    stratify_values = (
        dataframe["job_resource_class"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    train_indices, test_indices = train_test_split(
        np.arange(len(dataframe)),
        test_size=0.20,
        random_state=42,
        stratify=stratify_values,
    )

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_importance = []
    all_predictions = []

    for output_target, config in TARGET_CONFIGS.items():
        model, metrics, importance, predictions = train_one_model(
            X_train,
            X_test,
            dataframe,
            train_indices,
            test_indices,
            categorical_features,
            output_target,
            config,
        )

        model_path = MODEL_DIR / config["model_file"]
        model.save_model(model_path)
        print(f"Saved model to: {model_path}")

        all_metrics.append(metrics)
        all_importance.append(importance)
        all_predictions.append(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    importance_df = pd.concat(
        all_importance,
        ignore_index=True,
    )
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    lines = [
        "Improved Alibaba GPU Efficiency Model Results",
        "============================================",
        f"Rows: {len(dataframe):,}",
        f"Training rows: {len(train_indices):,}",
        f"Testing rows: {len(test_indices):,}",
        f"Predicted-runtime feature: {added_runtime_feature}",
        f"Machine-capacity features: {added_machine_features}",
        "",
        metrics_df.to_string(index=False),
        "",
        "Important interpretation:",
        "- CPU and memory ratio targets estimate usage/request.",
        "- Machine placement features are known after scheduling, not necessarily",
        "  at original job-submission time.",
        "- Sensor target values are never used as input features.",
    ]

    METRICS_PATH.write_text("\n".join(lines))

    print("\nImproved efficiency model summary")
    print("--------------------------------")
    print(metrics_df.to_string(index=False))

    print_segment_metrics(dataframe, predictions_df)

    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved metrics CSV to: {METRICS_CSV_PATH}")
    print(f"Saved feature importance to: {IMPORTANCE_PATH}")
    print(f"Saved test predictions to: {PREDICTIONS_PATH}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    try:
        print("Preparing improved Alibaba efficiency data...")

        (
            dataset,
            added_runtime_feature,
            added_machine_features,
        ) = load_and_prepare_data()

        print("\nTarget percentiles")
        print("--------------------------------")

        for target in [
            "avg_cpu_usage_cores",
            "avg_memory_used_gib",
            "cpu_request_utilization",
            "memory_request_utilization",
        ]:
            print(f"\n{target}")
            print(
                dataset[target].quantile(
                    [0, 0.50, 0.90, 0.95, 0.99, 0.999, 1.0]
                )
            )

        train_models(
            dataset,
            added_runtime_feature,
            added_machine_features,
        )

    except Exception:
        import traceback

        print("\nERROR: training did not complete.")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()