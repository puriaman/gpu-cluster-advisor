import pandas as pd
import numpy as np
from pathlib import Path

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("data/asi_opensource_job_execution_summary/part-000.parquet")
MODEL_PATH = Path("models/runtime_model.cbm")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    cols = [
        "gpu_spec_public",
        "priority_class",
        "job_type_public",
        "model_type_public",
        "is_genai_request",
        "gpu_request",
        "duration_hours",
    ]

    df = df[cols].dropna()
    df = df[df["duration_hours"] > 0]

    # Log-transform runtime because job durations are usually skewed
    df["log_duration_hours"] = np.log1p(df["duration_hours"])

    return df


def train_model(df: pd.DataFrame):
    features = [
        "gpu_spec_public",
        "priority_class",
        "job_type_public",
        "model_type_public",
        "is_genai_request",
        "gpu_request",
    ]

    target = "log_duration_hours"

    X = df[features]
    y = df[target]

    categorical_features = [
        "gpu_spec_public",
        "priority_class",
        "job_type_public",
        "model_type_public",
        "is_genai_request",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        verbose=100,
        random_seed=42,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_test, y_test),
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(MODEL_PATH)

    preds_log = model.predict(X_test)

    preds = np.expm1(preds_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2 = r2_score(actual, preds)

    print("Runtime Model Results")
    print("---------------------")
    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R²:   {r2:.3f}")


    print(f"\nSaved model to {MODEL_PATH}")


def main():
    df = load_data(DATA_PATH)
    train_model(df)


if __name__ == "__main__":
    main()