import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


MODEL_PATH = "models/runtime_model.cbm"


def predict_runtime(
    gpu_spec_public: str,
    priority_class: str,
    job_type_public: str,
    model_type_public: str,
    is_genai_request: bool,
    gpu_request: float,
):
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    row = pd.DataFrame(
        [
            {
                "gpu_spec_public": gpu_spec_public,
                "priority_class": priority_class,
                "job_type_public": job_type_public,
                "model_type_public": model_type_public,
                "is_genai_request": is_genai_request,
                "gpu_request": gpu_request,
            }
        ]
    )

    prediction_log = model.predict(row)[0]
    prediction_hours = np.expm1(prediction_log)

    return prediction_hours


if __name__ == "__main__":
    hours = predict_runtime(
        gpu_spec_public="H100",
        priority_class="HP",
        job_type_public="training",
        model_type_public="genai",
        is_genai_request=True,
        gpu_request=8,
    )

    print(f"Predicted runtime: {hours:.2f} hours")