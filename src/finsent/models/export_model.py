"""Export the @champion model from the MLflow Registry into a plain folder for serving."""

import json

import mlflow
import mlflow.transformers
from mlflow import MlflowClient

from finsent.config import MLFLOW_TRACKING_URI, MODEL_DIR
from finsent.models.register_model import ALIAS, REGISTERED_NAME


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 1. Which version does @champion point to?
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_NAME, ALIAS)
    print(f"{REGISTERED_NAME}@{ALIAS} -> v{mv.version} (run {mv.run_id})")

    # 2. Load it from the Registry as a Hugging Face pipeline
    pipe = mlflow.transformers.load_model(f"models:/{REGISTERED_NAME}@{ALIAS}")

    # 3. Save as a plain Hugging Face folder (no MLflow needed to load it)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(MODEL_DIR))

    # 4. Record which model this is, for the predictions table
    info = {
        "name": REGISTERED_NAME,
        "version": mv.version,
        "alias": ALIAS,
        "run_id": mv.run_id,
    }
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Exported {REGISTERED_NAME} v{mv.version} to {MODEL_DIR}")


if __name__ == "__main__":
    main()
