"""Register the selected (champion) run's model in the MLflow Model Registry."""

import mlflow
import mlflow.transformers
import torch
import transformers
from mlflow import MlflowClient
from transformers import pipeline

from finsent.config import MLFLOW_TRACKING_URI, PROJECT_ROOT
from finsent.models.train_baseline import EXPERIMENT_NAME

REGISTERED_NAME = "finsent-sentiment"
ALIAS = "champion"

# Where each candidate's saved model files live locally
MODEL_DIRS = {
    "finbert_ft": PROJECT_ROOT / "artifacts" / "finbert_ft",
    "finbert_full_local": PROJECT_ROOT / "artifacts" / "runs" / "full" / "final",
}

EXAMPLE_TWEETS = [
    "$AAPL beats estimates, raises guidance",
    "$NFLX cut to underweight at Morgan Stanley",
]


def find_champion():
    """The run tagged selected=true by select_model.py."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.selected = 'true'",
    )
    if len(runs) != 1:
        raise RuntimeError(f"Expected exactly 1 selected run, found {len(runs)}")
    row = runs.iloc[0]
    return row["tags.mlflow.runName"], row["run_id"]


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    name, run_id = find_champion()
    print(f"Champion: {name} ({run_id})")
    if name not in MODEL_DIRS:
        raise RuntimeError(f"No local model files listed for {name}")

    # Model + tokenizer + softmax + label names, bundled
    clf = pipeline("text-classification", model=str(MODEL_DIRS[name]))

    # Log in MLflow format inside the champion's own run, and register it
    with mlflow.start_run(run_id=run_id):
        info = mlflow.transformers.log_model(
            transformers_model=clf,
            name="registered_model",
            task="text-classification",
            input_example=EXAMPLE_TWEETS,
            registered_model_name=REGISTERED_NAME,
            pip_requirements=[
                f"torch=={torch.__version__.split('+')[0]}",
                f"transformers=={transformers.__version__}",
                f"mlflow=={mlflow.__version__}",
            ],
        )

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(REGISTERED_NAME, ALIAS, version)
    print(f"Registered {REGISTERED_NAME} v{version} as @{ALIAS}")

    # Verify exactly the way the API will load it
    loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@{ALIAS}")
    print(loaded.predict(EXAMPLE_TWEETS + ["Fed meeting scheduled for Wednesday"]))


if __name__ == "__main__":
    main()
