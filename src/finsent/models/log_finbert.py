"""Log the FinBERT runs (trained on Colab) into the local MLflow server."""

import json

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from finsent.config import MLFLOW_TRACKING_URI, PROJECT_ROOT
from finsent.models.evaluate import evaluate, plot_confusion_matrix
from finsent.models.train_baseline import EXPERIMENT_NAME

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FT_DIR = ARTIFACTS_DIR / "finbert_ft"
ZEROSHOT_CSV = ARTIFACTS_DIR / "finbert_zeroshot_val.csv"

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 64
TRAIN_ARG_KEYS = [
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "warmup_steps",
    "weight_decay",
    "fp16",
    "seed",
]


def log_eval(df, prefix, title):
    """Score a predictions CSV, log prefixed metrics + confusion matrix."""
    metrics = evaluate(df["label"], df["pred"])
    mlflow.log_metrics({f"{prefix}{k}": v for k, v in metrics.items()})
    fig = plot_confusion_matrix(df["label"], df["pred"], title)
    mlflow.log_figure(fig, f"confusion_matrix_{prefix.rstrip('_')}.png")
    plt.close(fig)
    return metrics


def log_zeroshot():
    with mlflow.start_run(run_name="finbert_zeroshot") as run:
        mlflow.log_params(
            {"model_name": MODEL_NAME, "mode": "zero-shot", "max_length": MAX_LENGTH}
        )
        df = pd.read_csv(ZEROSHOT_CSV)
        metrics = log_eval(df, "val_", "finbert_zeroshot (val)")
        mlflow.log_artifact(str(ZEROSHOT_CSV))
        return metrics, run.info.run_id


def log_finetuned():
    with open(FT_DIR / "training_args.json") as f:
        train_args = json.load(f)
    with open(FT_DIR / "run_info.json") as f:
        run_info = json.load(f)
    with open(FT_DIR / "log_history.json") as f:
        log_history = json.load(f)

    with mlflow.start_run(run_name="finbert_ft") as run:
        # Params: only the settings that matter
        params = {k: train_args[k] for k in TRAIN_ARG_KEYS if k in train_args}
        params.update(
            {
                "model_name": run_info["model_name"],
                "max_length": run_info["max_length"],
                "n_train": run_info["n_train"],
                "n_val": run_info["n_val"],
            }
        )
        mlflow.log_params(params)
        mlflow.log_metric("train_runtime_s", run_info["train_runtime_s"])

        # Per-epoch curves (skip the extra eval entry from trainer.evaluate())
        seen_epochs = set()
        for entry in log_history:
            if "eval_macro_f1" in entry:
                epoch = round(entry["epoch"])
                if epoch in seen_epochs:
                    continue
                seen_epochs.add(epoch)
                mlflow.log_metric(
                    "epoch_val_macro_f1", entry["eval_macro_f1"], step=epoch
                )
                mlflow.log_metric("epoch_val_loss", entry["eval_loss"], step=epoch)
            elif "loss" in entry:
                mlflow.log_metric("train_loss", entry["loss"], step=entry["step"])

        # Final val metrics = the kept (best) model
        df = pd.read_csv(FT_DIR / "val_preds.csv")
        metrics = log_eval(df, "val_", "finbert_ft (val)")

        mlflow.log_artifacts(str(FT_DIR), artifact_path="finbert_ft")
        mlflow.set_tag("trained_on", "colab-t4")
        return metrics, run.info.run_id


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for name, log_fn in [
        ("finbert_zeroshot", log_zeroshot),
        ("finbert_ft", log_finetuned),
    ]:
        metrics, run_id = log_fn()
        print(f"{name:<18} val_macro_f1 = {metrics['macro_f1']:.4f}  (run {run_id})")


if __name__ == "__main__":
    main()
