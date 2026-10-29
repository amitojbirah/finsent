import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from finsent.config import MLFLOW_TRACKING_URI
from finsent.data.split import RANDOM_STATE, VAL_SIZE, load_splits
from finsent.models.baseline import build_dummy, build_tfidf_lr
from finsent.models.evaluate import evaluate, plot_confusion_matrix

EXPERIMENT_NAME = "finsent-baselines"
SIMPLE_TYPES = (str, int, float, bool, tuple, type(None))


def simple_params(model):
    """Keep only simple, loggable values from model.get_params()."""
    return {k: v for k, v in model.get_params().items() if isinstance(v, SIMPLE_TYPES)}


def train_and_log(run_name, model, train, val):
    """Fit one model, log everything to one MLflow run, return (val metrics, run id)."""
    with mlflow.start_run(run_name=run_name) as run:
        # 1. Params
        mlflow.log_params(simple_params(model))
        mlflow.log_params(
            {
                "split_val_size": VAL_SIZE,
                "split_random_state": RANDOM_STATE,
                "n_train": len(train),
                "n_val": len(val),
            }
        )

        # 2. Train and predict
        model.fit(train["text"], train["label"])
        val_preds = model.predict(val["text"])

        # 3. Metrics
        metrics = evaluate(val["label"], val_preds)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

        # 4. Confusion matrix
        fig = plot_confusion_matrix(val["label"], val_preds, f"{run_name} (val)")
        mlflow.log_figure(fig, "confusion_matrix_val.png")
        plt.close(fig)

        # 5. Model
        mlflow.sklearn.log_model(model, name="model")

        return metrics, run.info.run_id


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train, val, test = load_splits()

    candidates = {
        "dummy": build_dummy(),
        "tfidf_lr": build_tfidf_lr(),
        "tfidf_lr_balanced": build_tfidf_lr("balanced"),
    }

    # 1. Train every candidate and score it on val
    results = {}
    for run_name, model in candidates.items():
        metrics, run_id = train_and_log(run_name, model, train, val)
        results[run_name] = {"metrics": metrics, "run_id": run_id, "model": model}
        print(f"{run_name:<20} val_macro_f1 = {metrics['macro_f1']:.4f}")

    # 2. Pick the winner using val only
    best_name = max(results, key=lambda name: results[name]["metrics"]["macro_f1"])
    best = results[best_name]
    print(f"\nSelected: {best_name}")

    # 3. Score the winner on test, once, inside its existing run
    with mlflow.start_run(run_id=best["run_id"]):
        test_preds = best["model"].predict(test["text"])
        test_metrics = evaluate(test["label"], test_preds)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        fig = plot_confusion_matrix(test["label"], test_preds, f"{best_name} (test)")
        mlflow.log_figure(fig, "confusion_matrix_test.png")
        plt.close(fig)

        mlflow.set_tag("selected", "true")

    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.4f}")


if __name__ == "__main__":
    main()
