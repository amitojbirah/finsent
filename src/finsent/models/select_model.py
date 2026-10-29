"""Pick the best run on val, score it on test once, and mark it as selected."""

import mlflow
import pandas as pd
from mlflow import MlflowClient

from finsent.config import MLFLOW_TRACKING_URI, PROJECT_ROOT
from finsent.models.log_finbert import log_eval
from finsent.models.train_baseline import EXPERIMENT_NAME

# Runs whose test predictions are saved as a CSV (scored here, not at training time)
TEST_PREDS = {
    "finbert_ft": PROJECT_ROOT / "artifacts" / "finbert_ft" / "test_preds.csv",
    "finbert_full_local": PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "full"
    / "final"
    / "test_preds.csv",
    "finbert_lora_local": PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "lora"
    / "final"
    / "test_preds.csv",
}


def leaderboard():
    """All finished runs with a val score, best first."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_macro_f1 DESC"],
    )
    return runs[runs["metrics.val_macro_f1"].notna()].reset_index(drop=True)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = leaderboard()

    print("Leaderboard (val macro-F1):")
    for _, row in runs.iterrows():
        print(f"  {row['tags.mlflow.runName']:<20} {row['metrics.val_macro_f1']:.4f}")

    winner = runs.iloc[0]
    name, run_id = winner["tags.mlflow.runName"], winner["run_id"]
    print(f"\nWinner: {name} ({run_id})")

    if name not in TEST_PREDS:
        print(f"No saved test predictions for {name}; stopping.")
        return

    # Score the winner on test, once, inside its own run
    test_df = pd.read_csv(TEST_PREDS[name])
    with mlflow.start_run(run_id=run_id):
        test_metrics = log_eval(test_df, "test_", f"{name} (test)")
        mlflow.set_tag("selected", "true")

    # Only one run should be "selected"
    client = MlflowClient()
    if "tags.selected" in runs.columns:
        old = runs[(runs["tags.selected"] == "true") & (runs["run_id"] != run_id)]
        for old_id in old["run_id"]:
            client.set_tag(old_id, "selected", "false")
            print(f"Unselected previous winner: {old_id}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.4f}")


if __name__ == "__main__":
    main()
