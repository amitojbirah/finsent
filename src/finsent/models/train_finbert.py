"""Fine-tune FinBERT locally (full or LoRA) and log the run to MLflow."""

import argparse

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from finsent.config import MLFLOW_TRACKING_URI, PROJECT_ROOT
from finsent.data.split import load_splits
from finsent.models.log_finbert import log_eval
from finsent.models.train_baseline import EXPERIMENT_NAME

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 64
SEED = 42
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"

# Same for both modes -> fair comparison
COMMON = {"per_device_train_batch_size": 32, "warmup_steps": 100, "weight_decay": 0.01}
# Each method's standard settings
MODES = {
    "full": {"learning_rate": 2e-5, "num_train_epochs": 4},
    "lora": {"learning_rate": 2e-4, "num_train_epochs": 5},
}
LORA = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["query", "value"],
}


def build_datasets(tokenizer, label2id):
    """Tokenized train/val/test Datasets (test without labels) + the val/test DataFrames."""
    train, val, test = load_splits()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    def to_dataset(df, with_labels):
        cols = {"text": df["text"].tolist()}
        if with_labels:
            ids = df["label"].map(label2id)
            if ids.isna().any():
                raise ValueError("Some labels are missing from the model's label2id")
            cols["labels"] = ids.astype(int).tolist()
        return Dataset.from_dict(cols).map(
            tokenize, batched=True, remove_columns=["text"]
        )

    datasets = {
        "train": to_dataset(train, with_labels=True),
        "val": to_dataset(val, with_labels=True),
        "test": to_dataset(test, with_labels=False),  # unscored until selection
    }
    return datasets, val, test


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def predictions_df(trainer, dataset, df, id2label):
    """Predict a split and return text / label / pred / confidence, like the Colab CSVs."""
    logits = trainer.predict(dataset).predictions
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    return pd.DataFrame(
        {
            "text": df["text"].values,
            "label": df["label"].values,
            "pred": [id2label[int(i)] for i in probs.argmax(axis=1)],
            "confidence": probs.max(axis=1),
        }
    )


def folder_size_mb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT (full or LoRA).")
    parser.add_argument("--mode", choices=["full", "lora"], required=True)
    mode = parser.parse_args().mode

    set_seed(SEED)
    hp = {**COMMON, **MODES[mode]}
    out_dir = RUNS_DIR / mode
    run_name = f"finbert_{mode}_local"

    # Model: always start from the ORIGINAL FinBERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    label2id, id2label = model.config.label2id, model.config.id2label
    if mode == "lora":
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA))
    trainable, total = count_params(model)
    print(
        f"{mode}: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)"
    )

    datasets, val_df, test_df = build_datasets(tokenizer, label2id)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        **hp,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        # Params
        params = {
            "mode": mode,
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            **hp,
        }
        if mode == "lora":
            params.update({f"lora_{k}": v for k, v in LORA.items()})
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": 100 * trainable / total,
            }
        )

        # Train (and measure time + peak GPU memory)
        torch.cuda.reset_peak_memory_stats()
        train_result = trainer.train()
        mlflow.log_metric("train_runtime_s", train_result.metrics["train_runtime"])
        mlflow.log_metric(
            "peak_gpu_mem_mb", torch.cuda.max_memory_allocated() / 1024**2
        )

        # Per-epoch curves
        for entry in trainer.state.log_history:
            if "eval_macro_f1" in entry:
                epoch = round(entry["epoch"])
                mlflow.log_metric(
                    "epoch_val_macro_f1", entry["eval_macro_f1"], step=epoch
                )
                mlflow.log_metric("epoch_val_loss", entry["eval_loss"], step=epoch)
            elif "loss" in entry:
                mlflow.log_metric("train_loss", entry["loss"], step=entry["step"])

        # Val metrics of the kept (best-epoch) model, scored like every other run
        val_preds = predictions_df(trainer, datasets["val"], val_df, id2label)
        metrics = log_eval(val_preds, "val_", f"{run_name} (val)")

        # Save model (LoRA: adapter only) and measure its size
        final_dir = out_dir / "final"
        trainer.save_model(str(final_dir))
        mlflow.log_metric("saved_size_mb", folder_size_mb(final_dir))

        # Predictions files (test saved, NOT scored)
        val_preds.to_csv(final_dir / "val_preds.csv", index=False)
        predictions_df(trainer, datasets["test"], test_df, id2label).to_csv(
            final_dir / "test_preds.csv", index=False
        )

        mlflow.log_artifacts(str(final_dir), artifact_path="model_files")
        mlflow.set_tag("trained_on", "rtx4060-laptop")

        print(
            f"\n{run_name}  val_macro_f1 = {metrics['macro_f1']:.4f}  (run {run.info.run_id})"
        )


if __name__ == "__main__":
    main()
