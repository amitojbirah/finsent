import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

LABELS = ["negative", "neutral", "positive"]


def evaluate(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
    }
    for label, score in zip(LABELS, per_class):
        metrics[f"f1_{label}"] = float(score)
    return metrics


def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(
        ax=ax, values_format="d", cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig
