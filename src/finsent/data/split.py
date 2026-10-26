import pandas as pd
from sklearn.model_selection import train_test_split

from finsent.config import PROCESSED_DIR

CLEAN_PATH = PROCESSED_DIR / "tfns_clean.csv"
VAL_SIZE = 0.15
RANDOM_STATE = 42


def load_splits():
    df = pd.read_csv(CLEAN_PATH)

    unexpected = set(df["split"].unique()) - {"train", "validation"}
    if unexpected:
        raise ValueError(f"Unexpected split values: {unexpected}")

    test = df[df["split"] == "validation"]
    pool = df[df["split"] == "train"]

    train, val = train_test_split(
        pool,
        test_size=VAL_SIZE,
        stratify=pool["label"],
        random_state=RANDOM_STATE,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )