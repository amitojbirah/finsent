"""Download Financial PhraseBank and save it to data/raw/."""

import pandas as pd
from datasets import load_dataset

from finsent.config import RAW_DIR

DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"


LABEL_NAMES = {0: "negative", 1: "positive", 2: "neutral"}


def download_tfns() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME)

    frames = []
    for split_name in ["train", "validation"]:
        part = ds[split_name].to_pandas()
        part["split"] = split_name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    unknown = set(df["label"]) - set(LABEL_NAMES)
    if unknown:
        raise ValueError(f"Unexpected label values: {unknown}")

    df["label"] = df["label"].map(LABEL_NAMES)
    return df[["text", "label", "split"]]


def save_raw(df: pd.DataFrame, filename: str) -> None:
    raw_path = RAW_DIR / filename
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)


if __name__ == "__main__":
    df = download_tfns()
    print(df.shape)
    print(df["label"].value_counts())
    save_raw(df, "tfns.csv")
