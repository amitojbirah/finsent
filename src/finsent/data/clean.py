"""Clean raw tweets and save to data/processed/."""

import html
import re

import pandas as pd

from finsent.config import PROCESSED_DIR, RAW_DIR

URL_PATTERN = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:

    text = html.unescape(text)
    text = URL_PATTERN.sub("", text)
    text = " ".join(text.split())
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text", "label", "split"], keep="first")

    label_counts = df.groupby("text")["label"].nunique()
    conflicting = label_counts[label_counts > 1].index
    df = df[~df["text"].isin(conflicting)]

    val_texts = set(df.loc[df["split"] == "validation", "text"])
    in_train = df["split"] == "train"
    df = df[~(in_train & df["text"].isin(val_texts))]
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = pd.read_csv(RAW_DIR / "tfns.csv")
    df = clean_dataframe(raw)

    print(f"Before: {len(raw)} rows")
    print(f"After:  {len(df)} rows ({len(raw) - len(df)} removed)")
    print(df.groupby("split")["label"].value_counts())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "tfns_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # ok lets see
