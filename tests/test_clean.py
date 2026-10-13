import pandas as pd

from finsent.data.clean import clean_dataframe, clean_text


def make_df(rows):
    """Build a small DataFrame in the same shape as tfns.csv."""
    return pd.DataFrame(rows, columns=["text", "label", "split"])


# ---------- clean_text ----------


def test_removes_urls():
    assert clean_text("$AAPL up https://t.co/abc123 today") == "$AAPL up today"


def test_decodes_html_entities():
    assert clean_text("$AAPL &amp; $MSFT") == "$AAPL & $MSFT"


def test_collapses_whitespace_and_strips_ends():
    assert clean_text("  profit   rose\n\ttoday  ") == "profit rose today"


def test_keeps_tickers_casing_and_numbers():
    # Guards against someone "helpfully" adding lowercasing or
    # punctuation stripping later — FinBERT needs these.
    assert clean_text("$TSLA DOWN 12%") == "$TSLA DOWN 12%"


# ---------- clean_dataframe ----------


def test_drops_null_text_or_label():
    df = make_df(
        [
            (None, "positive", "train"),
            ("$AAPL up", None, "train"),
            ("$MSFT flat", "neutral", "train"),
        ]
    )
    out = clean_dataframe(df)
    assert list(out["text"]) == ["$MSFT flat"]


def test_drops_tweets_that_were_only_a_url():
    df = make_df(
        [
            ("https://t.co/abc123", "neutral", "train"),
            ("$AAPL up", "positive", "train"),
        ]
    )
    out = clean_dataframe(df)
    assert list(out["text"]) == ["$AAPL up"]


def test_exact_duplicates_keep_one_copy():
    df = make_df(
        [
            ("$AAPL up", "positive", "train"),
            ("$AAPL up", "positive", "train"),
        ]
    )
    out = clean_dataframe(df)
    assert len(out) == 1


def test_conflicting_labels_dropped():
    df = make_df(
        [
            ("$TSLA recall", "negative", "train"),
            ("$TSLA recall", "positive", "train"),
            ("$AAPL up", "positive", "train"),
        ]
    )
    out = clean_dataframe(df)
    assert list(out["text"]) == ["$AAPL up"]


def test_train_validation_overlap_keeps_validation_copy():
    df = make_df(
        [
            ("Fed holds rates", "neutral", "train"),
            ("Fed holds rates", "neutral", "validation"),
        ]
    )
    out = clean_dataframe(df)
    # Checking the split, not just the count — we need to know
    # WHICH copy survived.
    assert list(out["split"]) == ["validation"]


def test_does_not_modify_input():
    df = make_df(
        [
            ("$AAPL down", "negative", "train"),
            ("$AAPL up", "positive", "train"),
        ]
    )
    before = df.copy()
    c = clean_dataframe(df)
    pd.testing.assert_frame_equal(c, before)
