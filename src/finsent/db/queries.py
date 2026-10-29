import pandas as pd
from sqlalchemy import text


def count_tweets(engine):
    with engine.connect() as conn:
        va = conn.execute(text("SELECT count(*) FROM tweets "))
        return va.scalar_one()


def ticker_scores(engine, min_tweets):
    sql = """
        WITH ticker_stats AS (
            SELECT   tt.ticker,
                     COUNT(*) AS total,
                     COUNT(*) FILTER (WHERE t.label = 'positive') AS positive_count,
                     COUNT(*) FILTER (WHERE t.label = 'neutral')  AS neutral_count,
                     COUNT(*) FILTER (WHERE t.label = 'negative') AS negative_count
            FROM     tweet_tickers tt
            JOIN     tweets t ON t.id = tt.tweet_id
            GROUP BY tt.ticker
            HAVING   COUNT(*) >= :min_tweets
        )
        SELECT   ticker,
                 total,
                 ROUND((positive_count - negative_count) * 1.0 / total, 2) AS score
        FROM     ticker_stats
        ORDER BY score DESC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"min_tweets": min_tweets})
    return df


def save_prediction(
    engine, input_text, predicted_label, confidence, model_version, tweet_id=None
):
    sql = """
        INSERT INTO predictions
            (tweet_id, input, model_version, predicted_label, confidence)
        VALUES
            (:tweet_id, :input_text, :model_version, :predicted_label, :confidence)
        RETURNING id
    """
    params = {
        "tweet_id": tweet_id,
        "input_text": input_text,
        "model_version": model_version,
        "predicted_label": predicted_label,
        "confidence": confidence,
    }
    with engine.begin() as conn:
        result = conn.execute(text(sql), params)
        return result.scalar_one()
