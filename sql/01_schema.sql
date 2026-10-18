Create table tweets (
    id bigint generated always as identity primary key,
    text text not null,
    label text not null check (label in ('negative', 'neutral', 'positive')),
    split text not null check (split in ('train', 'validation'))
);

Create table tweet_tickers(
    tweet_id bigint references tweets(id) on delete cascade,
    ticker text  not null check (upper(ticker)=ticker),
    primary key(tweet_id, ticker)
);

Create table predictions(
    id bigint generated always as identity primary key,
    tweet_id bigint references tweets(id) on delete set null,
    input text not null,
    model_version text not null,
    predicted_label text not null check (predicted_label in ('negative', 'neutral', 'positive')),
    confidence real check (confidence >= 0 and confidence <= 1),
    created_at timestamptz  default now() not null
);

create index idx_tweets_tickers on tweet_tickers(ticker);