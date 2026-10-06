"""Data loading: the news corpus and daily bars, both cached to parquet.

Two conventions set here that everything downstream depends on:

**Dates are tz-naive and normalised to midnight.** The news corpus carries date-level
timestamps only — no time of day — so a tz-aware index would imply precision that does
not exist. Bars are localised the same way so the two join cleanly.

**Bars are split- and dividend-adjusted** (`auto_adjust=True`), which adjusts open, high,
low and close on the same basis. That matters because execution happens at the open: a
mix of adjusted closes and raw opens would silently manufacture returns across every
split in the sample.
"""

from __future__ import annotations

import logging

import pandas as pd

from study.config import (
    BAR_WARMUP_DAYS,
    BARS_CACHE,
    DATA_DIR,
    MIN_ARTICLES_PER_TICKER,
    MIN_YEARS_PER_TICKER,
    NEWS_CACHE,
    NEWS_DATASET,
)

logger = logging.getLogger(__name__)

NEWS_COLUMNS = ["ticker", "date", "title", "content"]
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


# ── News ──────────────────────────────────────────────────────────────────────


def _download_news() -> pd.DataFrame:
    from datasets import load_dataset

    logger.info("Downloading %s from HuggingFace", NEWS_DATASET)
    dataset = load_dataset(NEWS_DATASET)

    # The train/test split is the dataset author's, for a modelling task that is not
    # ours; pooling them is correct here and the split carries no time ordering.
    frames = [
        split.to_pandas()[["date", "stock", "title", "content"]]
        for split in dataset.values()
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"stock": "ticker"})


def load_news(force: bool = False) -> pd.DataFrame:
    """All articles as [ticker, date, title, content], deduplicated and date-parsed."""
    if NEWS_CACHE.exists() and not force:
        logger.info("Loading cached news from %s", NEWS_CACHE)
        return pd.read_parquet(NEWS_CACHE)

    df = _download_news()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)

    before = len(df)
    df = df.dropna(subset=["date"])
    df = df[df["ticker"].str.len() > 0]
    # The corpus contains a small number of exact repeats of the same story
    df = df.drop_duplicates(subset=["ticker", "date", "title"])
    logger.info("News: %d articles (%d dropped as undated or duplicate)",
                len(df), before - len(df))

    df = df[NEWS_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(NEWS_CACHE, index=False)
    return df


def usable_tickers(news: pd.DataFrame) -> list[str]:
    """Tickers with enough articles and enough span to say anything measurable.

    Thresholds are in config and were set before any result was computed.
    """
    stats = news.groupby("ticker")["date"].agg(["size", "min", "max"])
    stats["years"] = (stats["max"] - stats["min"]).dt.days / 365.25

    keep = stats[
        (stats["size"] >= MIN_ARTICLES_PER_TICKER)
        & (stats["years"] >= MIN_YEARS_PER_TICKER)
    ]
    tickers = sorted(keep.index.tolist())

    logger.info(
        "Universe: %d of %d tickers clear >=%d articles and >=%.0f years",
        len(tickers), len(stats), MIN_ARTICLES_PER_TICKER, MIN_YEARS_PER_TICKER,
    )
    return tickers


# ── Bars ──────────────────────────────────────────────────────────────────────


def _flatten_yf(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """yfinance's wide, column-MultiIndex frame → long [ticker, date, ohlcv]."""
    if raw.empty:
        return pd.DataFrame(columns=["ticker", "date"] + BAR_COLUMNS)

    if isinstance(raw.columns, pd.MultiIndex):
        # columns are (field, ticker); stack the ticker level into the index
        long = raw.stack(level=1, future_stack=True).reset_index()
        long.columns = [str(c) for c in long.columns]
        long = long.rename(columns={long.columns[0]: "date", long.columns[1]: "ticker"})
    else:
        # single ticker: yfinance omits the ticker level entirely
        long = raw.reset_index()
        long.columns = [str(c) for c in long.columns]
        long = long.rename(columns={long.columns[0]: "date"})
        long["ticker"] = tickers[0]

    long.columns = [c.lower() for c in long.columns]
    long = long.rename(columns={"adj close": "close"})

    missing = [c for c in BAR_COLUMNS if c not in long.columns]
    if missing:
        raise RuntimeError(f"yfinance returned no {missing} columns")

    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    long["ticker"] = long["ticker"].astype(str).str.upper()

    return long[["ticker", "date"] + BAR_COLUMNS]


def load_bars(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force: bool = False,
) -> pd.DataFrame:
    """Adjusted daily bars as [ticker, date, open, high, low, close, volume].

    `start` is pulled back by BAR_WARMUP_DAYS so indicators are already warm on the first
    day any signal could fire — otherwise the earliest trades would be taken on a
    half-formed RSI.
    """
    cache_key = BARS_CACHE
    if cache_key.exists() and not force:
        cached = pd.read_parquet(cache_key)
        have = set(cached["ticker"].unique())
        if set(tickers).issubset(have):
            logger.info("Loading cached bars from %s", cache_key)
            return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)
        logger.info("Cache is missing %d ticker(s) — refetching",
                    len(set(tickers) - have))

    import yfinance as yf

    fetch_start = pd.Timestamp(start) - pd.Timedelta(days=BAR_WARMUP_DAYS)
    # yfinance treats `end` as exclusive
    fetch_end = pd.Timestamp(end) + pd.Timedelta(days=1)

    logger.info("Downloading bars for %d ticker(s), %s to %s",
                len(tickers), fetch_start.date(), pd.Timestamp(end).date())

    raw = yf.download(
        tickers,
        start=fetch_start.strftime("%Y-%m-%d"),
        end=fetch_end.strftime("%Y-%m-%d"),
        auto_adjust=True,      # adjusts O/H/L/C on one basis — see module docstring
        progress=False,
        group_by="column",
        threads=True,
    )

    bars = _flatten_yf(raw, tickers)
    bars = bars.dropna(subset=["open", "close"])
    bars = bars[bars["close"] > 0]
    bars = bars.drop_duplicates(subset=["ticker", "date"])
    bars = bars.sort_values(["ticker", "date"]).reset_index(drop=True)

    got = bars["ticker"].nunique()
    if got < len(tickers):
        logger.warning(
            "Bars returned for %d of %d requested tickers — %s had none",
            got, len(tickers), sorted(set(tickers) - set(bars["ticker"].unique())),
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(cache_key, index=False)
    logger.info("Bars: %d rows across %d tickers", len(bars), got)
    return bars


def trading_calendar(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Every date on which any name in the sample traded."""
    return pd.DatetimeIndex(sorted(bars["date"].unique()))
