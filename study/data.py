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
import time

import pandas as pd

from study.config import (
    BAR_FETCH_ATTEMPTS,
    BAR_RETRY_DELAY_SECONDS,
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
    """yfinance's wide, column-MultiIndex frame → long [ticker, date, ohlcv].

    Two yfinance quirks this has to absorb, both discovered the hard way:

    · The frame **sometimes** carries an `Adj Close` level alongside `Close` even with
      `auto_adjust=True` — specifically when one of the requested tickers failed to
      download, which changes the shape of what comes back. Under `auto_adjust=True`
      `Close` is already adjusted, so the duplicate is dropped rather than renamed onto
      it. Renaming would produce two columns called `close`, and every later column
      access raises "cannot reindex on an axis with duplicate labels".

    · A single ticker comes back with no ticker level at all.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["ticker", "date"] + BAR_COLUMNS)

    if isinstance(raw.columns, pd.MultiIndex):
        long = raw.stack(level=1, future_stack=True).reset_index()
    else:
        long = raw.reset_index()

    long.columns = [str(c).strip().lower() for c in long.columns]

    # Identify the date and ticker columns by name, falling back to position
    renames = {}
    if "date" not in long.columns:
        for candidate in ("datetime", "index", "level_0"):
            if candidate in long.columns:
                renames[candidate] = "date"
                break
        else:
            renames[long.columns[0]] = "date"
    if "ticker" not in long.columns:
        for candidate in ("symbol", "level_1"):
            if candidate in long.columns:
                renames[candidate] = "ticker"
                break
    long = long.rename(columns=renames)

    if "close" in long.columns and "adj close" in long.columns:
        long = long.drop(columns="adj close")
    elif "adj close" in long.columns:
        long = long.rename(columns={"adj close": "close"})

    if "ticker" not in long.columns:
        if len(tickers) != 1:
            raise RuntimeError(
                f"yfinance returned no ticker column for {len(tickers)} tickers"
            )
        long["ticker"] = tickers[0]

    duplicated = long.columns[long.columns.duplicated()].tolist()
    if duplicated:
        raise RuntimeError(f"duplicate columns after flattening: {duplicated}")

    missing = [c for c in BAR_COLUMNS if c not in long.columns]
    if missing:
        raise RuntimeError(f"yfinance returned no {missing} columns")

    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    long["ticker"] = long["ticker"].astype(str).str.upper()

    return long[["ticker", "date"] + BAR_COLUMNS]


def _download(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,      # adjusts O/H/L/C on one basis — see module docstring
        progress=False,
        group_by="column",
        threads=True,
    )
    bars = _flatten_yf(raw, tickers)
    bars = bars.dropna(subset=["open", "close"])
    bars = bars[bars["close"] > 0]
    return bars.drop_duplicates(subset=["ticker", "date"])


def load_bars(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force: bool = False,
    attempts: int = BAR_FETCH_ATTEMPTS,
) -> pd.DataFrame:
    """Adjusted daily bars as [ticker, date, open, high, low, close, volume].

    `start` is pulled back by BAR_WARMUP_DAYS so indicators are warm on the first day a
    signal could fire — otherwise the earliest trades run on a half-formed RSI.

    Two behaviours that exist because of a specific failure. Yahoo intermittently returns
    "possibly delisted; no price data found" for names that are plainly listed — MSFT came
    back empty on one run — so tickers that come back short are retried individually
    before being given up on. And new bars are **merged into** the cache rather than
    replacing it, so a bad fetch can never delete good data that is already there. The
    earlier version did neither, and silently dropped MSFT from an 86-name universe.
    """
    requested = sorted(set(tickers))
    fetch_start = pd.Timestamp(start) - pd.Timedelta(days=BAR_WARMUP_DAYS)
    fetch_end = pd.Timestamp(end) + pd.Timedelta(days=1)  # yfinance `end` is exclusive

    cached = pd.DataFrame(columns=["ticker", "date"] + BAR_COLUMNS)
    if BARS_CACHE.exists() and not force:
        cached = pd.read_parquet(BARS_CACHE)

    have = set(cached["ticker"].unique())
    missing = [t for t in requested if t not in have]

    if not missing:
        logger.info("Loading cached bars from %s", BARS_CACHE)
        return cached[cached["ticker"].isin(requested)].reset_index(drop=True)

    logger.info(
        "Downloading bars for %d ticker(s), %s to %s",
        len(missing), fetch_start.date(), pd.Timestamp(end).date(),
    )
    fetched = _download(missing, fetch_start, fetch_end)
    still_missing = [t for t in missing if t not in set(fetched["ticker"].unique())]

    # Retry one at a time: a batch request that partially fails tends to succeed when the
    # stragglers are asked for on their own.
    for attempt in range(2, attempts + 1):
        if not still_missing:
            break
        logger.warning(
            "Attempt %d: retrying %d ticker(s) individually — %s",
            attempt, len(still_missing), still_missing,
        )
        recovered = []
        for ticker in still_missing:
            time.sleep(BAR_RETRY_DELAY_SECONDS)
            one = _download([ticker], fetch_start, fetch_end)
            if not one.empty:
                fetched = pd.concat([fetched, one], ignore_index=True)
                recovered.append(ticker)
        still_missing = [t for t in still_missing if t not in recovered]
        if recovered:
            logger.info("Recovered on retry: %s", recovered)

    if still_missing:
        logger.warning(
            "No price data after %d attempts for %s — excluded from the universe. "
            "Verify these are genuinely delisted rather than a transient API failure.",
            attempts, still_missing,
        )

    # Union, keeping whatever was already cached — a failed fetch must never delete data
    pieces = [frame for frame in (cached, fetched) if not frame.empty]
    bars = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=["ticker", "date"] + BAR_COLUMNS)
    )
    bars = (
        bars.drop_duplicates(subset=["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(BARS_CACHE, index=False)

    out = bars[bars["ticker"].isin(requested)].reset_index(drop=True)
    logger.info("Bars: %d rows across %d tickers", len(out), out["ticker"].nunique())
    return out


def trading_calendar(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Every date on which any name in the sample traded."""
    return pd.DatetimeIndex(sorted(bars["date"].unique()))
