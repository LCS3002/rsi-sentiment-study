"""Signal construction: RSI and a continuous news-sentiment score.

Both signals are computed **as of the close of day D** and consumed by the backtest at the
open of D+1. Nothing in this module may look forward: every rolling operation here is
backward-looking by construction, and `tests/test_signals.py` asserts it on a series
where a forward peek would be visible.

RSI is written out rather than imported from a TA library. It is twelve lines, it removes
a numba-dependent package, and it means the arithmetic can be checked against the
textbook definition instead of trusted — which is the whole point of a study whose
conclusion rests on it.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from study.config import (
    MAX_TEXT_CHARS,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_CACHE,
    SENTIMENT_CHUNK_SIZE,
    SENTIMENT_HALFLIFE_DAYS,
    SENTIMENT_MODEL,
)

logger = logging.getLogger(__name__)

_LABEL_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ── RSI ───────────────────────────────────────────────────────────────────────


def wilder_rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder's RSI. SMA-seeded, then recursive 1/n smoothing (TA-Lib convention).

    First valid value lands at index `length`, since `length` deltas require `length + 1`
    closes. A flat stretch gives 0/0 and stays NaN rather than reading as 100 — undefined
    is the honest answer, and the backtest treats NaN as "no signal".
    """
    values = close.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    if values.size < length + 1:
        return pd.Series(out, index=close.index, name=f"rsi_{length}")

    delta = np.diff(values)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = gain[:length].mean()
    avg_loss = loss[:length].mean()

    def rsi_of(g: float, l: float) -> float:
        if l == 0:
            return 100.0 if g > 0 else np.nan
        return 100.0 - 100.0 / (1.0 + g / l)

    out[length] = rsi_of(avg_gain, avg_loss)
    for i in range(length, delta.size):
        avg_gain = (avg_gain * (length - 1) + gain[i]) / length
        avg_loss = (avg_loss * (length - 1) + loss[i]) / length
        out[i + 1] = rsi_of(avg_gain, avg_loss)

    return pd.Series(out, index=close.index, name=f"rsi_{length}")


def add_rsi(bars: pd.DataFrame, lengths: tuple[int, ...]) -> pd.DataFrame:
    """Add an `rsi_<n>` column per requested length, computed within each ticker."""
    bars = bars.sort_values(["ticker", "date"]).copy()

    for length in lengths:
        col = f"rsi_{length}"
        bars[col] = (
            bars.groupby("ticker", group_keys=False)["close"]
            .apply(lambda s, n=length: wilder_rsi(s, n))
            .to_numpy()
        )
        logger.info("RSI(%d): %d of %d rows non-null",
                    length, int(bars[col].notna().sum()), len(bars))

    return bars


# ── Sentiment ─────────────────────────────────────────────────────────────────


def _article_text(news: pd.DataFrame) -> pd.Series:
    """Headline plus the opening of the body, truncated.

    FinBERT truncates at 512 tokens anyway, and in a news article the tradeable content
    sits at the top — so this is a cost decision, not an information one.
    """
    combined = news["title"].fillna("") + ". " + news["content"].fillna("")
    return combined.str.slice(0, MAX_TEXT_CHARS)


def score_articles(news: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Add a continuous `score` = p_positive - p_negative in [-1, 1] to each article.

    Cached to parquet, because scoring the corpus is by far the slowest step in the study
    and never needs doing twice.

    The cache records the model and truncation length alongside each score, and rows
    produced under different settings are discarded rather than reused. Without that, a
    change to MAX_TEXT_CHARS would silently mix scores computed over different amounts of
    text — the kind of inconsistency that produces a result nobody can reproduce.
    """
    key_cols = ["ticker", "date", "title"]
    cache_cols = key_cols + ["score", "model", "max_chars"]

    merged = news.copy()
    merged["score"] = np.nan
    previous = pd.DataFrame(columns=cache_cols)

    if SENTIMENT_CACHE.exists() and not force:
        cached = pd.read_parquet(SENTIMENT_CACHE)

        if {"model", "max_chars"}.issubset(cached.columns):
            valid = cached[
                (cached["model"] == SENTIMENT_MODEL)
                & (cached["max_chars"] == MAX_TEXT_CHARS)
            ]
            stale = len(cached) - len(valid)
            if stale:
                logger.warning(
                    "Discarding %d cached score(s) computed under different settings "
                    "(model or truncation length changed)", stale,
                )
        else:
            logger.warning("Cache predates settings tracking — discarding it")
            valid = pd.DataFrame(columns=cache_cols)

        previous = valid
        merged = merged.drop(columns="score").merge(
            valid[key_cols + ["score"]], on=key_cols, how="left"
        )

    to_score = merged[merged["score"].isna()]
    if to_score.empty:
        logger.info("All %d article(s) already scored", len(merged))
        return merged

    logger.info("%d of %d article(s) need scoring", len(to_score), len(merged))

    # Scored in chunks, persisting after each one. The full corpus takes around two
    # hours on CPU, and writing only at the end means any interruption — a crash, a
    # dead battery — throws all of it away. Checkpointing turns that into losing a
    # few minutes, and makes the job resumable: a rerun picks up from the cache.
    SENTIMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    scored_so_far = previous[cache_cols].copy() if len(previous) else pd.DataFrame(columns=cache_cols)
    positions = list(to_score.index)
    total = len(positions)

    for start in range(0, total, SENTIMENT_CHUNK_SIZE):
        chunk_index = positions[start : start + SENTIMENT_CHUNK_SIZE]
        chunk = merged.loc[chunk_index]

        merged.loc[chunk_index, "score"] = _run_finbert(_article_text(chunk).tolist())

        done = merged.loc[chunk_index, key_cols + ["score"]].copy()
        done["model"] = SENTIMENT_MODEL
        done["max_chars"] = MAX_TEXT_CHARS
        scored_so_far = pd.concat([scored_so_far, done[cache_cols]], ignore_index=True)

        scored_so_far.drop_duplicates(
            subset=key_cols + ["model", "max_chars"], keep="last"
        ).to_parquet(SENTIMENT_CACHE, index=False)

        completed = min(start + SENTIMENT_CHUNK_SIZE, total)
        logger.info(
            "Checkpoint: %d / %d scored (%.0f%%) — cache saved",
            completed, total, 100.0 * completed / total,
        )

    return merged


@lru_cache(maxsize=1)
def _classifier():
    """Built once and reused. Now that scoring runs chunk by chunk this is called
    repeatedly, and rebuilding the pipeline each time would reload the weights on every
    checkpoint."""
    from transformers import pipeline

    logger.info("Loading %s", SENTIMENT_MODEL)
    return pipeline(
        "sentiment-analysis", model=SENTIMENT_MODEL, top_k=None, truncation=True
    )


def _run_finbert(texts: list[str]) -> np.ndarray:
    """FinBERT over `texts`, returning p_positive - p_negative per text."""
    out = np.zeros(len(texts), dtype=float)
    results = _classifier()(texts, batch_size=SENTIMENT_BATCH_SIZE)

    for i, entry in enumerate(results):
        if isinstance(entry, dict):
            entry = [entry]
        signed = 0.0
        for item in entry:
            sign = _LABEL_SIGN.get(str(item.get("label", "")).lower())
            if sign is None:
                logger.warning("Unexpected label %r — ignored", item.get("label"))
                continue
            signed += sign * float(item.get("score", 0.0))
        out[i] = signed

    return out


def daily_sentiment(
    scored: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    halflife: float = SENTIMENT_HALFLIFE_DAYS,
) -> pd.DataFrame:
    """Per ticker-day sentiment, decayed, on the trading calendar.

    Three steps, each with a reason:

    1. **Mean across same-day articles.** Multiple stories about one name on one day are
       repeated observations of the same event, not independent signals, so they average
       rather than accumulate.
    2. **Roll news dated on a non-trading day forward.** A Saturday story is first
       actionable at the next open; dropping it would discard every weekend.
    3. **EWMA with a 3-day half-life.** News does not stop mattering the day after it
       prints. `ewm` is strictly backward-looking, so this cannot leak.

    Returns [ticker, date, sentiment, articles] over `calendar`, where `sentiment` is the
    decayed score known as of that day's close.
    """
    calendar = pd.DatetimeIndex(sorted(calendar))

    per_day = (
        scored.groupby(["ticker", "date"], as_index=False)
        .agg(raw_score=("score", "mean"), articles=("score", "size"))
    )

    # Step 2: snap each news date onto the next trading day at or after it
    positions = calendar.searchsorted(per_day["date"].to_numpy(), side="left")
    inside = positions < len(calendar)
    dropped = int((~inside).sum())
    if dropped:
        logger.info("%d article-day(s) fall after the last trading date — dropped", dropped)
    per_day = per_day[inside].copy()
    per_day["date"] = calendar[positions[inside]]

    # a roll can land two news days on the same trading day; re-aggregate
    per_day = (
        per_day.groupby(["ticker", "date"], as_index=False)
        .agg(raw_score=("raw_score", "mean"), articles=("articles", "sum"))
    )

    frames = []
    for ticker, group in per_day.groupby("ticker"):
        series = (
            group.set_index("date")[["raw_score", "articles"]]
            .reindex(calendar)
        )
        decayed = series["raw_score"].ewm(halflife=halflife, ignore_na=True).mean()

        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": calendar,
                    "sentiment": decayed.to_numpy(),
                    "articles": series["articles"].fillna(0).to_numpy(),
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    logger.info(
        "Daily sentiment: %d ticker-days, %d with a live reading",
        len(out), int(out["sentiment"].notna().sum()),
    )
    return out


def build_features(
    bars: pd.DataFrame,
    sentiment: pd.DataFrame,
    rsi_lengths: tuple[int, ...],
) -> pd.DataFrame:
    """Bars + RSI + sentiment on one [ticker, date] grid, plus next-day open.

    `next_open` is the price a signal from this row actually executes at. Carrying it as
    an explicit column — rather than shifting returns somewhere downstream — keeps the
    one-bar delay visible in the data instead of buried in the backtest loop.
    """
    features = add_rsi(bars, rsi_lengths)
    features = features.merge(sentiment, on=["ticker", "date"], how="left")

    features = features.sort_values(["ticker", "date"])
    grouped = features.groupby("ticker", group_keys=False)
    features["next_open"] = grouped["open"].shift(-1)
    features["next_date"] = grouped["date"].shift(-1)

    return features.reset_index(drop=True)
