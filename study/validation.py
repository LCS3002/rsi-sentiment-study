"""Is the sentiment signal worth anything?

The study's conclusions rest on FinBERT, so FinBERT gets interrogated rather than assumed.

The obvious test is the wrong one. `ProsusAI/finbert` is fine-tuned on the Financial
PhraseBank, so scoring it against that benchmark measures memorisation and returns a
number near 0.97 that means nothing. Most write-ups that quote FinBERT's accuracy are
quoting exactly that.

So the test here is **benchmark-free**: does the sentiment score predict forward returns
in this sample at all? That question cannot be contaminated by what the model was trained
on, because realised returns were not in anyone's training set.

Three measures, in increasing order of how much they could flatter:

1. **Information coefficient** — the cross-sectional rank correlation between sentiment
   today and return over the next h days, averaged across days. This is the honest one:
   it is computed per day across names, so it cannot be inflated by a market-wide move
   that happens to coincide with generally good news.
2. **Quintile spread** — mean forward return of the most positive fifth minus the most
   negative fifth. Readable, but a single extreme name can move it.
3. **Score distribution** — a sanity check. If FinBERT labels almost everything neutral,
   or almost nothing, the signal has no cross-sectional variation to work with whatever
   the other two say.

Forward returns are measured from the **next open**, the same point the backtest executes,
so nothing here is measurable that the strategy could not have traded.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from study.metrics import newey_west_tstat

logger = logging.getLogger(__name__)

HORIZONS = (1, 5, 20)


def add_forward_returns(
    features: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Forward returns from the next open — the price the backtest would actually get.

    `next_open` on row D is the open of D+1, so an h-day forward return runs from the open
    of D+1 to the open of D+1+h. Measuring from the close of D instead would credit the
    signal with an overnight move it could never have captured.
    """
    features = features.sort_values(["ticker", "date"]).copy()
    grouped = features.groupby("ticker", group_keys=False)["next_open"]

    for horizon in horizons:
        features[f"fwd_{horizon}d"] = grouped.shift(-horizon) / features["next_open"] - 1.0

    return features


def information_coefficient(
    features: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily cross-sectional Spearman IC between sentiment and forward return.

    Days with fewer than `min_names` scored names are skipped — a rank correlation across
    three stocks is noise with a decimal point on it.
    """
    rows = []

    for horizon in horizons:
        column = f"fwd_{horizon}d"
        usable = features.dropna(subset=["sentiment", column])

        daily_ic = []
        for _, group in usable.groupby("date"):
            if len(group) < min_names or group["sentiment"].nunique() < 3:
                continue
            ic = group["sentiment"].corr(group[column], method="spearman")
            if np.isfinite(ic):
                daily_ic.append(ic)

        series = pd.Series(daily_ic, dtype=float)
        if series.empty:
            rows.append({"horizon_days": horizon, "days": 0})
            continue

        mean_ic = float(series.mean())
        rows.append(
            {
                "horizon_days": horizon,
                "days": int(len(series)),
                "mean_ic": mean_ic,
                "median_ic": float(series.median()),
                "ic_std": float(series.std(ddof=1)),
                # IC series is autocorrelated when horizons overlap, so HAC again
                "ic_tstat": newey_west_tstat(series, lags=max(horizon, 5)),
                "share_positive": float((series > 0).mean()),
                # the conventional "information ratio" of the IC series
                "ic_ir": float(mean_ic / series.std(ddof=1)) if series.std(ddof=1) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    logger.info("Information coefficient computed over %s horizons", list(horizons))
    return out


def quintile_spread(
    features: pd.DataFrame, horizon: int = 5, bins: int = 5
) -> pd.DataFrame:
    """Mean forward return by sentiment quintile, ranked within each day.

    Ranking within the day rather than pooling matters: pooled buckets would compare a
    positive reading in 2013 against a negative one in 2020 and call the difference
    sentiment, when most of it is the market.
    """
    column = f"fwd_{horizon}d"
    usable = features.dropna(subset=["sentiment", column]).copy()
    if usable.empty:
        return pd.DataFrame()

    def rank_within_day(scores: pd.Series) -> pd.Series:
        if len(scores) < bins or scores.nunique() < bins:
            return pd.Series(np.nan, index=scores.index)
        return pd.qcut(scores.rank(method="first"), bins, labels=False)

    usable["bucket"] = (
        usable.groupby("date", group_keys=False)["sentiment"]
        .apply(rank_within_day)
        .astype(float)
    )
    usable = usable.dropna(subset=["bucket"])
    if usable.empty:
        return pd.DataFrame()

    table = (
        usable.groupby("bucket")
        .agg(
            observations=(column, "size"),
            mean_sentiment=("sentiment", "mean"),
            mean_forward_return=(column, "mean"),
            hit_rate=(column, lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    table["bucket"] = table["bucket"].astype(int) + 1
    return table


def score_distribution(features: pd.DataFrame) -> dict:
    """Does the score have any cross-sectional variation to work with?"""
    scores = features["sentiment"].dropna()
    if scores.empty:
        return {"scored_ticker_days": 0}

    return {
        "scored_ticker_days": int(len(scores)),
        "coverage": float(features["sentiment"].notna().mean()),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "share_strongly_positive": float((scores > 0.20).mean()),
        "share_strongly_negative": float((scores < -0.20).mean()),
        "share_near_neutral": float((scores.abs() <= 0.20).mean()),
        "p05": float(scores.quantile(0.05)),
        "p50": float(scores.quantile(0.50)),
        "p95": float(scores.quantile(0.95)),
    }


def validate(features: pd.DataFrame) -> dict:
    """Everything the validation section of the note reports."""
    enriched = add_forward_returns(features)

    results = {
        "distribution": score_distribution(enriched),
        "information_coefficient": information_coefficient(enriched).to_dict("records"),
        "quintile_spread_5d": quintile_spread(enriched, horizon=5).to_dict("records"),
    }

    ic = results["information_coefficient"]
    for row in ic:
        if row.get("days"):
            logger.info(
                "IC %2dd: mean=%+.4f  t=%+.2f  days=%d",
                row["horizon_days"], row["mean_ic"], row["ic_tstat"], row["days"],
            )

    return results
