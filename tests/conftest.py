"""Shared fixtures. Nothing here touches the network or loads a model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_bars(tickers, n_days=120, seed=0, start="2021-01-04"):
    """Synthetic daily bars on a business-day calendar, in the loader's schema."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    frames = []

    for i, ticker in enumerate(tickers):
        close = 100 + np.cumsum(rng.normal(0.02, 1.2, n_days)) + i
        close = np.maximum(close, 1.0)
        # open gaps slightly from the prior close, so open != close everywhere
        open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.004, n_days - 1))])
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) + 0.5,
                    "low": np.minimum(open_, close) - 0.5,
                    "close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, n_days),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def make_sentiment(bars, seed=1):
    """A sentiment column on the same grid, with gaps so NaN handling gets exercised."""
    rng = np.random.default_rng(seed)
    out = bars[["ticker", "date"]].copy()
    values = rng.normal(0, 0.4, len(out)).clip(-1, 1)
    values[rng.random(len(out)) < 0.3] = np.nan     # 30% of days have no live reading
    out["sentiment"] = values
    out["articles"] = rng.integers(0, 4, len(out))
    return out


# The down-leg must be longer than the RSI window, or the lookback always contains enough
# gains to hold RSI up. With 8 down days inside a 14-day window RSI bottoms around 37 and
# never triggers — so the decline runs 16 days.
SAW_DOWN = 16
SAW_UP = 10
SAW_CYCLE = SAW_DOWN + SAW_UP


def make_sawtooth_bars(tickers, cycles=9, start="2021-01-04"):
    """Bars that deterministically drive RSI(14) well below 30 once per cycle.

    Random walks only rarely produce the oversold-plus-negative-news combination the
    falsification arm needs, so those tests would silently skip — and a skipped
    look-ahead test covers nothing. A sustained 16-day decline fills the whole RSI window
    with losses, pushing RSI near zero, and the 10-day recovery carries it back through 50
    so the reversion exit fires too.
    """
    n_days = cycles * SAW_CYCLE
    dates = pd.bdate_range(start=start, periods=n_days)
    frames = []

    for offset, ticker in enumerate(tickers):
        close = []
        level = 100.0 + offset
        for _ in range(cycles):
            for _ in range(SAW_DOWN):
                level *= 0.98
                close.append(level)
            for _ in range(SAW_UP):
                level *= 1.032
                close.append(level)

        close = np.asarray(close[:n_days], dtype=float)
        open_ = np.concatenate([[close[0]], close[:-1] * 1.001])
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.004,
                    "low": np.minimum(open_, close) * 0.996,
                    "close": close,
                    "volume": 2_000_000,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def make_structured_sentiment(bars):
    """Sentiment keyed to the sawtooth cycle, so regimes align with the oversold days.

    Cycle 0 carries negative news, cycle 1 none at all, cycle 2 positive, repeating. Tying
    this to the same cycle length as the price path is what guarantees the fixture contains
    oversold days *with* negative news, oversold days with *no* news, and positive days —
    rather than hoping two independent periodicities happen to line up.
    """
    out = bars[["ticker", "date"]].copy()
    values = np.full(len(out), np.nan)

    for _, index in out.groupby("ticker").groups.items():
        positions = np.arange(len(index))
        phase = (positions // SAW_CYCLE) % 3
        block = np.full(len(index), np.nan)
        block[phase == 0] = -0.70      # negative news
        block[phase == 1] = np.nan     # no news at all
        block[phase == 2] = 0.55       # clearly positive
        values[out.index.get_indexer(index)] = block

    out["sentiment"] = values
    out["articles"] = np.where(np.isnan(values), 0, 2)
    return out


@pytest.fixture
def bars():
    return make_bars(["AAA", "BBB", "CCC"], n_days=150)


@pytest.fixture
def features(bars):
    """Random-walk fixture — realistic, but not every arm is guaranteed to fire."""
    from study.signals import build_features

    return build_features(bars, make_sentiment(bars), (14, 2))


@pytest.fixture
def all_arms_features():
    """Fixture engineered so every arm, including the falsification arm, trades."""
    from study.signals import build_features

    sawtooth = make_sawtooth_bars(["AAA", "BBB"])
    return build_features(sawtooth, make_structured_sentiment(sawtooth), (14, 2))
