"""Sentiment validation: forward-return alignment and whether the IC detects real signal.

Forward returns are a second look-ahead surface — easier to get wrong than the backtest,
because an off-by-one here silently credits the signal with a move it could not have
traded, and the result looks better rather than broken.
"""

import numpy as np
import pandas as pd
import pytest

from study.signals import build_features
from study.validation import (
    add_forward_returns,
    information_coefficient,
    quintile_spread,
    score_distribution,
)

from .conftest import make_bars, make_sentiment


# ── Forward-return alignment ──────────────────────────────────────────────────


def test_forward_return_runs_from_the_next_open(features):
    """fwd_1d on row D must be open(D+2)/open(D+1) - 1 — the trade the backtest makes."""
    out = add_forward_returns(features, (1,))

    for ticker, group in out.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        opens = group["open"].to_numpy()
        actual = group["fwd_1d"].to_numpy()
        # row i trades at open[i+1] and exits at open[i+2]
        expected = opens[2:] / opens[1:-1] - 1.0
        np.testing.assert_allclose(actual[: len(expected)], expected, atol=1e-12)


def test_forward_returns_are_nan_at_the_tail(features):
    """There is no price after the last bar, so the tail must be NaN, not carried."""
    out = add_forward_returns(features, (5,))
    for _, group in out.groupby("ticker"):
        assert group.sort_values("date")["fwd_5d"].iloc[-6:].isna().all()


def test_forward_returns_do_not_leak_across_tickers():
    bars = make_bars(["AAA", "BBB"], n_days=60)
    out = add_forward_returns(build_features(bars, make_sentiment(bars), (14,)), (1, 5))

    tail = out.sort_values("date").groupby("ticker").tail(1)
    assert tail["fwd_1d"].isna().all()
    assert tail["fwd_5d"].isna().all()


def test_future_prices_cannot_change_an_earlier_forward_return(features):
    """Perturbing the tail must leave earlier forward returns untouched beyond the
    horizon's own reach."""
    base = add_forward_returns(features, (1,))

    tampered_input = features.copy()
    cutoff = sorted(tampered_input["date"].unique())[-10]
    tampered_input.loc[tampered_input["date"] >= cutoff, ["open", "next_open"]] *= 1.5
    tampered = add_forward_returns(tampered_input, (1,))

    # rows more than two sessions before the cutoff cannot reach the perturbed prices
    safe = base["date"] < cutoff - pd.Timedelta(days=6)
    np.testing.assert_allclose(
        base.loc[safe, "fwd_1d"].fillna(0).to_numpy(),
        tampered.loc[safe, "fwd_1d"].fillna(0).to_numpy(),
        atol=1e-12,
    )


# ── Does the IC detect signal that is actually there? ─────────────────────────


def _planted(n_days=200, n_names=30, strength=0.0, seed=0):
    """A panel where sentiment explains `strength` of the next-open-to-open return."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    rows = []

    for name in range(n_names):
        sentiment = rng.uniform(-1, 1, n_days)
        noise = rng.normal(0, 0.02, n_days)
        forward = strength * sentiment + noise
        # build an open path whose realised forward return matches `forward`
        opens = 100 * np.cumprod(np.concatenate([[1.0], 1 + forward[:-1]]))
        rows.append(
            pd.DataFrame(
                {
                    "ticker": f"T{name:02d}",
                    "date": dates,
                    "next_open": opens,
                    "sentiment": sentiment,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def test_ic_is_near_zero_when_sentiment_is_unrelated_to_returns():
    panel = add_forward_returns(_planted(strength=0.0, seed=1), (1,))
    table = information_coefficient(panel, (1,))

    assert table.iloc[0]["days"] > 150
    assert abs(table.iloc[0]["mean_ic"]) < 0.05
    assert abs(table.iloc[0]["ic_tstat"]) < 3.0


def test_ic_is_strongly_positive_when_sentiment_genuinely_predicts():
    """If this does not fire, the measure cannot detect a real effect and any null result
    from it would be meaningless."""
    panel = add_forward_returns(_planted(strength=0.03, seed=2), (1,))
    table = information_coefficient(panel, (1,))

    assert table.iloc[0]["mean_ic"] > 0.3
    assert table.iloc[0]["ic_tstat"] > 5.0
    assert table.iloc[0]["share_positive"] > 0.8


def test_ic_sign_follows_the_sign_of_the_relationship():
    panel = _planted(strength=0.03, seed=3)
    panel["sentiment"] *= -1          # invert the signal, keep the returns
    table = information_coefficient(add_forward_returns(panel, (1,)), (1,))

    assert table.iloc[0]["mean_ic"] < -0.3


def test_ic_skips_days_with_too_few_names():
    panel = add_forward_returns(_planted(n_names=4, strength=0.0, seed=4), (1,))
    table = information_coefficient(panel, (1,), min_names=10)
    assert table.iloc[0]["days"] == 0


def test_ic_reports_zero_days_rather_than_crashing_on_empty_input():
    empty = pd.DataFrame(columns=["ticker", "date", "sentiment", "next_open", "fwd_1d"])
    table = information_coefficient(empty, (1,))
    assert table.iloc[0]["days"] == 0


# ── Quintile spread ───────────────────────────────────────────────────────────


def test_quintile_spread_is_monotone_on_planted_signal():
    panel = add_forward_returns(_planted(strength=0.03, seed=5), (5,))
    table = quintile_spread(panel, horizon=5)

    assert len(table) == 5
    assert table["mean_sentiment"].is_monotonic_increasing
    assert table["mean_forward_return"].iloc[-1] > table["mean_forward_return"].iloc[0]


def test_quintile_spread_is_flat_on_noise():
    panel = add_forward_returns(_planted(strength=0.0, seed=6), (5,))
    table = quintile_spread(panel, horizon=5)

    spread = (
        table["mean_forward_return"].iloc[-1] - table["mean_forward_return"].iloc[0]
    )
    assert abs(spread) < 0.01


def test_quintile_spread_is_empty_without_data():
    empty = pd.DataFrame(columns=["ticker", "date", "sentiment", "next_open", "fwd_5d"])
    assert quintile_spread(empty, horizon=5).empty


# ── Distribution sanity ───────────────────────────────────────────────────────


def test_score_distribution_reports_coverage_and_spread():
    panel = _planted(strength=0.0, seed=7)
    panel.loc[panel.index[:500], "sentiment"] = np.nan

    stats = score_distribution(panel)
    assert stats["scored_ticker_days"] == len(panel) - 500
    assert 0.0 < stats["coverage"] < 1.0
    assert stats["std"] > 0
    assert stats["p05"] < stats["p50"] < stats["p95"]


def test_score_distribution_handles_an_entirely_unscored_panel():
    panel = _planted(seed=8)
    panel["sentiment"] = np.nan
    assert score_distribution(panel)["scored_ticker_days"] == 0
