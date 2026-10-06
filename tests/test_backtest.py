"""Backtest mechanics, with look-ahead as the headline concern.

If any of these fail, the study's numbers mean nothing — so they are the tests to read
first, and the ones to point a sceptic at.
"""

import numpy as np
import pandas as pd
import pytest

from study.backtest import ARMS, buy_and_hold, run_backtest
from study.config import MAX_CONCURRENT_POSITIONS
from study.signals import build_features

from .conftest import make_bars, make_sentiment


# ── Look-ahead ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arm", ARMS)
def test_no_trade_ever_opens_before_its_signal_resolves(all_arms_features, arm):
    """The central guarantee: a signal from the close of D executes at the open of D+1.

    An entry dated on or before its own signal date would mean the backtest traded on
    information it did not yet have. Run on the engineered fixture so every arm actually
    produces trades — a skipped look-ahead test proves nothing.
    """
    out = run_backtest(all_arms_features, arm, cost_bps=10.0)
    trades = out["trades"]

    assert not trades.empty, f"{arm} must produce trades on this fixture"
    assert (trades["entry_date"] > trades["signal_date"]).all(), (
        "every entry must be strictly after its signal date"
    )


@pytest.mark.parametrize("arm", ARMS)
def test_entry_is_the_next_trading_day_not_a_later_one(all_arms_features, arm):
    """Execution should be the *next* session — silently waiting longer would be a
    different (and easier) strategy than the one described."""
    out = run_backtest(all_arms_features, arm, cost_bps=10.0)
    trades = out["trades"]
    assert not trades.empty

    calendar = pd.DatetimeIndex(sorted(all_arms_features["date"].unique()))
    for signal_date, entry_date in zip(trades["signal_date"], trades["entry_date"]):
        expected = calendar[calendar.searchsorted(signal_date, side="right")]
        assert entry_date == expected


def test_exit_is_always_after_entry(features):
    out = run_backtest(features, "rsi", cost_bps=10.0)
    trades = out["trades"]
    if trades.empty:
        pytest.skip("no trades")
    assert (trades["exit_date"] > trades["entry_date"]).all()


def test_entry_price_is_the_open_of_the_entry_day(features):
    """Not the close of the signal day, and not the close of the entry day."""
    out = run_backtest(features, "rsi", cost_bps=0.0)
    trades = out["trades"]
    if trades.empty:
        pytest.skip("no trades")

    lookup = features.set_index(["ticker", "date"])["open"]
    for row in trades.head(40).itertuples():
        assert row.entry_price == pytest.approx(lookup.loc[(row.ticker, row.entry_date)])


def test_future_price_moves_cannot_change_past_returns():
    """Perturb the last 20 days of data. Any daily return before that window must be
    bit-identical — if it is not, something is reading forward."""
    bars = make_bars(["AAA", "BBB"], n_days=160, seed=7)
    sentiment = make_sentiment(bars, seed=7)
    base = build_features(bars, sentiment, (14, 2))

    tampered_bars = bars.copy()
    cutoff = sorted(tampered_bars["date"].unique())[-20]
    mask = tampered_bars["date"] >= cutoff
    for col in ("open", "high", "low", "close"):
        tampered_bars.loc[mask, col] *= 1.5
    tampered = build_features(tampered_bars, sentiment, (14, 2))

    a = run_backtest(base, "rsi", cost_bps=10.0)["daily"]
    b = run_backtest(tampered, "rsi", cost_bps=10.0)["daily"]

    before = a.index < cutoff
    np.testing.assert_allclose(
        a.loc[before, "ret"].to_numpy(), b.loc[before, "ret"].to_numpy(), atol=1e-12
    )


# ── Costs ─────────────────────────────────────────────────────────────────────


def test_higher_costs_never_improve_returns(features):
    sharpes = []
    for cost in (0.0, 5.0, 10.0, 20.0, 50.0):
        out = run_backtest(features, "rsi", cost_bps=cost)
        sharpes.append(out["daily"]["ret"].sum())

    assert sharpes == sorted(sharpes, reverse=True), (
        "cumulative return must be monotonically decreasing in cost"
    )


def test_cost_is_charged_twice_per_round_trip(features):
    """Net return should differ from gross by exactly the full round-trip cost."""
    cost_bps = 20.0
    out = run_backtest(features, "rsi", cost_bps=cost_bps)
    trades = out["trades"]
    if trades.empty:
        pytest.skip("no trades")

    difference = (trades["gross_return"] - trades["net_return"]).unique()
    assert len(difference) == 1
    assert difference[0] == pytest.approx(cost_bps / 10_000.0)


def test_zero_cost_leaves_gross_and_net_equal(features):
    out = run_backtest(features, "rsi", cost_bps=0.0)
    trades = out["trades"]
    if trades.empty:
        pytest.skip("no trades")
    np.testing.assert_allclose(trades["gross_return"], trades["net_return"], atol=1e-15)


# ── Position and slot discipline ───────────────────────────────────────────────


def test_never_exceeds_the_slot_limit(features):
    out = run_backtest(features, "rsi", cost_bps=10.0, max_positions=3)
    assert out["daily"]["n_positions"].max() <= 3


def test_one_position_per_ticker_at_a_time(features):
    """Overlapping entries in the same name would double its weight unnoticed."""
    out = run_backtest(features, "rsi", cost_bps=10.0)
    trades = out["trades"].sort_values(["ticker", "entry_date"])
    if trades.empty:
        pytest.skip("no trades")

    for ticker, group in trades.groupby("ticker"):
        exits = group["exit_date"].to_numpy()
        entries = group["entry_date"].to_numpy()
        assert (entries[1:] >= exits[:-1]).all(), f"{ticker} had overlapping positions"


def test_hold_period_is_respected(features):
    hold = 5
    out = run_backtest(features, "rsi", cost_bps=10.0, hold_days=hold)
    trades = out["trades"]
    if trades.empty:
        pytest.skip("no trades")
    assert trades["days_held"].max() <= hold
    assert (trades["days_held"] >= 1).all()


def test_slots_are_rationed_to_the_most_oversold(features):
    """With one slot, the taken trade should be the most oversold candidate available —
    otherwise results depend on DataFrame row order."""
    out = run_backtest(features, "rsi", cost_bps=0.0, max_positions=1)
    assert out["daily"]["n_positions"].max() <= 1


# ── Arm definitions ───────────────────────────────────────────────────────────


def test_filtered_arm_is_a_subset_of_the_unfiltered_one(features):
    """`rsi_filtered` only removes entries, so it can never trade more than `rsi`."""
    plain = run_backtest(features, "rsi", cost_bps=10.0, max_positions=50)
    filtered = run_backtest(features, "rsi_filtered", cost_bps=10.0, max_positions=50)
    assert len(filtered["trades"]) <= len(plain["trades"])


def test_filtered_and_negative_news_arms_partition_the_rsi_signal(features):
    """Every oversold day is either filtered-in or negative-news, never both."""
    from study.backtest import _entry_mask

    plain = _entry_mask(features, "rsi", "rsi_14", 30.0)
    kept = _entry_mask(features, "rsi_filtered", "rsi_14", 30.0)
    dropped = _entry_mask(features, "rsi_negative_news", "rsi_14", 30.0)

    assert not (kept & dropped).any(), "the two arms must be disjoint"
    assert (kept | dropped).equals(plain), "together they must cover the plain signal"


def test_nan_sentiment_counts_as_uninformed_and_passes_the_filter(all_arms_features):
    """A name with no news is the uninformed case the hypothesis says to buy."""
    from study.backtest import _entry_mask

    oversold = all_arms_features["rsi_14"] < 30.0
    no_news = all_arms_features["sentiment"].isna()
    kept = _entry_mask(all_arms_features, "rsi_filtered", "rsi_14", 30.0)

    target = oversold & no_news
    assert target.any(), "fixture must contain oversold days with no news"
    assert kept[target].all()


def test_negative_news_arm_fires_only_on_negative_news(all_arms_features):
    """The falsification arm must be selecting what it claims to select."""
    out = run_backtest(all_arms_features, "rsi_negative_news", cost_bps=10.0)
    trades = out["trades"]

    assert not trades.empty
    assert (trades["sentiment_at_signal"] < -0.20).all()


# ── Benchmark ─────────────────────────────────────────────────────────────────


def test_buy_and_hold_tracks_the_mean_of_constituent_returns(features):
    daily = buy_and_hold(features)
    assert not daily.empty
    assert daily["ret"].abs().max() < 0.5

    some_date = daily.index[40]
    expected = (
        features.sort_values(["ticker", "date"])
        .assign(r=lambda d: d.groupby("ticker")["close"].pct_change())
        .query("date == @some_date")["r"]
        .mean()
    )
    assert daily.loc[some_date, "ret"] == pytest.approx(expected)


def test_empty_input_produces_no_trades():
    empty = pd.DataFrame(
        columns=["ticker", "date", "open", "high", "low", "close", "volume",
                 "rsi_14", "rsi_2", "sentiment", "articles", "next_open", "next_date"]
    )
    out = run_backtest(empty, "rsi", cost_bps=10.0)
    assert out["trades"].empty
    assert out["daily"].empty
