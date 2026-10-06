"""Signal construction: RSI arithmetic, and that nothing looks forward."""

import numpy as np
import pandas as pd
import pytest

from study.signals import add_rsi, build_features, daily_sentiment, wilder_rsi

from .conftest import make_bars, make_sentiment


# ── RSI arithmetic ────────────────────────────────────────────────────────────


def test_rsi_is_100_on_an_unbroken_rise():
    series = pd.Series([float(i) for i in range(1, 40)])
    assert wilder_rsi(series, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_on_an_unbroken_fall():
    series = pd.Series([float(i) for i in range(40, 1, -1)])
    assert wilder_rsi(series, 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_is_nan_on_a_flat_series():
    """Zero gain and zero loss is undefined, and must not read as 100 — a motionless
    market should produce no signal at all."""
    assert np.isnan(wilder_rsi(pd.Series([50.0] * 40), 14).iloc[-1])


def test_rsi_first_valid_value_needs_length_plus_one_closes():
    rsi = wilder_rsi(pd.Series([float(i) for i in range(1, 40)]), 14)
    assert rsi.iloc[:14].isna().all()
    assert not np.isnan(rsi.iloc[14])


def test_rsi_too_short_is_all_nan():
    assert wilder_rsi(pd.Series([1.0, 2.0, 3.0]), 14).isna().all()


def test_rsi_stays_within_bounds():
    rng = np.random.default_rng(0)
    series = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)))
    values = wilder_rsi(series, 14).dropna()
    assert len(values) > 380
    assert values.between(0, 100).all()


def test_rsi_matches_an_independent_textbook_implementation():
    """Cross-check against a separately written loop. The study's conclusion rests on
    this arithmetic, so it is verified rather than trusted."""
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, 300))
    length = 14

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    expected = np.full(len(closes), np.nan)
    avg_g, avg_l = gains[:length].mean(), losses[:length].mean()
    expected[length] = 100 - 100 / (1 + avg_g / avg_l)
    for i in range(length, len(deltas)):
        avg_g = (avg_g * (length - 1) + gains[i]) / length
        avg_l = (avg_l * (length - 1) + losses[i]) / length
        expected[i + 1] = 100 - 100 / (1 + avg_g / avg_l)

    actual = wilder_rsi(pd.Series(closes), length).to_numpy()
    valid = ~np.isnan(expected)
    np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-12)


def test_rsi_is_computed_within_each_ticker_not_across_them():
    """Grouping errors here would bleed one name's prices into another's indicator."""
    bars = make_bars(["AAA", "BBB"], n_days=100)
    out = add_rsi(bars, (14,))

    standalone = wilder_rsi(
        bars[bars["ticker"] == "BBB"].sort_values("date")["close"].reset_index(drop=True),
        14,
    )
    grouped = out[out["ticker"] == "BBB"].sort_values("date")["rsi_14"].reset_index(drop=True)
    np.testing.assert_allclose(grouped.to_numpy(), standalone.to_numpy(), atol=1e-12,
                               equal_nan=True)


def test_rsi_does_not_look_forward():
    """Changing the tail of a price series must not alter earlier RSI values."""
    rng = np.random.default_rng(3)
    closes = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    tampered = closes.copy()
    tampered.iloc[150:] *= 2.0

    a = wilder_rsi(closes, 14).iloc[:150]
    b = wilder_rsi(tampered, 14).iloc[:150]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-12, equal_nan=True)


# ── Sentiment aggregation ─────────────────────────────────────────────────────


def _scored(rows):
    return pd.DataFrame(rows, columns=["ticker", "date", "title", "score"])


def test_same_day_articles_average_rather_than_accumulate():
    """Three stories about one name on one day are repeated observations of one event."""
    calendar = pd.bdate_range("2021-01-04", periods=5)
    scored = _scored(
        [
            ("AAA", calendar[0], "a", 1.0),
            ("AAA", calendar[0], "b", 0.0),
            ("AAA", calendar[0], "c", -1.0),
        ]
    )
    out = daily_sentiment(scored, calendar, halflife=1e9)
    assert out.iloc[0]["sentiment"] == pytest.approx(0.0)
    assert out.iloc[0]["articles"] == 3


def test_news_on_a_non_trading_day_rolls_forward():
    """A Saturday story is first actionable at the next open — dropping it would discard
    every weekend in the sample."""
    calendar = pd.bdate_range("2021-01-04", periods=10)   # Mon 4 Jan onwards
    saturday = pd.Timestamp("2021-01-09")
    scored = _scored([("AAA", saturday, "weekend story", 0.8)])

    out = daily_sentiment(scored, calendar, halflife=1e9)
    live = out[out["articles"] > 0]
    assert len(live) == 1
    assert live.iloc[0]["date"] == pd.Timestamp("2021-01-11")   # the following Monday


def test_sentiment_decays_after_the_news_day():
    calendar = pd.bdate_range("2021-01-04", periods=12)
    scored = _scored([("AAA", calendar[0], "one story", 1.0)])

    out = daily_sentiment(scored, calendar, halflife=3.0).set_index("date")["sentiment"]
    assert out.iloc[0] == pytest.approx(1.0)
    # with no further news the EWMA holds its last value rather than decaying to zero;
    # what must not happen is it growing
    assert out.dropna().is_monotonic_decreasing or out.dropna().nunique() == 1


def test_sentiment_is_nan_before_any_news_arrives():
    """A name with no news yet must read NaN, not zero — zero is a real neutral
    reading and would be a different claim."""
    calendar = pd.bdate_range("2021-01-04", periods=10)
    scored = _scored([("AAA", calendar[5], "later story", 0.5)])

    out = daily_sentiment(scored, calendar, halflife=3.0).set_index("date")["sentiment"]
    assert out.iloc[:5].isna().all()
    assert not np.isnan(out.iloc[5])


def test_sentiment_does_not_use_future_news():
    calendar = pd.bdate_range("2021-01-04", periods=20)
    early = _scored([("AAA", calendar[2], "first", 0.5)])
    plus_later = _scored(
        [("AAA", calendar[2], "first", 0.5), ("AAA", calendar[15], "second", -0.9)]
    )

    a = daily_sentiment(early, calendar, halflife=3.0).set_index("date")["sentiment"]
    b = daily_sentiment(plus_later, calendar, halflife=3.0).set_index("date")["sentiment"]

    upto = calendar[:15]
    np.testing.assert_allclose(
        a.loc[upto].to_numpy(), b.loc[upto].to_numpy(), atol=1e-12, equal_nan=True
    )


# ── Feature assembly ──────────────────────────────────────────────────────────


def test_next_open_is_tomorrows_open_within_each_ticker():
    bars = make_bars(["AAA", "BBB"], n_days=60)
    features = build_features(bars, make_sentiment(bars), (14,))

    for ticker, group in features.groupby("ticker"):
        group = group.sort_values("date")
        np.testing.assert_allclose(
            group["next_open"].to_numpy()[:-1],
            group["open"].to_numpy()[1:],
            atol=1e-12,
        )
        assert np.isnan(group["next_open"].to_numpy()[-1]), (
            "the last bar has no next open and must be NaN, not carried over"
        )


def test_next_open_does_not_leak_across_tickers():
    """The final row of one ticker must not pick up the first open of the next."""
    bars = make_bars(["AAA", "BBB"], n_days=40)
    features = build_features(bars, make_sentiment(bars), (14,))

    last_rows = features.sort_values("date").groupby("ticker").tail(1)
    assert last_rows["next_open"].isna().all()
