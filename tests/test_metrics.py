"""Performance statistics, with the Newey-West t-stat checked hardest.

Overlapping holds make the daily return series serially correlated, so a plain OLS
standard error would be too small and every t-stat in the study too large — in the
direction that flatters the result. That makes this the statistic most worth verifying.
"""

import numpy as np
import pandas as pd
import pytest

from study.metrics import (
    equity_curve,
    max_drawdown,
    newey_west_tstat,
    sentiment_bucket_table,
    subperiod_table,
    summarise,
)


def _daily(returns, start="2021-01-04"):
    index = pd.bdate_range(start=start, periods=len(returns), name="date")
    return pd.DataFrame({"ret": returns, "n_positions": 5}, index=index)


# ── Newey-West ────────────────────────────────────────────────────────────────


def test_tstat_of_pure_noise_is_small():
    rng = np.random.default_rng(0)
    t = newey_west_tstat(rng.normal(0, 0.01, 2000), lags=10)
    assert abs(t) < 3.0


def test_tstat_of_a_strong_constant_drift_is_large():
    rng = np.random.default_rng(1)
    series = rng.normal(0.001, 0.005, 2000)
    assert newey_west_tstat(series, lags=10) > 5.0


def test_zero_lags_reduces_to_the_plain_ols_tstat():
    """With no lag terms the HAC variance is just the sample variance, so this must match
    mean / (sd/sqrt(n)) — a useful anchor on the implementation."""
    rng = np.random.default_rng(2)
    values = rng.normal(0.0005, 0.01, 500)

    expected = values.mean() / (values.std(ddof=0) / np.sqrt(len(values)))
    assert newey_west_tstat(values, lags=0) == pytest.approx(expected, rel=1e-10)


def test_positive_autocorrelation_widens_the_error_and_shrinks_the_tstat():
    """The whole reason for using HAC: overlapping holds inflate a naive t-stat."""
    rng = np.random.default_rng(3)
    innovations = rng.normal(0.0004, 0.01, 3000)
    # AR(1) with positive persistence, as overlapping 5-day holds produce
    series = np.zeros_like(innovations)
    for i in range(1, len(series)):
        series[i] = 0.6 * series[i - 1] + innovations[i]

    naive = newey_west_tstat(series, lags=0)
    hac = newey_west_tstat(series, lags=10)
    assert abs(hac) < abs(naive), "HAC must be the more conservative of the two"


def test_tstat_is_nan_on_degenerate_input():
    assert np.isnan(newey_west_tstat([]))
    assert np.isnan(newey_west_tstat([0.01]))


def test_tstat_ignores_nans():
    rng = np.random.default_rng(4)
    clean = rng.normal(0.001, 0.01, 500)
    dirty = np.concatenate([clean, [np.nan] * 20])
    assert newey_west_tstat(dirty, lags=5) == pytest.approx(
        newey_west_tstat(clean, lags=5), rel=1e-12
    )


def test_matches_statsmodels_hac_when_available():
    """Cross-check against the standard implementation, skipped if it is not installed."""
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(5)
    values = rng.normal(0.0005, 0.01, 800)
    lags = 10

    model = sm.OLS(values, np.ones(len(values))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "kernel": "bartlett", "use_correction": False}
    )
    assert newey_west_tstat(values, lags=lags) == pytest.approx(
        float(model.tvalues[0]), rel=1e-6
    )


# ── Drawdown and equity ───────────────────────────────────────────────────────


def test_max_drawdown_of_a_known_path():
    # 100 -> 120 -> 90: peak 120, trough 90, so -25%
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_of_a_monotone_rise_is_zero():
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)


def test_equity_curve_compounds():
    curve = equity_curve(pd.Series([0.1, 0.1]))
    assert curve.iloc[-1] == pytest.approx(1.21)


def test_equity_curve_treats_missing_days_as_flat():
    assert equity_curve(pd.Series([0.1, np.nan, 0.1])).iloc[-1] == pytest.approx(1.21)


# ── Summary ───────────────────────────────────────────────────────────────────


def test_summarise_recovers_a_known_sharpe():
    """Constant daily returns have zero volatility, so use a clean two-state series
    whose annualised Sharpe can be computed by hand."""
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0004, 0.008, 2520)   # ~10 years
    stats = summarise(_daily(returns))

    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert stats["sharpe"] == pytest.approx(expected, rel=1e-9)
    assert stats["volatility"] == pytest.approx(returns.std(ddof=1) * np.sqrt(252), rel=1e-9)
    assert stats["days"] == len(returns)


def test_summarise_cagr_matches_compounded_growth():
    returns = np.full(252, 0.001)
    stats = summarise(_daily(returns))
    expected = (1.001 ** 252) - 1
    assert stats["cagr"] == pytest.approx(expected, rel=1e-6)


def test_summarise_handles_an_empty_frame():
    stats = summarise(pd.DataFrame({"ret": [], "n_positions": []}))
    assert stats["days"] == 0


def test_summarise_trade_statistics():
    trades = pd.DataFrame(
        {
            "net_return": [0.05, -0.02, 0.03, -0.01],
            "days_held": [5, 3, 5, 2],
            "sentiment_at_signal": [0.5, -0.5, 0.1, -0.3],
        }
    )
    stats = summarise(_daily(np.full(252, 0.0002)), trades)

    assert stats["trades"] == 4
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["avg_win"] == pytest.approx(0.04)
    assert stats["avg_loss"] == pytest.approx(-0.015)
    assert stats["profit_factor"] == pytest.approx(0.08 / 0.03)


def test_profit_factor_is_nan_when_nothing_lost():
    """Infinity in a results table is a reporting bug, not a finding."""
    trades = pd.DataFrame(
        {"net_return": [0.01, 0.02], "days_held": [5, 5], "sentiment_at_signal": [0.1, 0.2]}
    )
    stats = summarise(_daily(np.full(60, 0.0001)), trades)
    assert np.isnan(stats["profit_factor"])


# ── Tables ────────────────────────────────────────────────────────────────────


def test_subperiod_table_splits_the_sample():
    returns = np.full(2520, 0.0003)
    daily = _daily(returns, start="2014-01-01")
    table = subperiod_table(
        daily,
        (("early", "2014-01-01", "2016-12-31"), ("late", "2017-01-01", "2023-12-31")),
    )
    assert list(table["period"]) == ["early", "late"]
    assert (table["days"] > 0).all()


def test_subperiod_table_reports_empty_windows_as_zero_days():
    daily = _daily(np.full(100, 0.001), start="2021-01-04")
    table = subperiod_table(daily, (("absent", "1990-01-01", "1990-12-31"),))
    assert table.iloc[0]["days"] == 0


def test_sentiment_bucket_table_orders_by_sentiment():
    rng = np.random.default_rng(7)
    trades = pd.DataFrame(
        {
            "net_return": rng.normal(0, 0.02, 500),
            "sentiment_at_signal": rng.uniform(-1, 1, 500),
            "days_held": 5,
        }
    )
    table = sentiment_bucket_table(trades, bins=5)
    assert len(table) == 5
    assert table["mean_sentiment"].is_monotonic_increasing
    assert table["trades"].sum() == 500


def test_sentiment_bucket_table_is_empty_without_sentiment():
    assert sentiment_bucket_table(pd.DataFrame()).empty
