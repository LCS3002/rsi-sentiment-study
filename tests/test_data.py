"""Loader shape handling.

`_flatten_yf` absorbs yfinance's inconsistencies, and those inconsistencies are
environment-dependent — they show up only when a ticker fails to download, which is
exactly when you are least likely to be watching. These cases crashed a full run once.
"""

import numpy as np
import pandas as pd
import pytest

from study.data import BAR_COLUMNS, _flatten_yf, usable_tickers


def _wide(tickers, fields, n=5):
    dates = pd.bdate_range("2022-01-03", periods=n, name="Date")
    columns = pd.MultiIndex.from_product([fields, tickers])
    values = np.arange(1, n * len(columns) + 1, dtype=float).reshape(n, len(columns))
    return pd.DataFrame(values, index=dates, columns=columns)


STANDARD = ["Close", "High", "Low", "Open", "Volume"]


def test_multi_ticker_frame_flattens_to_long_form():
    out = _flatten_yf(_wide(["AAA", "BBB"], STANDARD), ["AAA", "BBB"])

    assert list(out.columns) == ["ticker", "date"] + BAR_COLUMNS
    assert set(out["ticker"]) == {"AAA", "BBB"}
    assert len(out) == 10


def test_adj_close_alongside_close_is_dropped_not_renamed():
    """Regression: yfinance emits an `Adj Close` level alongside `Close` when one of the
    requested tickers fails, even under auto_adjust=True. Renaming it produced two
    columns called `close`, and every later access raised "cannot reindex on an axis with
    duplicate labels" — which killed a full run partway through."""
    raw = _wide(["AAA", "BBB"], ["Adj Close"] + STANDARD)
    out = _flatten_yf(raw, ["AAA", "BBB"])

    assert list(out.columns).count("close") == 1
    assert not out.columns.duplicated().any()
    # `Close` is already adjusted under auto_adjust=True, so it is the one that survives
    expected = raw[("Close", "AAA")].iloc[0]
    assert out[out["ticker"] == "AAA"]["close"].iloc[0] == pytest.approx(expected)


def test_adj_close_alone_is_renamed_to_close():
    raw = _wide(["AAA"], ["Adj Close", "High", "Low", "Open", "Volume"])
    out = _flatten_yf(raw, ["AAA"])
    assert "close" in out.columns


def test_single_ticker_without_a_ticker_level_is_labelled():
    dates = pd.bdate_range("2022-01-03", periods=4, name="Date")
    raw = pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100},
        index=dates,
    )
    out = _flatten_yf(raw, ["AAA"])

    assert set(out["ticker"]) == {"AAA"}
    assert len(out) == 4


def test_dates_come_back_naive_and_normalised():
    """News dates are date-only, so a tz-aware bar index would not join against them."""
    dates = pd.date_range("2022-01-03 14:30", periods=3, freq="D", tz="UTC", name="Date")
    raw = pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100}, index=dates
    )
    out = _flatten_yf(raw, ["AAA"])

    assert out["date"].dt.tz is None
    assert (out["date"] == out["date"].dt.normalize()).all()


def test_empty_frame_returns_the_right_shape():
    out = _flatten_yf(pd.DataFrame(), ["AAA"])
    assert out.empty
    assert list(out.columns) == ["ticker", "date"] + BAR_COLUMNS


def test_missing_price_columns_raise_rather_than_pass_silently():
    raw = _wide(["AAA"], ["Close", "Volume"])
    with pytest.raises(RuntimeError, match="no \\['open'"):
        _flatten_yf(raw, ["AAA"])


def test_a_failed_fetch_never_deletes_already_cached_tickers(tmp_path, monkeypatch):
    """Regression: the cache used to be overwritten with whatever the latest download
    returned. Yahoo intermittently reports a listed name as delisted — MSFT came back
    empty on one run — so a transient failure silently erased it from an 86-name
    universe, and nothing downstream noticed."""
    import study.data as data

    cache = tmp_path / "bars.parquet"
    monkeypatch.setattr(data, "BARS_CACHE", cache)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)

    good = _flatten_yf(_wide(["AAA", "BBB"], STANDARD, n=10), ["AAA", "BBB"])
    good.to_parquet(cache, index=False)

    # the next fetch asks for a third ticker and gets nothing back at all
    monkeypatch.setattr(
        data, "_download",
        lambda tickers, start, end: pd.DataFrame(
            columns=["ticker", "date"] + BAR_COLUMNS
        ),
    )
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    out = data.load_bars(["AAA", "BBB", "CCC"], "2022-01-03", "2022-01-14", attempts=2)

    assert set(out["ticker"]) == {"AAA", "BBB"}, "CCC is absent, but AAA/BBB must survive"
    assert set(pd.read_parquet(cache)["ticker"]) == {"AAA", "BBB"}


def test_a_missing_ticker_is_retried_individually(tmp_path, monkeypatch):
    """A batch request that partially fails usually succeeds when the straggler is asked
    for on its own, so give up only after retrying."""
    import study.data as data

    monkeypatch.setattr(data, "BARS_CACHE", tmp_path / "bars.parquet")
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    calls = []

    def flaky(tickers, start, end):
        calls.append(list(tickers))
        if len(tickers) > 1:                 # the batch call drops BBB
            return _flatten_yf(_wide(["AAA"], STANDARD, n=5), ["AAA"])
        return _flatten_yf(_wide(list(tickers), STANDARD, n=5), list(tickers))

    monkeypatch.setattr(data, "_download", flaky)
    out = data.load_bars(["AAA", "BBB"], "2022-01-03", "2022-01-10", attempts=3)

    assert set(out["ticker"]) == {"AAA", "BBB"}, "BBB must be recovered on retry"
    assert ["BBB"] in calls, "the straggler should be retried on its own"


def test_cache_satisfies_the_request_without_refetching(tmp_path, monkeypatch):
    """A cache holding everything asked for must not trigger a download — otherwise
    permanently-unavailable tickers make every run refetch forever."""
    import study.data as data

    cache = tmp_path / "bars.parquet"
    monkeypatch.setattr(data, "BARS_CACHE", cache)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    _flatten_yf(_wide(["AAA"], STANDARD, n=8), ["AAA"]).to_parquet(cache, index=False)

    def explode(*_args, **_kwargs):
        raise AssertionError("should not download when the cache already has it")

    monkeypatch.setattr(data, "_download", explode)
    assert len(data.load_bars(["AAA"], "2022-01-03", "2022-01-14")) == 8


def test_usable_tickers_applies_both_thresholds():
    """A name needs enough articles *and* enough span; either alone is not enough."""
    rows = []
    # plenty of articles, but all inside one month
    rows += [("SHORT", pd.Timestamp("2015-01-05") + pd.Timedelta(days=i % 25))
             for i in range(400)]
    # long span, but only a handful of articles
    rows += [("SPARSE", pd.Timestamp("2010-01-05") + pd.Timedelta(days=i * 400))
             for i in range(10)]
    # clears both
    rows += [("GOOD", pd.Timestamp("2010-01-05") + pd.Timedelta(days=i * 4))
             for i in range(400)]

    news = pd.DataFrame(rows, columns=["ticker", "date"])
    assert usable_tickers(news) == ["GOOD"]
