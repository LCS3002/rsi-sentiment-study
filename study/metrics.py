"""Performance statistics.

The Newey-West standard error matters more here than it might look. Holds overlap — a
5-day position means today's return shares four days with yesterday's — so the daily
return series is serially correlated and a plain OLS standard error is too small. Using
one would inflate every t-stat in the study, in the direction that flatters the result.

Written out rather than imported from statsmodels: it is a dozen lines for the
intercept-only case, keeps the dependency list to numpy/pandas, and makes the
autocovariance weighting inspectable. `tests/test_metrics.py` cross-checks it against
statsmodels' HAC implementation when that package happens to be installed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from study.config import NEWEY_WEST_LAGS, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)


def newey_west_tstat(returns: pd.Series | np.ndarray, lags: int = NEWEY_WEST_LAGS) -> float:
    """t-statistic for the mean of `returns`, with a Bartlett-kernel HAC variance.

    This is the intercept-only case: regressing the series on a constant, the HAC variance
    of the mean is (1/n)[γ0 + 2 Σ_{j=1..L} (1 - j/(L+1)) γ_j], with γ_j the sample
    autocovariance at lag j.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return float("nan")

    demeaned = values - values.mean()
    gamma0 = float(demeaned @ demeaned) / n

    variance = gamma0
    usable_lags = min(lags, n - 1)
    for lag in range(1, usable_lags + 1):
        weight = 1.0 - lag / (usable_lags + 1.0)
        gamma = float(demeaned[lag:] @ demeaned[:-lag]) / n
        variance += 2.0 * weight * gamma

    if variance <= 0:
        # a Bartlett-weighted sum is not guaranteed positive on short, noisy samples
        logger.warning("Non-positive HAC variance (n=%d) — reporting NaN", n)
        return float("nan")

    standard_error = np.sqrt(variance / n)
    return float(values.mean() / standard_error)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall in the equity curve, as a negative fraction."""
    if equity.empty:
        return float("nan")
    running_peak = equity.cummax()
    return float((equity / running_peak - 1.0).min())


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    return (1.0 + daily_returns.fillna(0.0)).cumprod()


def summarise(
    daily: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    label: str = "",
) -> dict:
    """Headline statistics for one arm at one cost level."""
    returns = daily["ret"].astype(float).fillna(0.0)
    n_days = len(returns)

    if n_days == 0:
        return {"label": label, "days": 0}

    equity = equity_curve(returns)
    years = n_days / TRADING_DAYS_PER_YEAR

    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    excess = returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    sharpe = (
        float(excess.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if returns.std(ddof=1) > 0
        else float("nan")
    )

    stats = {
        "label": label,
        "days": n_days,
        "years": round(years, 2),
        "total_return": total,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "nw_tstat": newey_west_tstat(returns),
        "mean_daily_bps": float(returns.mean() * 10_000),
        "mean_positions": float(daily["n_positions"].mean()),
        "exposure": float((daily["n_positions"] > 0).mean()),
    }

    if trades is not None and not trades.empty:
        net = trades["net_return"].astype(float)
        wins, losses = net[net > 0], net[net <= 0]
        stats.update(
            {
                "trades": int(len(trades)),
                "trades_per_year": float(len(trades) / years) if years > 0 else float("nan"),
                "hit_rate": float((net > 0).mean()),
                "avg_win": float(wins.mean()) if len(wins) else float("nan"),
                "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
                "avg_trade": float(net.mean()),
                "avg_days_held": float(trades["days_held"].mean()),
                "worst_trade": float(net.min()),
                "best_trade": float(net.max()),
            }
        )
        # Profit factor is undefined when nothing lost; report NaN rather than infinity
        gross_loss = float(-losses.sum())
        stats["profit_factor"] = (
            float(wins.sum() / gross_loss) if gross_loss > 0 else float("nan")
        )
    else:
        stats["trades"] = 0

    return stats


def subperiod_table(
    daily: pd.DataFrame, subperiods: tuple[tuple[str, str, str], ...]
) -> pd.DataFrame:
    """Sharpe and return per subperiod — evidence the result is not one regime."""
    rows = []
    for name, start, end in subperiods:
        window = daily.loc[str(start) : str(end)]
        if window.empty:
            rows.append({"period": name, "days": 0})
            continue
        stats = summarise(window, label=name)
        rows.append(
            {
                "period": name,
                "days": stats["days"],
                "cagr": stats["cagr"],
                "sharpe": stats["sharpe"],
                "max_drawdown": stats["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def sentiment_bucket_table(trades: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Mean trade outcome by sentiment at signal — the mechanism evidence.

    If conditioning on sentiment does anything, it should show up here as a monotone-ish
    relationship between how negative the news was and how the trade turned out. If it is
    flat, the filter is not working through the channel the hypothesis claims.
    """
    if trades.empty or "sentiment_at_signal" not in trades.columns:
        return pd.DataFrame()

    dated = trades.dropna(subset=["sentiment_at_signal"]).copy()
    if dated.empty:
        return pd.DataFrame()

    try:
        dated["bucket"] = pd.qcut(dated["sentiment_at_signal"], bins, duplicates="drop")
    except ValueError:
        logger.warning("Too few distinct sentiment values to bucket")
        return pd.DataFrame()

    out = (
        dated.groupby("bucket", observed=True)
        .agg(
            trades=("net_return", "size"),
            mean_return=("net_return", "mean"),
            hit_rate=("net_return", lambda s: float((s > 0).mean())),
            mean_sentiment=("sentiment_at_signal", "mean"),
        )
        .reset_index()
    )
    out["bucket"] = out["bucket"].astype(str)
    return out
