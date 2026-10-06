"""Event-driven backtest with next-open execution and explicit costs.

Three commitments, each of which is the difference between a study and a curve-fit:

**Next-open execution.** A signal computed from the close of day D is executed at the
**open of D+1**, never the close of D. The one-bar delay is carried in the data itself
(`next_open`, from `signals.build_features`) rather than applied inside this loop, so it
is visible rather than trusted. `tests/test_backtest.py` asserts no position can ever open
at or before its own signal date.

**Fixed capital slots.** Capital is divided into `MAX_CONCURRENT_POSITIONS` slots; a trade
takes one slot and unused slots earn the risk-free rate (zero, and stated as such). The
tempting alternative — equal-weighting across whatever positions happen to be open —
silently levers the strategy up whenever few signals fire, and makes the Sharpe
incomparable with buy-and-hold. When more signals fire than there are free slots, the
most oversold candidates are taken first, deterministically.

**Long only.** The hypothesis is about buying dips, which is a long-side question. It also
avoids claiming short results that would need borrow cost and availability modelled to
mean anything.

Daily returns are decomposed honestly: on the entry day a position earns
`close / entry_price - 1`, thereafter `close / prev_close - 1`, and on the exit day
`exit_price / prev_close - 1`. Half the round-trip cost is charged at entry and half at
exit, against that position's notional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from study.config import (
    EXIT_ON_RSI_REVERSION,
    HOLD_DAYS,
    MAX_CONCURRENT_POSITIONS,
    RSI_EXIT_LEVEL,
    RSI_FAST_OVERSOLD,
    RSI_OVERSOLD,
    SENTIMENT_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ── Signal definitions ────────────────────────────────────────────────────────
#
# A NaN sentiment means "no news for this name around this date". For the filtered arm
# that counts as *uninformed*, which the hypothesis says is the case worth buying — so it
# passes the filter. The confirmation arm, which requires actual negative news, excludes
# it. Both choices follow from the hypothesis rather than from what scored better.


def _entry_mask(features: pd.DataFrame, arm: str, rsi_col: str, oversold: float) -> pd.Series:
    rsi = features[rsi_col]
    sent = features["sentiment"]

    if arm == "rsi":
        return rsi < oversold

    if arm == "sentiment":
        return sent > SENTIMENT_THRESHOLD

    if arm == "rsi_filtered":
        # The hypothesis: buy the oversold name unless the decline looks news-driven.
        return (rsi < oversold) & ~(sent < -SENTIMENT_THRESHOLD)

    if arm == "rsi_negative_news":
        # The falsification arm. If the mechanism is real, these are the trades the
        # filtered arm avoided, and they should be materially worse.
        return (rsi < oversold) & (sent < -SENTIMENT_THRESHOLD)

    raise ValueError(f"unknown arm {arm!r}")


ARMS = ("rsi", "sentiment", "rsi_filtered", "rsi_negative_news")


# ── Trade bookkeeping ─────────────────────────────────────────────────────────


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    signal_date: pd.Timestamp
    rsi_at_signal: float
    sentiment_at_signal: float
    days_held: int = 0
    prev_close: float = field(default=np.nan)


@dataclass
class Trade:
    ticker: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    days_held: int
    gross_return: float
    net_return: float
    rsi_at_signal: float
    sentiment_at_signal: float
    exit_reason: str


# ── Engine ────────────────────────────────────────────────────────────────────


def run_backtest(
    features: pd.DataFrame,
    arm: str,
    cost_bps: float,
    rsi_col: str = "rsi_14",
    oversold: float | None = None,
    hold_days: int = HOLD_DAYS,
    max_positions: int = MAX_CONCURRENT_POSITIONS,
) -> dict:
    """Run one arm at one cost level.

    Returns {"daily": DataFrame[date, ret, n_positions], "trades": DataFrame}.
    """
    if oversold is None:
        oversold = RSI_FAST_OVERSOLD if rsi_col == "rsi_2" else RSI_OVERSOLD

    features = features.sort_values(["date", "ticker"]).copy()
    features["entry_signal"] = _entry_mask(features, arm, rsi_col, oversold)

    one_way_cost = cost_bps / 2.0 / 10_000.0
    slot_weight = 1.0 / max_positions

    calendar = pd.DatetimeIndex(sorted(features["date"].unique()))
    by_date = {date: group for date, group in features.groupby("date")}

    open_positions: dict[str, Position] = {}
    pending: list[tuple[str, float, float, pd.Timestamp]] = []
    trades: list[Trade] = []
    daily_rows = []

    for date in calendar:
        today = by_date.get(date)
        if today is None:
            continue
        rows = today.set_index("ticker")

        day_return = 0.0

        # ── 1. exits, at today's open ─────────────────────────────────────────
        for ticker in list(open_positions):
            position = open_positions[ticker]
            if ticker not in rows.index:
                continue

            row = rows.loc[ticker]
            position.days_held += 1

            rsi_now = row.get(rsi_col, np.nan)
            reverted = (
                EXIT_ON_RSI_REVERSION
                and not pd.isna(rsi_now)
                and rsi_now > RSI_EXIT_LEVEL
            )
            matured = position.days_held >= hold_days

            if not (reverted or matured):
                continue

            exit_price = float(row["open"])
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue

            # the leg from the previous close into today's open
            leg = exit_price / position.prev_close - 1.0
            day_return += slot_weight * (leg - one_way_cost)

            gross = exit_price / position.entry_price - 1.0
            trades.append(
                Trade(
                    ticker=ticker,
                    signal_date=position.signal_date,
                    entry_date=position.entry_date,
                    exit_date=date,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    days_held=position.days_held,
                    gross_return=gross,
                    net_return=gross - 2 * one_way_cost,
                    rsi_at_signal=position.rsi_at_signal,
                    sentiment_at_signal=position.sentiment_at_signal,
                    exit_reason="rsi_reversion" if reverted else "max_hold",
                )
            )
            del open_positions[ticker]

        # ── 2. mark remaining positions to today's close ──────────────────────
        for ticker, position in open_positions.items():
            if ticker not in rows.index:
                continue
            close = float(rows.loc[ticker, "close"])
            if not np.isfinite(close):
                continue
            day_return += slot_weight * (close / position.prev_close - 1.0)
            position.prev_close = close

        # ── 3. entries from yesterday's signals, at today's open ──────────────
        free_slots = max_positions - len(open_positions)
        if pending and free_slots > 0:
            # most oversold first, so slot rationing is deterministic rather than
            # dependent on row order
            pending.sort(key=lambda item: item[1])
            for ticker, rsi_at_signal, sentiment_at_signal, signal_date in pending:
                if free_slots == 0:
                    break
                if ticker in open_positions or ticker not in rows.index:
                    continue

                row = rows.loc[ticker]
                entry_price = float(row["open"])
                close = float(row["close"])
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(close):
                    continue

                # entry-day leg runs from the fill to today's close
                day_return += slot_weight * (close / entry_price - 1.0 - one_way_cost)

                open_positions[ticker] = Position(
                    ticker=ticker,
                    entry_date=date,
                    entry_price=entry_price,
                    signal_date=signal_date,
                    rsi_at_signal=rsi_at_signal,
                    sentiment_at_signal=sentiment_at_signal,
                    prev_close=close,
                )
                free_slots -= 1

        # ── 4. collect today's signals for execution tomorrow ─────────────────
        signalled = today[today["entry_signal"] & today["next_open"].notna()]
        pending = [
            (
                str(r.ticker),
                float(getattr(r, rsi_col)) if not pd.isna(getattr(r, rsi_col)) else np.inf,
                float(r.sentiment) if not pd.isna(r.sentiment) else np.nan,
                date,
            )
            for r in signalled.itertuples()
        ]

        daily_rows.append(
            {"date": date, "ret": day_return, "n_positions": len(open_positions)}
        )

    if daily_rows:
        daily = pd.DataFrame(daily_rows).set_index("date")
    else:
        # No tradeable days at all. Return the right shape rather than crashing on a
        # missing index column — a caller filtering down to an empty window is a
        # legitimate case, not a programming error.
        daily = pd.DataFrame(
            {"ret": pd.Series(dtype=float), "n_positions": pd.Series(dtype=int)},
            index=pd.DatetimeIndex([], name="date"),
        )

    trades_df = pd.DataFrame(
        [t.__dict__ for t in trades],
        columns=[f.name for f in Trade.__dataclass_fields__.values()],
    )

    logger.info(
        "%-18s cost=%4.1fbps  trades=%5d  mean_positions=%4.1f",
        arm, cost_bps, len(trades_df), daily["n_positions"].mean(),
    )
    return {"daily": daily, "trades": trades_df}


def buy_and_hold(features: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight the universe, rebalanced daily, as the benchmark.

    Costs are not charged: this is the do-nothing alternative the strategy has to beat,
    and charging it a daily rebalance it does not need would flatter the strategy.
    """
    features = features.sort_values(["ticker", "date"]).copy()
    features["ret"] = features.groupby("ticker")["close"].pct_change()

    daily = (
        features.dropna(subset=["ret"])
        .groupby("date")
        .agg(ret=("ret", "mean"), n_positions=("ticker", "count"))
    )
    return daily
