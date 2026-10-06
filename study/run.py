"""Study runner. Regenerates every number and figure the note reports.

    python -m study.run --smoke     # 5 tickers, 2 years, under a minute
    python -m study.run             # full universe, 2010-2023

Everything lands in results/ as JSON and CSV. The note cites those files rather than
transcribing numbers by hand, so a stale figure in the write-up is not possible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from study import config
from study.backtest import ARMS, buy_and_hold, run_backtest
from study.data import load_bars, load_news, trading_calendar, usable_tickers
from study.metrics import sentiment_bucket_table, subperiod_table, summarise
from study.signals import build_features, daily_sentiment, score_articles
from study.validation import validate

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("yfinance", "transformers", "datasets", "urllib3", "filelock", "peft"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def build(smoke: bool, force_data: bool, force_sentiment: bool) -> pd.DataFrame:
    """Load everything and return the feature grid the backtest consumes."""
    news = load_news(force=force_data)

    if smoke:
        tickers = list(config.SMOKE_TICKERS)
        start, end = config.SMOKE_START, config.SMOKE_END
    else:
        tickers = usable_tickers(news)
        start = config.SUBPERIODS[0][1]
        end = str(news["date"].max().date())

    news = news[news["ticker"].isin(tickers)]
    news = news[news["date"].between(start, end)]
    logger.info("Universe: %d tickers, %s to %s, %d articles",
                len(tickers), start, end, len(news))

    bars = load_bars(tickers, start, end, force=force_data)
    bars = bars[bars["date"] <= pd.Timestamp(end)]

    scored = score_articles(news, force=force_sentiment)
    calendar = trading_calendar(bars)
    sentiment = daily_sentiment(scored, calendar)

    features = build_features(
        bars, sentiment, (config.RSI_LENGTH_SLOW, config.RSI_LENGTH_FAST)
    )
    # Trim the indicator warmup so the first tradeable day has a formed RSI
    features = features[features["date"] >= pd.Timestamp(start)]

    logger.info("Feature grid: %d ticker-days", len(features))
    return features


def run_all(features: pd.DataFrame, rsi_col: str) -> dict:
    results: dict = {"arms": {}, "benchmark": {}, "validation": {}, "config": {}}

    # Interrogate the sentiment input before using its output. If it predicts nothing,
    # that bounds what any arm built on it can honestly claim.
    results["validation"] = validate(features)

    # ── benchmark ─────────────────────────────────────────────────────────────
    bh_daily = buy_and_hold(features)
    results["benchmark"]["buy_and_hold"] = summarise(bh_daily, label="buy_and_hold")
    results["benchmark"]["subperiods"] = subperiod_table(
        bh_daily, config.SUBPERIODS
    ).to_dict("records")

    # ── strategy arms across the cost grid ────────────────────────────────────
    curves: dict[str, pd.Series] = {}
    for arm in ARMS:
        results["arms"][arm] = {"by_cost": {}}
        for cost in config.COST_GRID_BPS:
            out = run_backtest(features, arm, cost_bps=cost, rsi_col=rsi_col)
            stats = summarise(out["daily"], out["trades"], label=f"{arm}@{cost}bps")
            results["arms"][arm]["by_cost"][str(cost)] = stats

            if cost == config.BASE_COST_BPS:
                results["arms"][arm]["subperiods"] = subperiod_table(
                    out["daily"], config.SUBPERIODS
                ).to_dict("records")
                results["arms"][arm]["sentiment_buckets"] = sentiment_bucket_table(
                    out["trades"]
                ).to_dict("records")
                curves[arm] = (1 + out["daily"]["ret"].fillna(0)).cumprod()
                if not out["trades"].empty:
                    out["trades"].to_csv(
                        config.RESULTS_DIR / f"trades_{arm}.csv", index=False
                    )

    curves["buy_and_hold"] = (1 + bh_daily["ret"].fillna(0)).cumprod()
    pd.DataFrame(curves).to_csv(config.RESULTS_DIR / "equity_curves.csv")

    results["config"] = {
        "rsi_col": rsi_col,
        "hold_days": config.HOLD_DAYS,
        "max_positions": config.MAX_CONCURRENT_POSITIONS,
        "cost_grid_bps": list(config.COST_GRID_BPS),
        "base_cost_bps": config.BASE_COST_BPS,
        "sentiment_threshold": config.SENTIMENT_THRESHOLD,
        "sentiment_halflife_days": config.SENTIMENT_HALFLIFE_DAYS,
        "sentiment_chunk_size": config.SENTIMENT_CHUNK_SIZE,
        "max_text_chars": config.MAX_TEXT_CHARS,
        "rsi_oversold": config.RSI_OVERSOLD,
        "newey_west_lags": config.NEWEY_WEST_LAGS,
        "tickers": int(features["ticker"].nunique()),
        "start": str(features["date"].min().date()),
        "end": str(features["date"].max().date()),
        "ticker_days": int(len(features)),
    }
    return results


def report(results: dict) -> None:
    validation = results.get("validation", {})
    if validation:
        dist = validation.get("distribution", {})
        print(f"\n{'='*86}")
        print("SENTIMENT VALIDATION — does the signal predict anything at all?")
        print(f"{'='*86}")
        if dist.get("scored_ticker_days"):
            print(
                f"coverage {dist['coverage']:.1%} of ticker-days · "
                f"mean {dist['mean']:+.3f} · sd {dist['std']:.3f} · "
                f"{dist['share_near_neutral']:.1%} near-neutral"
            )
        print(f"\n{'horizon':<10}{'mean IC':>10}{'t (NW)':>9}{'IC IR':>8}{'% days +':>10}{'days':>8}")
        print("-" * 86)
        for row in validation.get("information_coefficient", []):
            if not row.get("days"):
                continue
            print(
                f"{row['horizon_days']:>2}d{'':<7}{row['mean_ic']:>+10.4f}"
                f"{row['ic_tstat']:>+9.2f}{row['ic_ir']:>8.2f}"
                f"{row['share_positive']:>10.1%}{row['days']:>8d}"
            )

        buckets = validation.get("quintile_spread_5d", [])
        if buckets:
            print(f"\n{'quintile':<10}{'sentiment':>12}{'5d return':>12}{'hit rate':>11}{'n':>10}")
            print("-" * 86)
            for row in buckets:
                print(
                    f"Q{row['bucket']:<9}{row['mean_sentiment']:>+12.3f}"
                    f"{row['mean_forward_return']:>+12.3%}{row['hit_rate']:>11.1%}"
                    f"{row['observations']:>10,d}"
                )
            spread = buckets[-1]["mean_forward_return"] - buckets[0]["mean_forward_return"]
            print(f"{'Q5 - Q1':<10}{'':<12}{spread:>+12.3%}")

    base = str(config.BASE_COST_BPS)
    print(f"\n{'='*86}")
    print(f"HEADLINE — at {config.BASE_COST_BPS:.0f}bps round-trip cost")
    print(f"{'='*86}")
    header = f"{'arm':<20}{'CAGR':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>9}{'t-NW':>7}{'trades':>8}{'hit':>7}"
    print(header)
    print("-" * 86)

    for arm in ARMS:
        s = results["arms"][arm]["by_cost"][base]
        print(
            f"{arm:<20}{s['cagr']:>8.2%}{s['volatility']:>8.1%}{s['sharpe']:>8.2f}"
            f"{s['max_drawdown']:>9.1%}{s['nw_tstat']:>7.2f}{s['trades']:>8d}"
            f"{s.get('hit_rate', float('nan')):>7.1%}"
        )

    b = results["benchmark"]["buy_and_hold"]
    print("-" * 86)
    print(
        f"{'buy_and_hold':<20}{b['cagr']:>8.2%}{b['volatility']:>8.1%}{b['sharpe']:>8.2f}"
        f"{b['max_drawdown']:>9.1%}{b['nw_tstat']:>7.2f}{'-':>8}{'-':>7}"
    )

    print(f"\n{'='*86}")
    print("COST SENSITIVITY — Sharpe by round-trip cost")
    print(f"{'='*86}")
    costs = [str(c) for c in config.COST_GRID_BPS]
    print(f"{'arm':<20}" + "".join(f"{c + 'bps':>12}" for c in costs))
    print("-" * 86)
    for arm in ARMS:
        row = "".join(
            f"{results['arms'][arm]['by_cost'][c]['sharpe']:>12.2f}" for c in costs
        )
        print(f"{arm:<20}{row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RSI/sentiment study")
    parser.add_argument("--smoke", action="store_true",
                        help="5 tickers, 2 years — exercises every path quickly")
    parser.add_argument("--rsi", choices=["14", "2"], default="14",
                        help="RSI length to trade (default 14)")
    parser.add_argument("--force-data", action="store_true", help="re-download bars/news")
    parser.add_argument("--force-sentiment", action="store_true", help="re-score articles")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    np.random.seed(config.RANDOM_SEED)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features = build(args.smoke, args.force_data, args.force_sentiment)
    results = run_all(features, rsi_col=f"rsi_{args.rsi}")
    results["config"]["smoke"] = args.smoke

    out_path = config.RESULTS_DIR / ("results_smoke.json" if args.smoke else "results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=_json_safe)

    report(results)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
