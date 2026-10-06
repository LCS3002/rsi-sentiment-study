"""Study parameters — all fixed a priori, none fitted.

This is the central methodological commitment of the study, so it lives in one file where
it can be audited. Every number here was chosen from convention before any result was
computed: RSI 14 and 30/70 are Wilder's originals, RSI(2) is Connors' short-term
reversal, a 5-day hold is the standard horizon for short-term reversal work, and the
cost grid spans the plausible range for US large-cap round trips.

Nothing in this file was tuned to improve a backtest. That is not modesty — it is the
only honest answer available to "how do you know this isn't overfitted?" when a study has
one researcher, one dataset and no out-of-sample period held in reserve.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

NEWS_CACHE = DATA_DIR / "news.parquet"
BARS_CACHE = DATA_DIR / "bars.parquet"
SENTIMENT_CACHE = DATA_DIR / "sentiment.parquet"

# ── Data ──────────────────────────────────────────────────────────────────────

# 105,540 articles over 100 US large caps, 2009-12-14 to 2023-10-23. Verified to carry
# a per-article ticker and date; see docs/data.md.
NEWS_DATASET = "oliverwang15/us_stock_news_with_price"

# Daily bars come from yfinance rather than the dataset's own bundled ts_* price columns,
# so the adjustment basis is known and real opens are available for next-open execution.
BAR_SOURCE = "yfinance"

# A ticker needs enough articles and enough span to contribute anything measurable.
MIN_ARTICLES_PER_TICKER = 200
MIN_YEARS_PER_TICKER = 3.0

# yfinance needs a little runway before the first news date to warm up the indicators.
BAR_WARMUP_DAYS = 90

# ── Sentiment ─────────────────────────────────────────────────────────────────

SENTIMENT_MODEL = "ProsusAI/finbert"
SENTIMENT_BATCH_SIZE = 64

# Characters of (title + body) fed to FinBERT. 512 chars is roughly the headline plus the
# opening two or three sentences — the lede, which is where a news article puts the
# tradeable claim. This is a throughput decision as well as a modelling one, and the two
# happen to agree: FinBERT on CPU runs at about 6 articles/sec at 1500 chars versus ~18 at
# 512, which is the difference between a 5-hour and a 1.5-hour pass over the corpus.
# Raising it re-scores from scratch (the sentiment cache keys on this value), so the
# full-text variant is a one-line change and a long wait, not a rewrite.
MAX_TEXT_CHARS = 512

# Half-life in trading days for the sentiment EWMA. News does not stop mattering the day
# after it prints, but it decays; 3 days is a conventional short-horizon choice.
SENTIMENT_HALFLIFE_DAYS = 3.0

# ── Signals ───────────────────────────────────────────────────────────────────

RSI_LENGTH_SLOW = 14          # Wilder's original
RSI_LENGTH_FAST = 2           # Connors short-term reversal; far more trades
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0

# RSI(2) is a much twitchier series, so it uses the thresholds that variant is normally
# run with rather than reusing 30/70.
RSI_FAST_OVERSOLD = 10.0
RSI_FAST_OVERBOUGHT = 90.0

# A sentiment reading must clear this to count as meaningfully positive or negative.
# score = p_positive - p_negative, so this is on a [-1, 1] scale.
SENTIMENT_THRESHOLD = 0.20

# ── Backtest ──────────────────────────────────────────────────────────────────

HOLD_DAYS = 5                 # fixed horizon; no exit optimisation
EXIT_ON_RSI_REVERSION = True  # also exit early if RSI crosses back through 50
RSI_EXIT_LEVEL = 50.0

# Equal-weight across concurrent positions, capped so a single name cannot dominate.
MAX_CONCURRENT_POSITIONS = 20
POSITION_WEIGHT_CAP = 0.10

# Round-trip cost in basis points. The study is reported across all of these — the point
# is to show where the edge dies, not to pick the flattering one.
COST_GRID_BPS = (0.0, 5.0, 10.0, 20.0)
BASE_COST_BPS = 10.0          # the headline figure

# ── Evaluation ────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0          # Sharpe is reported excess of zero; stated, not hidden

# Newey-West lag for the t-stat on mean daily return. Overlapping 5-day holds induce
# serial correlation, so a plain OLS standard error would be too small.
NEWEY_WEST_LAGS = 10

# Three roughly equal subperiods, to show the result is not one regime.
SUBPERIODS = (
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2023", "2020-01-01", "2023-12-31"),
)

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Smoke mode: enough to exercise every code path end to end quickly. One year rather than
# two, because sentiment scoring dominates the runtime and a smoke run that takes twenty
# minutes is not a smoke run. It is not meant to produce meaningful statistics.
SMOKE_TICKERS = ("AAPL", "MSFT", "AMZN", "NVDA", "TSLA")
SMOKE_START = "2022-01-01"
SMOKE_END = "2022-12-31"
