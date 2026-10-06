# Does news sentiment improve an RSI mean-reversion signal?

A walk-forward study on 86 US large caps, 2010–2023, with transaction costs.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Articles](https://img.shields.io/badge/articles-105%2C493-informational)
![Universe](https://img.shields.io/badge/universe-86%20large%20caps-informational)

> **Status:** engine complete and tested; full-sample results pending. See
> [`notes/research-note.md`](notes/research-note.md) for the write-up.

---

## The question

Short-term RSI mean reversion buys names that have fallen. Some of those declines are
noise — a liquidity imbalance that reverts — and some are information, where the price
fell because something happened and it is not coming back.

**Hypothesis.** An oversold reading is a buy when the decline is *uninformed* and a trap
when it is *informed*. Conditioning RSI entries on news sentiment should therefore raise
the hit rate and cut the left tail, at the cost of fewer trades.

This sits on two established results: Tetlock (2007) on media sentiment predicting returns,
and Nagel (2012) on short-term reversal returns as compensation for liquidity provision.
The contribution here is not the idea — it is measuring honestly whether the combination
survives costs.

## The four arms

The comparison *is* the result, so each arm isolates one thing:

| Arm | Entry rule | What it tests |
|---|---|---|
| `rsi` | RSI(14) < 30 | The baseline reversal signal alone |
| `sentiment` | sentiment > +0.20 | Sentiment alone, with no reversal signal |
| `rsi_filtered` | RSI < 30 **and not** sentiment < −0.20 | **The hypothesis** — buy the dip unless the news is bad |
| `rsi_negative_news` | RSI < 30 **and** sentiment < −0.20 | **The falsification arm** — exactly the trades the filter removed |

Plus equal-weight buy-and-hold as the benchmark.

That last arm is the one that makes the study falsifiable. If the mechanism is real,
`rsi_negative_news` should be materially worse than `rsi_filtered`. If the two are
indistinguishable, the filter is not working through the channel the hypothesis claims,
whatever the headline Sharpe says.

---

## Method

Three commitments do most of the work, and each is enforced by a test rather than asserted
in prose.

**No parameters were tuned.** Every number in `study/config.py` was fixed from convention
before any result was computed — RSI 14 and 30/70 are Wilder's, RSI(2) with 10/90 is
Connors', a 5-day hold is the standard short-reversal horizon. There is no in-sample
optimisation because there is no optimisation at all. With one researcher and one dataset,
that is the only honest answer to "how do you know this isn't overfitted?"

**Next-open execution.** A signal computed from the close of day D executes at the **open
of D+1**. The delay is carried in the data as an explicit `next_open` column rather than
applied inside the backtest loop, so it is visible instead of trusted. News dated day D is
treated as actionable from D+1 — and since the corpus carries date-level timestamps with no
time of day, that is a requirement, not a conservatism.

**Costs are charged, and the whole study is reported across a cost grid** of 0 / 5 / 10 /
20 bps round trip, with 10 bps as the headline. The point is to show where the edge dies.

Capital is divided into fixed slots (20), each trade taking one, with unused slots earning
zero. The tempting alternative — equal-weighting whatever happens to be open — silently
levers up when few signals fire and makes the Sharpe incomparable with buy-and-hold.

Long only: the hypothesis is about buying dips, and claiming short results would need
borrow cost and availability modelled to mean anything.

---

## Data

| | |
|---|---|
| News | [`oliverwang15/us_stock_news_with_price`](https://huggingface.co/datasets/oliverwang15/us_stock_news_with_price) — 105,493 articles after dedup, 2009-12-14 to 2023-10-23 |
| Bars | yfinance daily, `auto_adjust=True` |
| Universe | 86 of 100 tickers clearing ≥200 articles and ≥3 years |
| Sentiment | `ProsusAI/finbert`, scored as a continuous `p_positive − p_negative` in [−1, 1], EWMA with a 3-day half-life |

Bars come from yfinance rather than the dataset's bundled price columns, so the adjustment
basis is known and real **opens** are available. `auto_adjust=True` adjusts open, high, low
and close on one basis — mixing adjusted closes with raw opens would manufacture a return
at every split in the sample.

### Known limitations

Stated here rather than buried, because they bound what the result can mean.

- **Survivorship bias.** The universe is names present in the corpus, which was assembled
  from a recent index membership. Companies that failed or were delisted are absent, so the
  benchmark in particular is flattered.
- **Sector concentration.** The universe is NASDAQ/technology-heavy (AAPL, GOOG, AMZN,
  MSFT, NVDA, INTC, AMD, plus biotech). These names share a strong common factor, so 86
  tickers provide materially less independent information than 86 names normally would.
- **Coverage bias.** News coverage skews toward large caps and toward eventful days. A name
  with no article is not a name about which nothing happened.
- **FinBERT is not ground truth.** `ProsusAI/finbert` is fine-tuned on the Financial
  PhraseBank, so evaluating it against that benchmark would measure nothing. It is also
  easy to surprise — "Apple traded flat" scores as 92% negative. Treat its output as a
  noisy feature.
- **Date-level timestamps.** No intraday precision, so this cannot distinguish a story that
  broke pre-open from one that broke after the close.

---

## Running it

```bash
pip install -r requirements.txt

python -m study.run --smoke     # 5 tickers, 2 years — exercises every path
python -m study.run             # full universe
python -m study.run --rsi 2     # RSI(2) instead of RSI(14)
```

Everything is written to `results/` as JSON and CSV. The note cites those files rather
than transcribing numbers by hand, so a stale figure in the write-up is not possible.

> **Runtime.** Scoring the full corpus with FinBERT on CPU is the slow step — roughly
> 6 articles/sec, so about 4–5 hours for 105k articles. It is cached to parquet afterwards
> and never repeats. `--smoke` takes minutes.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

| File | Covers |
|---|---|
| `test_backtest.py` | **Look-ahead guards** — no entry before its signal resolves, and perturbing future prices cannot change past returns. Plus cost accounting, slot limits, and that the filtered and negative-news arms exactly partition the plain RSI signal. |
| `test_signals.py` | RSI against an independently written textbook implementation; per-ticker isolation; that future news cannot change past sentiment; weekend news rolls to the next open |
| `test_metrics.py` | Newey-West against statsmodels' HAC, and that it is more conservative than OLS under positive autocorrelation |

---

## Structure

```
study/
├── config.py      # every parameter, all fixed a priori
├── data.py        # news corpus + yfinance bars, parquet-cached
├── signals.py     # Wilder RSI, FinBERT scoring, sentiment decay
├── backtest.py    # next-open execution, fixed slots, explicit costs
├── metrics.py     # Sharpe, Newey-West t-stat, drawdown, bucket tables
└── run.py         # CLI; regenerates every number in the note
```
