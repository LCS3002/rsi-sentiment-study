# Does news sentiment improve an RSI mean-reversion signal?

**A walk-forward study on {{config.tickers}} US large caps, {{config.start}} to {{config.end}}, with transaction costs.**

Lorenz Huber · [github.com/LCS3002/rsi-sentiment-trading-study](https://github.com/LCS3002/rsi-sentiment-trading-study)

> Every figure in this note is generated from `results/results.json` by `study/note.py`.
> None is transcribed by hand, and `python -m study.note --check` fails if the note and
> the results have drifted apart.

---

## Summary

**No.** Conditioning RSI mean-reversion entries on news sentiment did not improve the
signal. Sharpe fell slightly, from {{arm.rsi.sharpe}} to {{arm.rsi_filtered.sharpe}}, and
annualised return from {{arm.rsi.cagr:pct}} to {{arm.rsi_filtered.cagr:pct}}.

The reason is visible one layer down: **the sentiment score has no measurable predictive
power in this sample.** Its information coefficient against forward returns is
{{validation.ic.5.mean_ic:sig4}} at five days with a t-statistic of
{{validation.ic.5.ic_tstat}}, and sorting entries into sentiment quintiles produces a
top-minus-bottom spread of {{validation.quintile_spread:pct3}} — indistinguishable from
zero, and marginally the wrong sign. A filter cannot work through a channel that carries
no information.

The falsification arm makes this sharper rather than softer. The oversold days the filter
*removes* — those with genuinely negative news — returned **{{arm.rsi_negative_news.avg_trade:pct}}
per trade against {{arm.rsi_filtered.avg_trade:pct}} for the days it keeps.** The trades
the hypothesis says to avoid were, on average, the better ones.

One thing the filter did do: it cut the tail. Maximum drawdown improved from
{{arm.rsi.max_drawdown:pct}} to {{arm.rsi_filtered.max_drawdown:pct}}, volatility from
{{arm.rsi.volatility:pct}} to {{arm.rsi_filtered.volatility:pct}}, and the worst single
trade from {{arm.rsi.worst_trade:pct}} to {{arm.rsi_filtered.worst_trade:pct}}. That is a
real effect, but it is a risk trade rather than an edge: you give up mean return in
exchange for a thinner left tail.

**None of it came close to simply holding the universe,** which returned
{{bench.cagr:pct}} a year at a Sharpe of {{bench.sharpe}} over the same period. That
comparison needs a caveat — the strategy sits in cash most of the time — but the gap is
too wide for that to account for it.

---

## 1. The question, and why it might have an answer

Short-term RSI mean reversion buys names that have fallen. The strategy is old, well
documented, and usually attributed to liquidity provision rather than prediction: a seller
who needs to transact quickly pays whoever is willing to take the other side, and the
price reverts once that pressure clears. Nagel (2012) makes this case directly, showing
that reversal returns behave like compensation for supplying liquidity, and that they rise
when liquidity is scarce.

That account has an immediate implication. If the return comes from absorbing *uninformed*
selling pressure, then it should not be there when the selling is informed — when the
price fell because something happened and the new price is correct. Buying that dip is not
liquidity provision; it is standing in front of news.

**Hypothesis.** An oversold reading is a buy when the decline is uninformed and a trap when
it is informed. Conditioning RSI entries on news sentiment should therefore raise the hit
rate and cut the left tail, at the cost of fewer trades.

Tetlock (2007) supplies the other half: media pessimism carries information about returns,
which is what makes news sentiment a candidate proxy for "was this decline informed?" in
the first place.

Neither idea is mine. What this note contributes is a measurement of whether the
combination survives contact with transaction costs — and a test of whether it works
through the channel the hypothesis claims, rather than by coincidence.

### The four arms

The comparison is the result, so each arm isolates one thing.

| Arm | Entry rule | What it tests |
|---|---|---|
| **RSI alone** | RSI(14) < 30 | The baseline reversal signal |
| **Sentiment alone** | sentiment > +{{config.sentiment_threshold}} | Sentiment with no reversal signal |
| **RSI + sentiment filter** | RSI < 30 **and not** sentiment < −{{config.sentiment_threshold}} | **The hypothesis** |
| **RSI on negative news** | RSI < 30 **and** sentiment < −{{config.sentiment_threshold}} | **The falsification arm** |

The fourth arm is what makes this falsifiable. It trades precisely the entries the filter
removes. If the mechanism is real, it should be materially worse than the filtered arm. If
the two are indistinguishable, the filter is not working through the channel the hypothesis
describes — whatever the headline Sharpe happens to be.

A name with no news is treated as *uninformed* and passes the filter, because that is what
the hypothesis says to do with a dip nobody wrote about. The falsification arm requires
actual negative coverage and so excludes it.

---

## 2. Data

| | |
|---|---|
| News | 103,036 articles, {{config.start}} to {{config.end}} |
| Universe | {{config.tickers}} US large caps |
| Observations | {{config.ticker_days:int}} ticker-days |
| Bars | yfinance daily, split- and dividend-adjusted |
| Sentiment | `ProsusAI/finbert`, continuous `p_positive − p_negative`, EWMA half-life {{config.sentiment_halflife_days}} days |

Bars come from yfinance rather than the news corpus's own bundled price columns, so the
adjustment basis is known and real **opens** are available. `auto_adjust=True` adjusts
open, high, low and close on one basis; mixing an adjusted close with a raw open would
manufacture a return at every split in the sample.

Two names in the corpus — EA and WBA — no longer resolve at all, both having been taken
private in 2025. They are excluded. That is worth stating plainly rather than quietly
dropping, because it is survivorship bias happening in miniature and in public: a universe
assembled from a recent index membership has already lost the companies that left.

---

## 3. Method

Three rules do most of the work, and each is enforced by a test rather than simply stated.

**No parameters were tuned.** Every value in `study/config.py` is a convention fixed before
any result was computed: RSI 14 with 30/70 thresholds is Wilder's original, RSI(2) with
10/90 is Connors', {{config.hold_days}} days is the standard short-reversal horizon. There
is no in-sample optimisation because there is no optimisation. With one researcher, one
dataset and no period held in reserve, that is the only honest answer available to "how do
you know this isn't overfitted?"

**Signals execute at the next open.** A signal computed from the close of day D is executed
at the open of D+1. The delay is carried in the data as an explicit `next_open` column
rather than applied inside the backtest loop, so it can be inspected directly. Two
tests enforce it: no entry may precede its own signal, and perturbing the last twenty days
of prices must leave every earlier daily return bit-identical. Both were verified to fail
against a deliberately introduced same-day-execution bug.

News dated day D is likewise actionable only from D+1. The corpus carries date-level
timestamps with no time of day, so this is forced rather than cautious: there is no way to
know whether a story broke before the open or after the close.

**Costs are charged, and the study is reported across a grid** of {{config.cost_grid_bps}}
basis points round trip, with {{config.base_cost_bps:.0f}}bps as the headline. The point is to
show where the edge dies, not to find the cost assumption that flatters it.

Capital is divided into {{config.max_positions}} fixed slots; each trade takes one and
unused slots earn zero. The tempting alternative — equal-weighting whatever happens to be
open — silently levers the strategy up whenever few signals fire, and makes the Sharpe
incomparable with buy-and-hold. When more signals fire than there are free slots, the most
oversold candidates are taken first.

Long only. The hypothesis is about buying dips, and claiming short results would require
borrow cost and availability to be modelled before they meant anything.

Standard errors are Newey-West with {{config.newey_west_lags}} lags. Overlapping
{{config.hold_days}}-day holds make the daily return series serially correlated, so an OLS
standard error would be too small — and too small in the direction that inflates every
t-statistic in the study.

---

## 4. Is the sentiment signal worth anything?

The study rests on FinBERT, so FinBERT is interrogated before it is used.

The obvious test is the wrong one. `ProsusAI/finbert` is fine-tuned on the Financial
PhraseBank, so scoring it against that benchmark measures memorisation and returns a number
near 0.97 that means nothing at all. A great many write-ups quoting FinBERT's accuracy are
quoting exactly that figure.

The test here is benchmark-free instead: **does the score predict forward returns in this
sample?** That question cannot be contaminated by training data, because realised returns
were not in anyone's.

Coverage is {{validation.dist.coverage:pct1}} of ticker-days, with
{{validation.dist.share_near_neutral:pct1}} of scored readings near neutral.

**Information coefficient** — the daily cross-sectional rank correlation between sentiment
and forward return. Computed per day across names, so it cannot be inflated by a
market-wide move that happens to coincide with generally good news.

{{table.ic}}

**Forward return by sentiment quintile**, ranked within each day:

{{table.quintiles}}

**There is nothing here.** The information coefficient is between
{{validation.ic.1.mean_ic:sig4}} and {{validation.ic.20.mean_ic:sig4}} depending on
horizon. That is small before significance is even considered: an IC of 0.005 is a rank
correlation of half of one percent between today's sentiment and the next five days. None of the three horizons
clears conventional significance once standard errors account for the overlap, and the
t-statistic *falls* as the horizon lengthens ({{validation.ic.1.ic_tstat}} at one day,
{{validation.ic.20.ic_tstat}} at twenty) — the opposite of what a real slow-decaying
signal looks like.

The quintile table says the same thing more legibly. Sorting every RSI entry by how
positive the news was produces five buckets whose five-day returns sit between
+0.42% and +0.50%, in no particular order. The most *negative* quintile returned slightly
**more** than the most positive, a top-minus-bottom spread of
{{validation.quintile_spread:pct3}}. That is indistinguishable from zero, and the sign
points the wrong way for the hypothesis.

Two features of the score itself are worth recording. Coverage is high at
{{validation.dist.coverage:pct1}} of ticker-days, so this is not a sparsity problem. And
FinBERT is markedly optimistic on this corpus: {{validation.dist.share_strongly_positive:pct1}}
of readings are strongly positive against {{validation.dist.share_strongly_negative:pct1}}
strongly negative, with a mean of {{validation.dist.mean:sig4}}. A model that calls most
financial news good news has limited room to flag the bad.

This bounds everything that follows. Whatever the strategy results show, they cannot be
evidence that sentiment carries tradeable information, because measured directly against
returns across 270,184 ticker-days it does not.

---

## 5. Results

All figures net of {{config.base_cost_bps:.0f}}bps round-trip cost.

{{table.headline}}

![Equity curves](../results/exhibit_1_equity_curves.png)

The baseline reversal signal works, modestly. RSI alone earned {{arm.rsi.cagr:pct}} a year
at a Sharpe of {{arm.rsi.sharpe}} across {{arm.rsi.trades:int}} trades, with a
{{arm.rsi.hit_rate:pct1}} hit rate and a Newey-West t-statistic of {{arm.rsi.nw_tstat}}.
That is a real effect, and consistent with the liquidity-provision account: you are paid a
little, fairly reliably, for taking the other side of short-term selling.

Adding the sentiment filter changed almost nothing, and what it changed was not Sharpe.
Return fell from {{arm.rsi.cagr:pct}} to {{arm.rsi_filtered.cagr:pct}} and Sharpe from
{{arm.rsi.sharpe}} to {{arm.rsi_filtered.sharpe}}, while the hit rate moved from
{{arm.rsi.hit_rate:pct1}} to {{arm.rsi_filtered.hit_rate:pct1}} — a fifth of a percentage
point, on {{arm.rsi_filtered.trades:int}} trades. The hypothesis predicted a materially
better hit rate. It is not there.

Sentiment alone is the clearest failure: {{arm.sentiment.trades:int}} trades for a Sharpe
of {{arm.sentiment.sharpe}} and a negative return. It is fully invested
({{arm.sentiment.mean_positions}} of {{config.max_positions}} slots on average) and turns
over every {{arm.sentiment.avg_days_held}} days, which is why cost destroys it — see below.

And the benchmark wins comfortably. Buy-and-hold returned {{bench.cagr:pct}} at Sharpe
{{bench.sharpe}}, against {{arm.rsi.sharpe}} for the best strategy arm.

**The honest caveat on that comparison**, which cuts in the strategy's favour and still
does not save it: the RSI arms hold only {{arm.rsi.mean_positions}} of
{{config.max_positions}} slots on average, so roughly five-sixths of capital sits in cash
earning nothing. CAGR against a fully-invested benchmark is therefore not like-for-like.
Sharpe is the fair comparison because it normalises for that — and on Sharpe the gap is
{{bench.sharpe}} against {{arm.rsi.sharpe}}, which is not a gap a deployment artefact
closes. The period matters too: 2010–2023 in US large-cap technology was one of the
strongest equity runs on record, and a long-only strategy in cash most of the time was
never going to beat it.

### Where the edge dies

Sharpe ratio as costs are applied:

{{table.cost}}

![Cost sensitivity](../results/exhibit_2_cost_sensitivity.png)

This is the exhibit that separates the arms, and it does so by turnover rather than by
signal quality.

**Sentiment alone is a cost illusion.** At zero cost it has the *highest* Sharpe of any
arm — {{arm.sentiment.cost.0.sharpe}}, better than RSI's {{arm.rsi.cost.0.sharpe}}. By
10bps it is {{arm.sentiment.sharpe}}; by 20bps it is
{{arm.sentiment.cost.20.sharpe}}. Nothing about the signal changed; it simply trades
{{arm.sentiment.trades:int}} times with an average holding period of
{{arm.sentiment.avg_days_held}} days, and an average trade of
{{arm.sentiment.avg_trade:pct}} does not survive paying a spread twice. Reported gross, this strategy looks like an edge. It is not one.

**The RSI arms are robust to cost** precisely because they trade so little:
{{arm.rsi.trades:int}} trades over fourteen years, holding {{arm.rsi.avg_days_held}} days,
so Sharpe decays only from {{arm.rsi.cost.0.sharpe}} to {{arm.rsi.cost.20.sharpe}} across
the whole grid. Whatever is wrong with this strategy, frictions are not it.

That contrast is the practical lesson of the study, independent of the hypothesis: cost
sensitivity is a property of turnover, and a backtest reported gross cannot tell you which
of these two cases you are looking at.

### The mechanism

If the filter works through the channel the hypothesis describes, the trades it removes
should be worse than the trades it keeps, and outcomes should improve as sentiment at entry
improves.

![Mean trade outcome by sentiment at signal](../results/exhibit_3_sentiment_buckets.png)

On the headline numbers the falsification arm looks like weak support: RSI on negative news
returned {{arm.rsi_negative_news.cagr:pct}} against {{arm.rsi_filtered.cagr:pct}} for the
filtered arm. Dips with bad news did worse. That is the predicted direction.

It does not survive a second look. That arm holds only
{{arm.rsi_negative_news.mean_positions}} of {{config.max_positions}} slots on average
against {{arm.rsi_filtered.mean_positions}} for the filtered arm, so most of the CAGR gap
is capital sitting idle, not trades going wrong. **Per trade, the ranking inverts:**
{{arm.rsi_negative_news.avg_trade:pct}} on the negative-news days against
{{arm.rsi_filtered.avg_trade:pct}} on the ones the filter keeps. The entries the hypothesis
identifies as traps were, on average, the better trades.

The bucket chart agrees. Sorting RSI entries by sentiment at signal gives five groups whose
mean outcomes run +0.62%, +1.01%, +0.56%, +0.54%, +1.01% — no ordering, no trend, roughly
450 trades each. If sentiment separated informed declines from uninformed ones, this is
where it would show, and it is flat.

**Where the hypothesis does get partial support is the tail, not the mean.** The filter
removed the single worst trade in the sample: the worst loss falls from
{{arm.rsi.worst_trade:pct}} to {{arm.rsi_filtered.worst_trade:pct}} once negative-news
entries are excluded, and maximum drawdown improves from {{arm.rsi.max_drawdown:pct}} to
{{arm.rsi_filtered.max_drawdown:pct}}. So the mechanism may be real in the extreme — very
bad news does precede very bad outcomes — while carrying no information at all about the
average case. That is a narrower claim than the one I set out to test, and with
{{arm.rsi_negative_news.trades:int}} trades in that arm it is not one this sample can
establish. It is a hypothesis for a larger study, not a finding of this one.

### Subperiod stability

{{table.subperiods}}

The baseline signal is present in all three subperiods — Sharpe 0.50, 1.09 and 0.39 — so it
is not an artefact of one regime, though it is clearly strongest in 2015–2019 and weakest
in 2020–2023, where the drawdown also triples. The filtered arm tracks it closely
throughout (0.50, 1.19, 0.34), which is the subperiod restatement of the main finding: the
filter changes very little, consistently.

Sentiment alone deteriorates monotonically across the three windows, ending at a negative
Sharpe in 2020–2023. Buy-and-hold beats every arm in every subperiod.

---

## 6. What this cannot tell you

- **Survivorship bias.** The universe comes from a recent index membership, so companies
  that failed or were acquired are absent. Two of them — EA and WBA — left during the
  sample and are missing for exactly that reason. The benchmark is flattered most.
- **Sector concentration.** The universe is NASDAQ/technology-heavy, plus biotech. These
  names share a strong common factor, so {{config.tickers}} tickers carry materially less
  independent information than {{config.tickers}} names normally would, and every t-statistic
  here should be read with that in mind.
- **Coverage bias.** News coverage skews toward large caps and toward eventful days. A name
  with no article is not a name about which nothing happened.
- **FinBERT is a noisy feature, not ground truth.** Section 4 puts a number on how noisy.
  It is also easy to surprise: "Apple traded flat" scores as 92% negative.
- **Date-level timestamps.** No intraday precision, so a story that broke pre-open is
  indistinguishable from one that broke after the close.
- **One market, one regime of market structure.** US large caps over
  {{config.start}}–{{config.end}}, a period with unusually low rates for most of its span.

---

## 7. What I would do next

Three things, in the order I think they would move the answer.

**Replace the sentiment model before replacing the hypothesis.** The null here is a joint
test of "sentiment predicts returns" and "FinBERT measures sentiment", and section 4 says
the second conjunct is doing the damage: a model that calls
{{validation.dist.share_strongly_positive:pct1}} of financial news strongly positive and
only {{validation.dist.share_strongly_negative:pct1}} strongly negative has little
discriminating power left for the cases that matter. Before concluding anything about news
and reversals, I would re-run this with a model that is not fine-tuned on a three-class
phrase benchmark — or with a direct measure such as abnormal volume or the realised
overnight gap, which require no model at all and are not obviously worse proxies for "was
this decline informed?"

**Get timestamps.** Date-level resolution forces the next-open convention, which is
correct but blunt: it cannot distinguish a story that broke at 6am from one that broke
after the close, and those are different trades. Intraday timestamps would make the event
study possible and would sharpen the tail result in section 5, which is where the only
surviving signal appears to be.

**Widen the universe beyond large-cap technology.** Eighty-four names that share a dominant
common factor provide far less independent information than the count suggests, and the
liquidity-provision story predicts the effect should be *strongest* in less liquid names —
which is exactly where this sample has no observations. A universe that included small and
mid caps would test the mechanism where it should be most visible, rather than where news
coverage happens to be densest.

What I would not do is tune the parameters. The result is a null, and a null reached
without optimisation is worth more than a positive result reached with it.

---

## Reproducing this

```bash
pip install -r requirements.txt
python -m study.run          # regenerates results/results.json and every exhibit
python -m study.note         # regenerates this note from those results
pytest -q                    # no network or model required
```

The suite is {{meta.test_count}} test functions across seven files (pytest reports more,
since several are parametrised). Scoring the corpus with FinBERT on CPU takes roughly two
hours and is checkpointed every {{config.sentiment_chunk_size:int}} articles; everything
downstream of it runs in minutes.

**Sources.** Tetlock, P. (2007), "Giving Content to Investor Sentiment: The Role of Media
in the Stock Market", *Journal of Finance* 62(3). · Nagel, S. (2012), "Evaporating
Liquidity", *Review of Financial Studies* 25(7).
