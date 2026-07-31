# Audit: This Project vs. López de Prado's "7 Reasons Most ML Funds Fail"

Lecture 10 (ssrn-3447398) lists the seven errors he believes account for
most ML-fund failures. This is an honest audit of our system against each,
with the evidence.

---

## #1 — The Sisyphus Paradigm

> "Firms directing quants to work in silos, or to develop individual
> strategies, are asking the impossible."

**Status: applies, and is structural.** This project is one strategy
developed in one silo — exactly the pattern the lecture calls futile. The
"meta-strategy" alternative (specialized teams: data curation, feature
research, backtesting, execution) isn't available to a solo effort.

**Mitigation actually in place**: the project has been unusually disciplined
about *not* letting a single researcher's optimism drive conclusions —
frozen holdouts, mechanical leakage tests, chance baselines. That's the
closest a solo project gets to the adversarial separation of duties the
lecture recommends.

---

## #2 — Integer Differentiation

> "Returns are stationary but memory-less; prices have memory but are
> non-stationary."

**Status: CONFIRMED ERROR — we were committing it. Now fixed.**

Every return feature in our set (`logret_1` … `logret_200`, all the
`_slope_`, `_dist_` features) uses integer differentiation, d=1.

Measured directly on BTC hourly log price (`ffd_d_sweep.csv`):

| d | ADF stat | stationary @95%? | correlation with original |
|---|---|---|---|
| 0.0 | −1.49 | No | 1.000 |
| **0.1** | **−2.95** | **Yes** | **0.979** |
| 0.2 | −5.82 | Yes | 0.923 |
| 0.4 | −20.38 | Yes | 0.734 |
| **1.0 (what we used)** | −101.35 | Yes | **0.011** |

d=0.1 is stationary while retaining **97.9%** correlation with the price
series. Our d=1 features retain **1.1%**. We were discarding ~98% of the
memory for no statistical benefit — precisely the error described.

**Fixed**: `frac_diff_ffd` in `afml_lectures.py` (vectorized via convolution,
proven identical to the direct loop, and proven causal by a
corrupt-the-future test). 10 FFD features added at d ∈ {0.1…0.5}.

---

## #3 — Inefficient Sampling

> "Information does not arrive to the market at a constant entropy rate…
> dollar bars tend to exhibit more stable sampling frequencies."

**Status: PARTIALLY APPLIES.** Everything in this project is built on
**hourly time bars** — the sampling scheme the lecture explicitly advises
against.

**Implemented**: `dollar_bars`, `volume_bars` (Lecture 3 slides 5–6) and the
`cusum_filter` event sampler (slide 10), all unit-tested. CUSUM
structural-break features are now in the feature set.

**Honest caveat**: our source data is *already* hourly OHLCV, not tick data.
Dollar bars built by aggregating hourly bars are a coarse approximation of
true dollar bars — the intra-hour path is lost. Getting the real benefit
requires tick/trade data, which we don't have. Flagging this rather than
pretending the aggregation is equivalent.

---

## #4 — Wrong Labeling

> "Time bars do not exhibit good statistical properties… the same threshold
> τ is applied regardless of the observed volatility."

**Status: MOSTLY ADDRESSED, one residual issue.**

- Folder 10's label is `sign(close[t+n] − close[t])` — threshold-free, so
  the "fixed τ across volatility regimes" critique doesn't bite directly.
- But the lecture's deeper point does apply: a fixed *horizon* ignores the
  price *path*. A position that would have been stopped out mid-window still
  gets labeled by its endpoint.
- Folders 03–06 *do* use triple-barrier labeling (the lecture's recommended
  method) with a volatility-scaled barrier `max(2×ATR%, 1.0)` — correct in
  spirit, though the 1.0% floor reintroduces a fixed threshold that will
  dominate in low-vol regimes, which is the exact failure mode described.

---

## #5 — Weighting of Non-IID Samples

> "Because labels overlap in time, we cannot be certain about what observed
> features caused an effect."

**Status: MEASURED, and it materially changes our error bars.**

Overlapping labels mean far fewer independent observations than rows
(`cv_leakage_comparison.py`):

| horizon | mean concurrency | avg uniqueness | effective n (of 30,000) |
|---|---|---|---|
| n=1 | 2.0 | 0.500 | 15,000 (50.0%) |
| n=5 | 6.0 | 0.167 | 5,000 (16.7%) |
| n=20 | 21.0 | 0.048 | 1,429 (4.8%) |
| n=50 | 51.0 | 0.020 | 589 (2.0%) |

**Consequence**: every confidence interval this project has reported for
horizons > 1 is too narrow — by ~4.6× at n=20 and ~7× at n=50.
`n=20: AUC 0.5379 ± 0.0183` should read closer to ±0.08.

**Implemented**: `num_co_events`, `get_avg_uniqueness`, `seq_bootstrap`,
`sample_weights_by_return`, `time_decay_weights` in `afml.py`.

**Note on where it bites**: for our *fixed*-horizon labels, uniqueness is
identical across rows, so uniqueness weighting changes nothing. It matters
for **variable**-span labels — i.e. the triple-barrier datasets in folders
03–06, where a trade exiting on a barrier after 2 bars is far more unique
than one running a full 10-bar timeout. That remains the highest-value
unapplied item.

---

## #6 — Cross-Validation Leakage

**Status: NOT COMMITTED here, and now quantified.**

Our walk-forward is strictly forward-looking with an explicit embargo — more
conservative than `PurgedKFold` (which also uses purged post-test data).

We measured what shuffling *would* have cost, same model and data, only the
CV scheme changing:

| horizon | shuffled StratifiedKFold | purged + 1% embargo | inflation |
|---|---|---|---|
| n=1 | 0.5530 | 0.5529 | **+0.0001** |
| n=5 | 0.6594 | 0.5410 | **+0.1184** |
| n=20 | 0.8503 | 0.5232 | **+0.3271** |
| n=50 | 0.9209 | 0.4845 | **+0.4364** |

The n=1 row is the control proving the mechanism: no label overlap → no
inflation. At n=50, shuffling reports **0.92** on data whose honest AUC is
**0.48**.

---

## #7 — Backtest Overfitting

> "After only 1,000 independent backtests, the expected maximum Sharpe ratio
> is 3.26, even if the true Sharpe ratio of the strategy is zero!"

**Status: WAS A LIVE RISK — we ran 129,600 backtests. Now formally tested.**

Applying the False Strategy Theorem (Lecture 10 p39 / Lecture 6) and CSCV
to our own strategy search (`dsr_pbo_on_search.py`):

```
Best config found (profit factor 1.097 in-sample):
  per-trade Sharpe observed                          = 0.0345
  E[max Sharpe] by LUCK ALONE across 129,600 trials  = 0.1093
  observed / expected-by-luck                        = 0.32x

  Deflated Sharpe Ratio @ 129,600 trials             = 0.0035   (need >0.95)
  Probability of Backtest Overfitting (CSCV)         = 0.4372
```

**The best configuration we found is 3× WORSE than what pure luck would
produce from that many trials.** DSR 0.0035 vs. the 0.95 bar. PBO 0.44 —
selecting the in-sample winner is essentially a coin flip out of sample.

This is the formal, theory-backed confirmation of what the earlier
out-of-sample test showed empirically (54 configs profitable in both periods
vs. 255.6 expected by chance).

---

## Scorecard

| # | Error | Verdict |
|---|---|---|
| 1 | Sisyphus paradigm | Applies structurally; partially mitigated by process discipline |
| 2 | Integer differentiation | **Was committing it** → fixed with FFD |
| 3 | Inefficient sampling | Partially applies; tools built, limited by lack of tick data |
| 4 | Wrong labeling | Mostly addressed; 1.0% barrier floor is a residual issue in folders 03–06 |
| 5 | Non-IID weighting | Measured; **widens all our error bars for n>1** |
| 6 | CV leakage | Not committed; now quantified (up to +0.44 AUC if it had been) |
| 7 | Backtest overfitting | Was a live risk; **formally refuted the best config** (DSR 0.0035) |

Two of the seven were genuinely being committed (#2, and #7 as a risk the
search created). Both are now fixed or formally tested. #5 doesn't change
our conclusions but does widen the uncertainty around them.
