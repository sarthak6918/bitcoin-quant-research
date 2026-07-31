# López de Prado ORIE 5256 Lectures — Applied to This System

Nine lecture decks (L2–L10) were supplied. This documents what was
implemented, what it changed, and — importantly — what it didn't.

## What was implemented

| Lecture | Technique | File | Tests |
|---|---|---|---|
| L4 §7.4 | Purging, embargo, `PurgedKFold` | `afml.py` | 9/9 |
| L4 / Ch.4 | Concurrency, avg uniqueness, sequential bootstrap, return & time-decay weights | `afml.py` | ✓ |
| L5 / Ch.12 | Combinatorial Purged CV | `afml.py` | ✓ |
| L3 slides 24–29 | Fractional differentiation (FFD), min-d ADF sweep | `afml_lectures.py` | 15/15 |
| L3 slides 5–6 | Dollar bars, volume bars | `afml_lectures.py` | ✓ |
| L3 slide 10 | CUSUM event filter (`getTEvents`) | `afml_lectures.py` | ✓ |
| L6 / L10 p39 | False Strategy Theorem, Deflated Sharpe Ratio | `afml_lectures.py` | ✓ |
| L6 | Probability of Backtest Overfitting (CSCV) | `afml_lectures.py` | ✓ |

Every implementation is original code written against the described
algorithm, unit-tested against properties the lectures assert (e.g. FFD at
d=1 must reduce to plain differencing; PBO on pure noise must be ≈0.5; PBO
with one genuinely skilled strategy must be low).

---

## Result 1 — Integer differentiation: we were committing the error

BTC hourly log price, ADF sweep (`ffd_d_sweep.csv`):

| d | stationary @95%? | correlation with original |
|---|---|---|
| 0.0 | No | 1.000 |
| **0.1** | **Yes** | **0.979** |
| 0.4 | Yes | 0.734 |
| **1.0 (what every feature used)** | Yes | **0.011** |

Textbook confirmation of Lecture 3's thesis: d=0.1 achieves stationarity
retaining 97.9% of the memory; our log returns retained 1.1%.

## Result 2 — …but fixing it did NOT improve prediction

Same protocol, same model, same seeds; only the feature set changed
(`afml_feature_comparison.csv`):

| horizon | baseline (63 feat) | +FFD/CUSUM (76 feat) | delta | fold sd | verdict |
|---|---|---|---|---|---|
| n=1 | 0.5722 | 0.5699 | −0.0022 | 0.0031 | within noise |
| n=5 | 0.5580 | 0.5566 | −0.0014 | 0.0092 | within noise |
| n=20 | 0.5465 | 0.5513 | +0.0047 | 0.0179 | within noise |

**The theoretically-correct fix produced no measurable predictive lift.**

The FFD/CUSUM features alone score 0.5168 / 0.5095 / 0.5228 — above random,
so they *do* carry signal, but signal we already had. The likely reason:
the baseline already contains `logret` at nine lookbacks (1, 2, 3, 5, 10,
20, 50, 100, 200), and a gradient-boosted tree can combine those into
approximations of the same long-memory structure FFD encodes. FFD is a
**more parsimonious and more principled** representation of that
information — not a **richer** one.

This is worth stating plainly because it would have been easy to report
"implemented López de Prado's fractional differentiation" and let the
implication of improvement stand. It didn't improve anything.

## Result 3 — Backtest overfitting: the search is formally refuted

False Strategy Theorem + CSCV applied to our own 129,600-config search
(`dsr_pbo_results.csv`):

```
Best config (in-sample profit factor 1.097):
  observed per-trade Sharpe                          = 0.0345
  E[max Sharpe] by LUCK ALONE across 129,600 trials  = 0.1093
  observed / expected-by-luck                        = 0.32x
  Deflated Sharpe Ratio                              = 0.0035  (bar: >0.95)
  Probability of Backtest Overfitting (CSCV)         = 0.4372
```

The best configuration found is **3× worse than what pure noise would
produce** from that many trials. This is the formal counterpart to the
earlier empirical finding (54 configs profitable in both periods vs. 255.6
expected by chance).

## Result 4 — Non-IID samples widen every error bar for n>1

| horizon | avg uniqueness | effective n (of 30,000) | CI understated by |
|---|---|---|---|
| n=1 | 0.500 | 15,000 | 1.4× |
| n=5 | 0.167 | 5,000 | 2.4× |
| n=20 | 0.048 | 1,429 | 4.6× |
| n=50 | 0.020 | 589 | 7.1× |

`n=20: AUC 0.5379 ± 0.0183` should be read closer to ±0.08. Conclusions
unchanged (still indistinguishable from random), but several apparent
differences between horizons were never statistically meaningful.

---

## Result 5 — Uniqueness weighting on the triple-barrier data: no effect

This was flagged as "the highest-value remaining item," on the reasoning
that folder 05's triple-barrier labels have *variable* spans (`bars_to_exit`
1–5) unlike folder 10's fixed horizons, so Ch.4 weighting should finally
bite. **Measuring first showed that reasoning was largely wrong**
(`uniqueness_weighted_triplebarrier.py`):

| metric | folder 05 (triple-barrier signals) | folder 10 (n=20 bar labels) |
|---|---|---|
| mean concurrency | **1.015** | 21.0 |
| mean avg uniqueness | **0.985** | 0.048 |
| effective sample size | **4,050 of 4,112 (98.5%)** | 1,429 of 30,000 (4.8%) |
| rows with uniqueness < 1 | 367 (8.9%) | ~all |

Signals fire sparsely — median 13-bar gap versus a max 5-bar span — so
labels almost never overlap. The non-IID problem that dominates bar-level
data is nearly absent here. (The purge step confirmed it independently:
**0 rows** were purged at any fold boundary.)

Five weighting schemes, everything else held identical to the folder 06
pipeline (same 33 features, hyperparameters, seeds, expanding walk-forward):

| scheme | walk-forward AUC | test AUC (2025–26) |
|---|---|---|
| A. none (control) | 0.5048 ± 0.0211 | 0.4799 |
| **B. time-decay only (production)** | **0.4906 ± 0.0209** | **0.5309** |
| C. uniqueness only | 0.5017 ± 0.0209 | 0.4942 |
| D. uniqueness × time-decay | 0.4900 ± 0.0176 | 0.5102 |
| E. return-attribution × time-decay | 0.4816 ± 0.0263 | 0.4684 |

**Every scheme is within noise of the production baseline**, and every one
sits at ~0.48–0.53 — indistinguishable from random, consistent with this
dataset's previously established 0.532 walk-forward AUC.

Two things worth stating plainly:
- The prediction registered *before* running (uniqueness would do nothing;
  return-attribution had the best chance of differing) was half right:
  C and D changed nothing, and E did differ — but it was the **worst**
  scheme, not an improvement.
- The correct order of operations here was to measure uniqueness *first*.
  Had the weighting been applied without that diagnostic, the null result
  would have looked like "AFML weighting doesn't work" rather than the
  accurate "this dataset has almost no label redundancy to correct."

## Honest bottom line

The lectures delivered **three real methodological corrections** (integer
differentiation, non-IID error bars, formal multiple-testing correction) and
**zero predictive improvement**. The ~0.57 AUC ceiling is unmoved.

That is itself informative: the ceiling is not an artifact of naive
methodology. We have now applied the field's most rigorous published
anti-overfitting toolkit to this data and the answer got *more* firmly
negative, not less. Every remaining explanation points outside the current
inputs.

## What remains unapplied, and why

- ~~Uniqueness-weighted training on the triple-barrier datasets~~ —
  **done, see Result 5. No effect**: those labels turn out to be 98.5%
  unique because signals fire far apart, so there was almost no redundancy
  for the weighting to correct.
- **True dollar/imbalance bars** — implemented, but our source is hourly
  OHLCV, not ticks. Aggregating hourly bars loses the intra-hour path, so
  this can't deliver its real benefit without trade-level data.
- **Meta-labeling (L3 slides 17–19)** — worth being precise about: the
  primary/secondary split *is* what this project's original architecture
  was (pinescript = side, ML = take/skip/reverse). Meta-labeling improves
  **precision on an existing edge**; it cannot manufacture edge in a primary
  model that has none. Given DSR 0.0035 and PBO 0.44 on the primary, this is
  not a promising direction until a primary model with demonstrable edge
  exists.
- **Entropy features (L8)** and **structural-break/explosiveness tests
  (L8, CUSUM/SADF)** — the most plausible remaining *feature* lever, though
  on the evidence of the FFD result I would not expect a large effect.
