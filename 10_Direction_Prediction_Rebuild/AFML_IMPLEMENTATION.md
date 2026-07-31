# López de Prado (AFML) Purged Cross-Validation — Implementation & Findings

Implements the solution from **"Advances in Financial Machine Learning"
(Marcos López de Prado, 2018), Chapter 7 — "Cross-Validation in Finance"**,
plus the companion sample-weighting machinery from **Chapter 4** and the
combinatorial extension from **Chapter 12**.

## Where the solution is in the book

**§7.3 — why k-fold fails in finance.** Shuffled / stratified k-fold assumes
IID observations. Financial data violates this twice:
1. **Overlapping labels** — a label at bar `t` is built from bars `[t, t+n]`,
   so labels at `t` and `t+1` share `n-1` of the same future bars. Shuffle
   them across the train/test boundary and the training set literally
   contains the test set's outcome.
2. **Serially correlated features** — adjacent bars have near-identical
   feature vectors, so shuffling seeds the training set with near-duplicates
   of every test row.

**§7.4 — the fix**, in two parts:
- **§7.4.1 Purging** (snippet 7.1, `getTrainTimes`) — drop from *training*
  every observation whose label interval `[t₀, t₁]` overlaps any test label
  interval. Three cases: train label starts inside test, ends inside test,
  or envelops it.
- **§7.4.2 Embargo** (snippet 7.2, `getEmbargoTimes`) — purging alone is
  insufficient under serial correlation, so additionally drop training
  observations in a window immediately *after* each test block. Book's rule
  of thumb: `h = 0.01 × n_obs`.
- **§7.4.3** (snippet 7.3) — both combined in **`PurgedKFold`**, a drop-in
  replacement for `KFold(shuffle=True)`.

**Chapter 4 — the companion fix.** Purging repairs *evaluation*; it does not
change the fact that overlapping labels mean far fewer independent
observations than rows. §4.3 concurrency, §4.4 average uniqueness, §4.5
sequential bootstrap, §4.6 return-attribution weights, §4.7 time decay.

**Chapter 12 §12.4 — CPCV**, which produces many backtest paths rather than
one, as a defence against backtest overfitting.

## What's implemented — `afml.py`

| Function / class | Book reference |
|---|---|
| `get_train_times` | §7.4.1, snippet 7.1 — purging |
| `get_embargo_times` | §7.4.2, snippet 7.2 — embargo |
| `PurgedKFold` | §7.4.3, snippet 7.3 — the main solution |
| `cpcv_splits` | Ch.12 §12.4 — combinatorial purged CV |
| `num_co_events` | §4.3, snippet 4.1 — concurrency |
| `get_avg_uniqueness` | §4.4, snippet 4.2 — average uniqueness |
| `seq_bootstrap` | §4.5, snippets 4.4–4.5 — sequential bootstrap |
| `sample_weights_by_return` | §4.6, snippet 4.10 |
| `time_decay_weights` | §4.7, snippet 4.11 |

These are original implementations written against the described algorithms,
not copies of the book's listings. Where the book is O(n²), a vectorized
equivalent is used and proven equal in `test_afml.py`.

### Verified, not assumed — `test_afml.py` (9/9 pass)

- vectorized concurrency ≡ the book's naive per-label accumulation loop
- average uniqueness ≡ naive per-label mean
- uniqueness = 1.0 for disjoint labels, exactly 1/k when k labels coincide
- **every PurgedKFold fold has zero label overlap** and respects the embargo
- embargo shrinks the training set monotonically
- **sequential bootstrap achieves higher uniqueness than IID bootstrap** —
  the book's central claim for §4.5

### One deliberate deviation from the book

Snippet 7.3 keeps training observations where `t1 <= t0` (test start). That
admits a training label ending *exactly on* the test block's first bar —
those two labels then share that bar. `test_purgedkfold_no_leakage` caught
this. We use strict `<` instead, so no bar is ever shared. Flagged in-code.

---

## Finding: exactly how much AUC shuffling manufactures

Identical model and data, four CV schemes, 30,000 hourly bars
(`cv_leakage_comparison.py`):

| horizon | mean concurrency | **effective n** (of 30,000) | shuffled StratifiedKFold | KFold (no shuffle) | Purged only | **Purged + 1% embargo** | **inflation** |
|---|---|---|---|---|---|---|---|
| n=1 | 2.0 | 15,000 (50.0%) | 0.5530 | 0.5550 | 0.5552 | **0.5529** | **+0.0001** |
| n=5 | 6.0 | 5,000 (16.7%) | 0.6594 | 0.5433 | 0.5396 | **0.5410** | **+0.1184** |
| n=20 | 21.0 | 1,429 (4.8%) | 0.8503 | 0.5226 | 0.5253 | **0.5232** | **+0.3271** |
| n=50 | 51.0 | 589 (2.0%) | 0.9209 | 0.4940 | 0.4887 | **0.4845** | **+0.4364** |

**The n=1 row is the control that proves the mechanism.** At a 1-bar horizon
labels barely overlap, and shuffling inflates AUC by +0.0001 — nothing. As
overlap grows, inflation grows in lockstep, reaching **+0.44 at n=50, where
shuffled CV reports AUC 0.92 on data whose honest AUC is 0.48.** A model
with no predictive power whatsoever can be made to look near-perfect purely
by shuffling overlapping labels.

### On the project's original 0.85–0.91 AUC — suggestive, NOT proven

Shuffled CV at n=20 gives **0.8503** and at n=50 gives **0.9209**, which
brackets the original models' unexplained 0.85–0.91 almost exactly, and
`00_METHODOLOGY_AND_FINDINGS.md` records that root cause as never fully
isolated.

**But I checked, and the packaged training scripts (`03_`, `04_`) use
chronological splits and do not shuffle.** With a chronological split and a
10-bar label only ~10 boundary rows overlap — the project already measured
this as "2 rows out of 4,921", far too few to move AUC from 0.53 to 0.85.
So this does **not**, on the evidence available here, explain the original
number. The numerical coincidence is real and worth chasing, but calling it
the answer would repeat exactly the pattern of premature conclusions this
project has already had to unwind twice.

**What would settle it**: `retrain_from_signals_all_signals.py`, the
original production labeling/training script referenced in
`09_Bugs_Found_And_Fixed/README.md` but not included in this package. If
that script shuffles, or does any random split, this is the mechanism. If
it doesn't, the original inflation came from somewhere else.

### The finding that changes the rest of the project's numbers

**Effective sample size.** At n=20, 30,000 rows carry only **1,429
independent observations** (4.8%); at n=50, **589** (2.0%). Every confidence
interval reported anywhere in this project for horizons > 1 is therefore
**too narrow — by roughly √(1/uniqueness), i.e. ~4.6× at n=20 and ~7× at
n=50.** The walk-forward result `n=20: AUC 0.5379 ± 0.0183` should be read
closer to ±0.08. That does not change the conclusion (still indistinguishable
from random), but it means several earlier "differences" between horizons
were never statistically meaningful to begin with.

### Was this project's own earlier work leaky? No.

The existing `train_walkforward.py` is **strictly forward-looking** — it
only ever trains on data before the test fold, with an explicit embargo.
That is *more* conservative than `PurgedKFold`, which also uses post-test
data (purged). So the 0.53–0.57 results already reported stand unchanged;
this chapter's machinery confirms them rather than overturning them.

## Practical recommendation

For the **fixed-horizon** labels used in folder 10, uniqueness weighting
changes nothing — every label spans exactly `n` bars, so all weights are
equal. It matters for **variable-span labels**, which is precisely what the
**triple-barrier labels in folders 03–06** are: a trade exiting on a barrier
hit after 2 bars is far more unique than one running the full 10-bar
timeout. Those datasets should be trained with
`get_avg_uniqueness`-derived `sample_weight`, using each trade's actual exit
bar as `t₁`. That is the highest-value remaining application of this
chapter to the existing work.

## Files

| File | What it is |
|---|---|
| `afml.py` | The implementation — Ch.4, Ch.7, Ch.12 |
| `test_afml.py` | 9 correctness tests, all passing |
| `cv_leakage_comparison.py` | The four-scheme experiment above |
| `cv_leakage_comparison.csv` / `cv_leakage.log` | Full results |
