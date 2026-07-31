# Methodology & Findings — Full Project Narrative

This document explains every piece of methodology used across this
project, in the order the investigation actually happened, and states the
final, honest conclusion plainly.

---

## Part 1 — The HMM Regime Model

**Goal**: classify each day of BTC/USDT history into a latent market
regime using an unsupervised model that discovers regime character
statistically, rather than via hand-picked thresholds.

**Method**: Gaussian Hidden Markov Model, fit on daily bars (regimes
persist over days-to-weeks; hourly fitting would mostly capture noise).

**Features**: `feat_logret` (3-day smoothed daily log return), `feat_vol`
(14-day rolling volatility), `feat_adx` (14-period ADX, direction-agnostic
trend strength). Two features (return+vol) can't distinguish "calm
trending" from "chop" — both look identical on those axes alone. ADX was
tested against an autocorrelation-based alternative and won: it cleanly
separated states with tight per-state variance, while autocorrelation
mostly added noise and shortened state persistence.

**Choosing K**: fit K=2 through 7, each with 20 random EM restarts (EM
only finds local optima). A health check rejects any restart with a
degenerate solution (non-finite parameters, an empty state, an invalid
transition matrix) — validated on synthetic ground-truth data before being
trusted on real data. K=5 was selected by BIC, cross-checked against
economic interpretability (does each state actually describe something
distinct and persistent, not a fine slice of an existing regime).

**Live-safe decoding**: standard HMM decoding (Viterbi, `predict_proba`)
uses the entire sequence — past and future — which is lookahead bias if
used live. A hand-implemented **causal forward algorithm** is used for
anything live-facing instead. Agreement with the smoothed decode is ~98%
on validation data; disagreements cluster at regime-change boundaries,
exactly where the distinction matters.

**Deployment**: the daily regime is forward-filled onto hourly bars with a
1-day lag (a day's regime isn't known until that day's bar closes) — no
lookahead introduced in the mapping step either.

**Was regime added to the entry classifier?** Tested extensively — win/loss
AUC impact (not significant, χ²=2.30, p=0.68), loss-magnitude AUC impact
(the one thing regime IS related to — still didn't clear the bar), and a
redundancy check (Cramér's V 0.30 vs. `vol_regime` — not fully redundant,
but not enough independent signal to earn a place in the classifier). The
better-supported use is direct, rule-based position sizing (bear-regime
trades lose ~5.7× more on average), not a model feature.

---

## Part 2 — The Triple-Barrier Labeling Method

Every entry signal gets labeled by placing three exit conditions around
the entry price and checking which is hit first, bar by bar:

1. **Upper barrier** (favorable) — above entry for BUY, below for SELL
2. **Lower barrier** (stop-loss) — below entry for BUY, above for SELL
3. **Vertical barrier** (timeout) — a fixed number of bars forward

**Barrier width** (confirmed empirically to machine-epsilon precision
across every non-timeout historical row):
```
barrier_pct = max(2 × ATR_pct_at_entry, 1.0)
```

**Vertical timeout**: the actually-deployed model uses **10 bars**
(confirmed directly from `retrain_from_signals_all_signals.py` and
`model_monitor.py` source code). A 5-bar variant was tested as a research
question — see Model B — and does not resolve the live-performance gap;
if anything it makes it slightly worse.

**Direction convention**: the label uses the raw `Signal` column, never
`execution_signal` — see Bug 3 in `09_Bugs_Found_And_Fixed/README.md`.

---

## Part 3 — Probability Threshold / Bucket Methodology

```
prob >= 0.65   ->  Bucket A (TAKE)     — trade the raw signal as-is
prob <= 0.35   ->  Bucket E (REVERSE)  — trade the OPPOSITE of the raw signal
0.35 < prob < 0.65  ->  Bucket S (SKIP)  — no trade
```

A probability well below 0.5 is active evidence the opposite direction is
more likely to work, so Bucket E captures that rather than discarding it.
A threshold sweep was tested separately on live data and explicitly NOT
used to pick a new "optimal" threshold — every candidate's confidence
interval overlapped too heavily with every other candidate to trust a
re-tuned number without a genuinely fresh validation slice.

---

## Part 4 — The Investigation: Why Does Historical AUC Not Hold Up Live?

This was the central question of the project. Every hypothesis was tested
directly, not just discussed:

| Hypothesis | Result |
|---|---|
| Small-sample noise | Ruled out — grew n from 66 to 300+, the below-random scare disappeared but the real number stayed ~0.55 |
| Wrong barrier/label definition | Ruled out — exact formula reconstructed and verified; tested both 5-bar and 10-bar directly |
| Feature computation window (200-bar live vs. full-history fit) | Ruled out — differences ~0.0002, negligible |
| Train/Validation boundary shift | Ruled out — controlled comparison with identical boundaries |
| Model staleness | Ruled out — retrained with 7 months of live data folded in; barely moved the needle |
| HMM Viterbi lookahead bias | Not applicable — the classifier never consumes HMM output directly as a trained-in feature in the models that show the gap |
| Global feature scaler leakage | Not applicable — never used; CatBoost doesn't need scaling |
| In-sample regime state labeling | Not applicable — state name strings are cosmetic, never fed as a feature |
| Split-boundary label overlap | Real but negligible — 2 rows out of 4,921 |
| **Feature price-basis mismatch** (Bug 2) | **Real, confirmed** — fixed and retested directly; did not change AUC |
| **Timezone/IST mismatch** (Bug 1) | **Real, confirmed** — fixed and retested directly; did not change AUC |
| **Direction convention bug** (Bug 3) | **Real, confirmed** — fixed; corrected specific trade outcomes but didn't change the aggregate finding |
| Model capacity (is CatBoost too weak?) | Ruled out — MLP on the same features performs the same |
| Feature representation (are hand-engineered indicators hiding info?) | Ruled out — LSTM on raw OHLCV performs the same |
| Path-conditioned exit strategies | Tested — no statistically significant path divergence by model confidence |

## The actual answer

Building the **entire dataset from scratch** (Section 05) — fresh feature
computation, fresh triple-barrier labels, all three bugs fixed, a
completely independent data source — and running walk-forward validation
on **purely historical data** (2019-2024, no live data involved at all)
gives **AUC = 0.532 ± 0.022**. Barely above random.

This means the original ~0.85-0.91 historical AUC was very likely inflated
by artifacts in how that older dataset was constructed (most likely the
feature-basis bug, compounded by other subtle issues never fully isolated)
— **not** genuine predictive signal that later "decayed." There was no
2026 cliff. The model was never as strong as it looked; the illusion was
in the training data's construction, not in a changing market.

## Where the evidence points next

Three architectures (CatBoost, MLP, LSTM) and two representations (hand-
engineered features, raw OHLCV) all converge on the same ~0.49-0.54
ceiling. This is strong evidence the current inputs — price, volume,
volatility, and derived technical indicators — do not carry enough
information about this specific 5-bar outcome for any model to reliably
separate winners from losers. The path-analysis check (Section 08) also
ruled out that model confidence carries useful path information for exit
design.

**What hasn't been tried**: genuinely new information sources — order flow
/ order book imbalance, funding rate and open interest (perp positioning
data, explicitly descoped early in this project as "phase 2"), on-chain
flow, cross-asset leading signals. If none of those move the needle
either, the honest conclusion is that this specific entry trigger (StochRSI
+ ADX cross) at this specific horizon (5-10 bars) may not have exploitable
edge, and testing a different trigger/horizon/asset entirely is the more
productive next step than continuing to re-model the same inputs.

---

## Part 5 — Training Discipline (applies throughout)

- Chronological splits only, never shuffled
- Exponential recency sample weighting, 180-day half-life
- Multiple random seeds (5), averaged at inference, to distinguish real
  effects from seed-to-seed noise
- Frozen permanent holdouts, touched exactly once per model version
- No threshold or hyperparameter tuning against the holdout used for the
  final reported number
- `vol_regime` bins always recomputed from the training pool only at each
  new split, never from the full dataset or the test set
