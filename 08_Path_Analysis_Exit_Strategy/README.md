# 08 — Path Analysis / Exit Strategy Investigation

Tests a different question than "is the 5-bar terminal outcome
predictable": does model confidence correlate with *path behavior* —
do high-confidence trades keep drifting favorably over time, and do
low-confidence trades drift against the base direction? If true, this
could inform exit rules (cut losers early, let winners run) independent
of whether the terminal binary outcome itself is predictable.

## File

| File | What it is |
|---|---|
| `path_returns_by_confidence_bucket.csv` | For each test-set trade: cumulative return in the base (raw Signal) direction at bars 1 through 20, plus the model's predicted probability and confidence bucket (top 25% / bottom 25% / middle 50%) |

## Result: no, this doesn't hold up either

```
Bar    HIGH conf (top 25%)   LOW conf (bottom 25%)   MID (middle 50%)
5      -0.033%                -0.040%                 -0.149%
10     -0.096%                +0.104%                 -0.157%
20     -0.099%                +0.325%                 -0.024%
```

The gradient over longer horizons is actually backwards from the
hypothesis (low-confidence trades show the stronger favorable drift past
bar 8), and — critically — a bootstrap test on the HIGH-vs-LOW gap at
bars 5, 10, and 20 shows every confidence interval spans zero. None of
these apparent differences are statistically distinguishable from chance
given ~176 trades per extreme bucket.

## Why this is consistent with everything else found

The model's predicted probabilities are tightly clustered (std ≈ 0.035,
range 0.37–0.58) — a model with near-zero discriminative power correctly
produces path behavior that's indistinguishable across confidence buckets,
because it isn't meaningfully sorting trades in the first place. This
result would need to change if the underlying confidence signal itself
improves (see the deep learning experiments and the "new information"
recommendation in the master methodology doc).
