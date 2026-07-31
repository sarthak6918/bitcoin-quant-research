# Strategy Optimization @ 0.5 Trades/Day — Final Findings

**Task chosen**: maximize win rate on the base pinescript strategy at ~0.5
trades/day. (Of the three options offered, "build an accurate model" was
already a wall hit three independent ways, and "maximum correlation with the
features" was already proven structurally unreachable by tuning this rule
family — see `STRATEGY_INPUT_SEARCH_FINDINGS.md`.)

Everything below is measured with a **faithful bar-by-bar simulation of the
actual pinescript** (`strategy_sim.py`) — real entries, real close-based
3%/1%/opposite-signal exits, real 0.03%/side commission — not a proxy label.

- **Search period**: 2017-08 → 2024-12 (all tuning happens here)
- **Validation period**: 2025-01 → 2026-07 (touched once, at the end)

---

## Result 1 — The requested deliverable: win rate WAS maximized, and it holds

| | search | validation |
|---|---|---|
| **Original pinescript** | 54.9% win, PF 0.849 | 56.1% win, PF 0.860 |
| **Optimized (max win rate)** | **74.3%** win, 0.50 trades/day | **72.0%** win, 0.43 trades/day |

Win rate genuinely generalizes: across all 28,701 qualifying configs, the
rank correlation between search-period and validation-period win rate is
**ρ = +0.87 (p ≈ 0)**. This is not a fluke — a config that wins often
historically keeps winning often.

Delivered as `optimized_strategy.pine`.

## Result 2 — …and it is worthless, because it still loses money

The 72%-win-rate config has **profit factor 0.85 and negative total PnL in
both periods**. It reaches 72% the trivial way: a 1% take-profit with **no
stop loss**. Many tiny wins, unbounded losers.

Win rate is mechanically controlled by exit geometry, not by edge:

| fixed stop | mean win rate | mean profit factor |
|---|---|---|
| 2% | 48.9% | 0.877 |
| 3% | 54.0% | 0.860 |
| 5% | 58.4% | 0.883 |

Widen the stop → win rate climbs 10 points → **profit factor doesn't move
at all**. Win rate and profitability are nearly decoupled here
(ρ = 0.13). This is why optimizing win rate in isolation is a trap, and why
this document leads with the caveat rather than the headline.

## Result 3 — The decisive test: NO configuration has persistent edge

Profitability, unlike win rate, does **not** generalize — it anti-correlates:

| relationship (search → validation) | Spearman ρ | verdict |
|---|---|---|
| win rate | **+0.87** | generalizes strongly |
| profit factor | **−0.22** | does not generalize |
| total PnL | **−0.17** | does not generalize |

Configs that made money in the search period were *less* likely to make
money in validation.

To rule out "wrong exit structure" as the cause, a second sweep tested the
best entries against **9,720 entry × exit combinations** — including
disabling the fixed stop, disabling trailing entirely, and adding fixed
take-profits from 1% to 8%:

```
profitable in search     : 32.85%
profitable in validation :  8.00%
profitable in BOTH       : 54 configs
expected by chance       : 255.6 configs
                           -> observed is 0.21x chance (5x WORSE than random)
spearman(search PF, validation PF) = -0.028
```

Finding fewer both-period winners than chance would produce is conclusive:
there is no persistent profitable configuration in this strategy family.
Mean validation profit factor is ~0.82 whether or not a config was
profitable in search — knowing the historical result tells you nothing.

**An entry with genuine edge should be profitable under *some* sane exit
rule. This one isn't profitable under any of 9,720.** The problem is the
entry trigger, not the risk management — no amount of parameter tuning
fixes it.

---

## Honest bottom line

I delivered the literal ask: a **72% out-of-sample win rate at 0.43
trades/day**, validated on data never used for tuning, and it reproduces.
I am telling you plainly that it loses money, because handing you a 72%
win-rate number without that context would be the single most misleading
thing I could do in this project.

Consistent with everything else found here (~0.53–0.57 AUC ceiling across
three model architectures and two label formulations), the StochRSI+ADX
crossover on hourly BTC/USDT does not appear to carry exploitable edge.
The evidence now covers the entry trigger's parameters, its exit structure,
and ML models built on the same data — all converging.

**What I'd genuinely suggest next**, in order of expected value:
1. **A different data source** — order flow / order-book imbalance, funding
   rate + open interest. Repeatedly the strongest untested lever.
2. **A different instrument** — BTC/USDT spot is among the most arbitraged
   markets there is; the same rule family may survive on a less efficient one.
3. **A different horizon** — everything here is 1–50 hourly bars. Multi-day
   to weekly horizons are where macro/trend factors dominate noise.
4. **A different trigger entirely** — mean-reversion on a range-bound pair,
   or a cross-sectional/relative-value setup rather than a single-asset
   directional one.

## Files

| File | What it is |
|---|---|
| `strategy_sim.py` | Numba-compiled faithful simulator of the pinescript (`simulate`) + generalized-exit variant (`simulate_exits`) |
| `optimize_winrate.py` | 129,600-config grid search on the search period |
| `validate_winrate.py` | Out-of-sample evaluation of all 28,701 qualifying configs |
| `analyze_validation.py` | Generalization tests, decay analysis, chance baselines |
| `exit_structure_test.py` | 9,720 entry × exit combinations — the "is it the entry or the exit?" test |
| `optimized_strategy.pine` | The max-win-rate parameterization, with warnings, ready to paste into TradingView |
| `winrate_search_results.csv` / `winrate_validation_results.csv` / `exit_structure_results.csv` | Full raw results |
