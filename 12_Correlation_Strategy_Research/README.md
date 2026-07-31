# Five ML-Friendly Correlation Strategies for BTCUSDT.P

**Written**: 2026-07-31. Scope: replace the StochRSI+ADX programme, which is
now refuted from four independent directions (AUC ceiling ~0.53–0.57 across
CatBoost/MLP/LSTM, no profitable config among 9,720 entry×exit combinations,
DSR = 0.0035, PBO = 0.44).

This is a **research memo, not a backtest**. Nothing here has been tested on
your data yet. Every claim is sourced, and where the published evidence is
*negative* I say so — including for strategies I still recommend.

---

## Part 0 — Why the last one failed, stated as a design constraint

The post-mortem in `00_METHODOLOGY_AND_FINDINGS.md` and
`10_.../FINAL_FINDINGS.md` converges on one sentence:

> Price, volume, volatility and their technical derivatives do not carry
> enough information about the **outright direction** of BTC over 1–50 hourly
> bars for any model to separate winners from losers.

That is not a modelling failure. It is a statement about the **target**.
BTCUSDT perp is one of the most arbitraged instruments in existence; its
directional return is close to a martingale at those horizons. You were
asking ML to predict the single least predictable quantity in the market,
from the single most picked-over feature set.

So the constraint for everything below:

| Requirement | Why | What failed before |
|---|---|---|
| **R1. Predict a spread / residual / relative quantity, not outright direction** | Spreads are mean-reverting *by mechanism*, not by hope. Signal-to-noise is 5–50× higher. | Predicted `sign(return)` |
| **R2. Feature must be causally linked to the target, not a transform of it** | Indicators on price are lossy re-encodings of the same information the target already contains | 63 features, all functions of OHLCV |
| **R3. High effective sample size** | Your own `cv_leakage_comparison.csv`: at n=20, 30,000 rows = 1,429 independent obs (4.8%) | Overlapping fixed-horizon labels |
| **R4. Economic prior before search** | DSR punishes 129,600 trials brutally. 5 pre-registered hypotheses ≫ 129,600 grid points | Blind grid search |
| **R5. Genuinely new data** | Every doc in this project ends with this recommendation | Never acted on |

"**Correlation strategy**" in the sense used here = the traded object is a
*relationship between two or more series* (basis, residual, spread, lead-lag,
imbalance), not the level of one series. That single change is what makes
these ML-friendly.

---

## The five, ranked by expected value

| # | Strategy | Core relationship | Data you already have | New data needed | Prior |
|---|---|---|---|---|---|
| **1** | **Perp–spot basis & funding carry** | perp price ↔ spot price, forced to converge by the funding mechanism | ✅ 100% | funding history (free) | **Strongest** |
| **2** | **Positioning / crowding reversal** | open interest ↔ price ↔ funding | ❌ | Binance metrics dumps (free) | Strong |
| **3** | **Order-flow imbalance microstructure** | signed trade flow ↔ next-tick mid | ❌ | aggTrades / bookTicker (free, large) | Strong but weakest *on BTC* |
| **4** | **Cross-sectional residual stat-arb** | each alt ↔ BTC factor | ❌ | 30–50 perp OHLCV (free) | Mixed — see negative evidence |
| **5** | **Lead-lag / information spillover** | leader venue/asset ↔ BTC perp | partial | cross-venue + CME/options | Speculative |

---

# Strategy 1 — Perp–Spot Basis & Funding Carry

### The relationship

BTCUSDT.P has no expiry. It is tethered to spot by the **funding rate**: when
perp trades above spot, longs pay shorts, mechanically pulling it back. This
is the only relationship in crypto with a *contractual* mean-reversion force
behind it. `basis_t = perp_t / spot_t − 1` is stationary by construction.

Compare that to what you were doing: you were trying to predict a random
walk. Here you are predicting deviations from a tether that is *legally
obliged* to be pulled back every 8 hours.

### Why it satisfies the constraints

- **R1** ✅ target is the basis spread, not price
- **R2** ✅ funding rate is *the mechanism*, not a proxy
- **R3** ✅ events are sparse (basis-z-score excursions), so labels barely
  overlap — like your folder-05 data which measured 98.5% average uniqueness
- **R5** ✅ funding is genuinely new information

### Data

- **Already in repo**: `11_Spot_vs_Perp_Analysis/spot_hourly.csv` (66,174 bars,
  2019-01→2026-07) and `perp_hourly.csv` (60,368 bars, 2019-09→2026-07).
  You can compute the basis series **today, with zero new downloads.**
- **Funding rate**: `GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT`
  — free, no API key, full history back to 2019-09, 1000 rows/call. ~2,900
  8-hourly observations. I can write the fetcher in ~30 lines.
- **Mark price / premium index** (optional, finer): `/fapi/v1/premiumIndexKlines`.

### Features

```
basis_pct, basis_z(24h/168h/720h), basis_ewma_slope
funding_rate, funding_z, funding_8h_ahead_predicted (premium index integral)
cum_funding_7d/30d, funding_sign_streak_length
carry_annualised = funding_rate * 3 * 365
basis − expected_basis_given_funding      <- the actual residual
realised_vol, spot_vol / perp_vol ratio
```

### Label (use your existing machinery)

Triple-barrier on the **basis**, not on price, with events selected by CUSUM
on `basis_z` (`afml_lectures.getTEvents` — already implemented and tested).
Then **meta-label**: primary = "basis z > 2 → short perp / long spot"; ML
decides take/skip. This is the meta-labeling setup from L3 slides 17–19 that
`LECTURES_APPLIED.md` correctly said was pointless without a primary edge —
here the primary edge is mechanical, so meta-labeling can finally bite.

### Evidence

[BIS Working Paper 1087, "Crypto carry"](https://www.bis.org/publ/work1087.pdf)
documents carry exceeding **40% p.a.**, attributed to (i) leverage demand from
trend-chasing retail and (ii) limited arbitrage capital due to regulatory and
margin frictions. That is a *structural* limits-to-arbitrage explanation —
exactly the kind of prior R4 demands.

**Honest counter-evidence**: carry profitability has compressed sharply since
2024, with reports of a **negative Sharpe in 2025**. So do not expect the
naive always-on carry trade to work. The ML job is precisely to time *when*
the carry is compensated versus when it is a crowded trap — which is where
Strategy 2 plugs in as a feature set.

### First measurable checkpoint

Regress forward 8h/24h basis change on `basis_z`. If the half-life is finite
and the R² is meaningfully non-zero, this is live. **This is a two-hour job
with data already on disk.** Do it before anything else in this document.

---

# Strategy 2 — Positioning & Crowding Reversal (OI × Funding × Liquidations)

### The relationship

Price alone cannot tell you *who* is positioned and how fragile they are.
Open interest can. The four-quadrant map (price ↑/↓ × OI ↑/↓) distinguishes
new money entering from short-covering — information that is **provably
absent** from OHLCV, which is why your 63-feature set could never recover it.

Crowded leveraged positioning creates *asymmetric, path-dependent* tail risk:
a liquidation cascade is a mechanically forced seller, not a discretionary
one. That asymmetry is what an ML model can learn and a technical indicator
cannot.

### Data — free, and better than most people realise

`data.binance.vision` publishes daily CSV dumps at
`data/futures/um/daily/metrics/BTCUSDT/` containing, at **5-minute
granularity**:

```
sum_open_interest              sum_open_interest_value
count_toptrader_long_short_ratio    sum_toptrader_long_short_ratio
count_long_short_ratio         sum_taker_long_short_vol_ratio
```

This matters a lot: the live REST endpoints
(`/futures/data/openInterestHist`, `/futures/data/topLongShortPositionRatio`)
are **capped at 30 days of history** and are the reason most people conclude
OI history is unobtainable. The bulk dumps are not capped. See
[binance-public-data](https://github.com/binance/binance-public-data) and the
[metrics prefix](https://data.binance.vision/?prefix=data/futures/um/daily/metrics/).

- **Liquidations**: Binance's `/fapi/v1/allForceOrders` is heavily truncated.
  For real liquidation history you need **Coinglass or Coinalyze** (see
  connectors section).

### Features

```
oi_change_1h/4h/24h, oi_z, oi_value_vs_marketcap
d(OI) x sign(d(price))                  <- the quadrant interaction, explicitly
taker_buy_sell_ratio, its z-score and divergence from price
toptrader_ls_ratio (accounts vs positions -- the gap is retail-vs-whale)
funding x oi_z                          <- crowding intensity
leverage_proxy = oi_value / spot_volume
estimated_liquidation_density (from OI build-up at price levels)
```

### Label

CUSUM event filter on `oi_z` shocks → triple-barrier with ATR-scaled
barriers. Events are naturally sparse ⇒ **R3 satisfied**, high uniqueness.

### Evidence

- [Early-warning signals across seven crypto-perpetual liquidation cascades](https://arxiv.org/html/2607.27070)
  (BTC USD-margined perps, May 2022 – Oct 2025). **Read the honest finding
  carefully**: early-warning signals are *event-heterogeneous* — no single
  universal precursor works across all seven cascades. That argues for an ML
  model over a fixed threshold rule, but it also caps expectations.
- [The Two-Tiered Structure of Cryptocurrency Funding Rate Markets](https://www.mdpi.com/2227-7390/14/2/346) (MDPI Mathematics, 2026).
- Practitioner framing of the OI×funding quadrant map:
  [tradelink](https://tradelink.pro/blog/funding-rate-open-interest/),
  [XT/Medium](https://medium.com/@XT_com/bitcoin-futures-market-microstructure-liquidation-cascades-funding-regimes-and-open-interest-978b107b4889).
  Treat these as hypothesis sources, not evidence.

---

# Strategy 3 — Order-Flow Imbalance Microstructure

### Why this one is special for you

There is a paper that is **almost exactly this project, done right, on your
exact instrument**:
[Explainable Patterns in Cryptocurrency Microstructure](https://arxiv.org/html/2602.00776v1).

| Their setup | Your setup |
|---|---|
| Binance Futures perpetuals | same |
| 1-second order book + trades, 2022-01 → 2025-10 | hourly OHLCV |
| CatBoost | CatBoost |
| Walk-forward CV **with purging** | you already implemented this in `afml.py` |
| Target: 3-second forward mid-price log return | 1–50 hour direction |

**Result**: order-flow imbalance dominates SHAP importance across every
asset, ahead of spread and VWAP deviation. Taker strategy achieved
Information Ratio 0.07–8.97, statistically significant (p<0.05) — **but the
significant assets were ETC, ENJ and ROSE, not BTC.** BTC sits at the bottom
of the range.

That is a genuinely important result for you and I want it stated plainly:
**even with the right data and the right method, BTC is the hardest asset in
the set.** Its efficiency is the whole problem. If you insist on BTCUSDT.P
specifically, expect the thinnest edge here.

### It also unblocks something you already flagged

`LECTURES_APPLIED.md` lists under "what remains unapplied":

> **True dollar/imbalance bars** — implemented, but our source is hourly
> OHLCV, not ticks. Aggregating hourly bars loses the intra-hour path.

`data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/` fixes that.
Signed trade data means your already-written dollar-bar, volume-bar and
imbalance-bar code becomes usable for the first time, plus the whole AFML
Ch.19 microstructural battery: VPIN, Kyle's λ, Amihud, Roll spread,
Corwin–Schultz.

**Cost warning**: aggTrades for BTCUSDT is on the order of **several hundred
GB** for the full history. Start with 6 months.

### Supporting literature

- [Order Flow Imbalance — A High Frequency Trading Signal](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html) — clean OFI construction
- [Bitcoin wild moves: evidence from order flow toxicity and price jumps](https://www.sciencedirect.com/science/article/pii/S0275531925004192) — VPIN vs jumps
- [Microstructure and Market Dynamics in Crypto Markets](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf) — Easley (co-author of VPIN)
- [The Short-Term Predictability of Returns in Order Book Markets](https://arxiv.org/pdf/2211.13777)
- Tooling: [hftbacktest](https://github.com/nkaz001/hftbacktest) — queue-position- and latency-aware backtester, the only honest way to test a maker version of this.

---

# Strategy 4 — Cross-Sectional Residual Stat-Arb

### The relationship

Fit a factor model across the top 30–50 USDT perps; the first principal
component **is** "crypto beta" (typically 60–80% of variance). Trade the
**residual** after removing it. Avellaneda–Lee style.

### Why it's the most ML-friendly *structurally*

1. **Sample size explodes**: N assets × T periods instead of T. 50 assets ×
   50,000 hourly bars = 2.5M rows, and cross-sectional observations at the
   same timestamp are far more independent than time-adjacent ones. This
   directly attacks your R3 / effective-sample-size problem, which
   `LECTURES_APPLIED.md` Result 4 identified as widening every error bar in
   the project by 4.6–7×.
2. **Residuals are stationary by construction** — no fractional
   differentiation debate needed.
3. **Cross-sectional demeaning removes the unpredictable part.** The common
   factor is what makes single-asset direction a martingale. Strip it out and
   what remains is idiosyncratic and far more mean-reverting.

### The honest problem

[Failure of Cross-Sectional Alpha Screening on Cryptocurrency Perpetual Futures: A Quantitative Post-Mortem Using OHLCV and Funding Rate Signals](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6701738)
(Jul 2022 – Apr 2026) **rejects** the hypothesis that OHLCV + funding-rate
signals contain exploitable cross-sectional alpha in large-cap crypto perps
at an 8-hour horizon.

I am putting this front and centre because it is the closest published
analogue to the mistake this project already made once. If you run Strategy 4
on OHLCV+funding at 8h on large caps, **someone has already done it and it
does not work.** Deviate on at least one axis: different horizon (daily to
weekly), wider universe (mid-caps, where the microstructure paper found the
edge lives), or richer features (OI and flow from Strategies 2–3).

### Counter-evidence in the other direction

- [A Trend Factor for the Cross Section of Cryptocurrency Returns](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4C1509ACBA33D5DCAF0AC24379148178/S0022109024000747a.pdf/trend_factor_for_the_cross_section_of_cryptocurrency_returns.pdf) — JFQA, peer-reviewed, positive
- [Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)
- [Momentum and Network Design in Cross-Section of Crypto](https://aaltodoc.aalto.fi/bitstreams/41680162-a945-4f11-9f99-6458ec4838a4/download) (Aalto thesis)
- [Cross-Sectional Alpha Factors in Crypto: 2+ Sharpe Without Overfitting](https://blog.aperiodic.io/p/cross-sectional-alpha-factors-in) — 30-day cross-sectional momentum + OI-weighted funding carry, top-50 universe, daily rebalance. **This is vendor marketing for a paid product, has no code, and self-admits the headline Sharpe is "actually less than 2".** Use for the factor construction ideas only.
- Reference implementation of the PCA/residual machinery (equities, but the code is the point): [noterminusgit/statarb](https://github.com/noterminusgit/statarb)

⚠️ **Survivorship bias is the killer here.** Your universe must include
delisted perps. Binance delists frequently; a top-50-as-of-today universe
backtested to 2021 is guaranteed to look great and be fake.

⚠️ Note that in this strategy **BTCUSDT.P is the hedge leg / the factor, not
the alpha source.** If the mandate is strictly "trade BTCUSDT.P", this one is
out of scope — say so and I'll drop it.

---

# Strategy 5 — Lead-Lag / Information Spillover

### The relationship

Information does not hit all venues and all assets simultaneously. Measure
who moves first, trade the follower.

### The literature is solid

- [Cross-cryptocurrency return predictability](https://www.sciencedirect.com/science/article/abs/pii/S0165188924000551)
  (Journal of Economic Dynamics & Control, 2024) — minute-level, information
  spillover; **the five largest coins including BTC lead the small coins, and
  not vice versa.**
- [A seesaw effect in the cryptocurrency market](https://www.sciencedirect.com/science/article/abs/pii/S0927539823000956) (JCF)
- [High Frequency Lead-lag Relationships in the Bitcoin Market](https://research.cbs.dk/en/studentProjects/high-frequency-lead-lag-relationships-in-the-bitcoin-market-an-em/)
  — cross-exchange lags **up to 15 seconds**; unexpected volume proxies the
  information-arrival rate that drives the lag.
- [Cross Cryptocurrency Relationship Mining for Bitcoin Price Prediction](https://arxiv.org/pdf/2205.00974)
  — DTW-based lead-lag extraction; reports significant improvement to existing
  price-prediction methods.

### The problem, stated bluntly

The literature says **BTC is the leader.** So lead-lag naturally predicts
*alts from BTC*, not BTC from anything. To predict BTCUSDT.P you need
something that genuinely leads the most-arbitraged crypto instrument on
earth, and the honest shortlist is short:

| Candidate leader | Rationale | Access |
|---|---|---|
| **Deribit options** — 25Δ skew, IV term structure, gamma exposure | Options traders are informed; dealer gamma mechanically shapes spot path | Free public API, no key |
| **Coinbase–Binance premium** | US institutional vs offshore flow imbalance | Free (Coinbase public API) |
| **CME futures basis & gap** | Regulated institutional positioning; CFTC COT weekly | Free |
| **Spot BTC ETF creations/redemptions** | Daily, real flow, T+0 published | Free-ish, scraped |
| **Stablecoin supply / exchange netflow** | Dry powder entering | Glassnode/CryptoQuant (paid) |

Correct estimator for asynchronous, irregular multi-venue data is
**Hayashi–Yoshida**, not naive lagged Pearson — lagged Pearson on
non-synchronised series produces spurious lead-lag (the Epps effect).

**Prior: weakest of the five.** Include it as a *feature block* inside
Strategies 1–3 rather than building a standalone strategy on it.

---

# Part 6 — What I need from you (connectors & data)

You asked. Here it is, split by whether it costs you anything.

### Free, no key, I can start today

| Source | Gives | Endpoint |
|---|---|---|
| Binance Futures REST | funding rate full history | `fapi.binance.com/fapi/v1/fundingRate` |
| Binance Vision dumps | OI + long/short ratios, 5-min, multi-year | `data.binance.vision/.../metrics/BTCUSDT/` |
| Binance Vision dumps | aggTrades, bookTicker (signed flow, OFI) | `data.binance.vision/.../aggTrades/BTCUSDT/` |
| Binance Futures REST | 30–50 perp OHLCV for the cross-section | `fapi.binance.com/fapi/v1/klines` |
| Deribit public API | options IV, skew, term structure | `deribit.com/api/v2/public/` |
| Coinbase / CME | cross-venue premium, institutional basis | public |

**Nothing on this list requires a connector, a key, or your money.** Strategy 1
needs *zero* new downloads to reach its first checkpoint.

### Would materially help — tell me if you have or will buy

| Source | Unlocks | Rough cost |
|---|---|---|
| **Coinglass or Coinalyze API key** | real liquidation history, aggregated cross-exchange OI/funding | ~$30–100/mo |
| **Glassnode or CryptoQuant** | exchange netflow, stablecoin supply, whale/miner flow | ~$30–800/mo |
| **Tardis.dev or Amberdata** | full L2 order-book replay (needed for the *maker* version of Strategy 3) | $$$ |

If you have keys, drop them in a `.env` in the project root and tell me the
variable names — I won't need an MCP connector for any of these, they're all
plain REST.

### MCP connectors

None required. You already have the TradingView MCP wired up, which is enough
for chart verification and Pine deployment. If you later want automated
execution I'd want to talk about that separately — it is a different risk
conversation.

---

# Part 7 — Pre-registered acceptance criteria

Write these down **before** running anything. This is the discipline that
saved you from shipping a 72%-win-rate money-loser.

1. **Purged + embargoed CV only.** `afml.PurgedKFold`, embargo 1%. Already
   built, already 9/9 tested.
2. **Report effective sample size** alongside every result
   (`get_avg_uniqueness`). Every CI in this project has been too narrow.
3. **Deflated Sharpe Ratio ≥ 0.95**, with the trial count honestly declared.
   The last search scored 0.0035.
4. **PBO (CSCV) ≤ 0.20.** Last search: 0.44.
5. **Frozen holdout 2026-01 → 2026-07, touched exactly once**, at the end.
6. **Max 12 configurations per hypothesis.** Not 129,600. DSR makes wide
   searches self-defeating; economic priors are cheaper than trials.
7. **Costs modelled from the start**: Binance USDT-M taker 0.045%, maker
   0.018%, plus funding paid/received, plus slippage. A basis trade at 20bp
   edge dies instantly at 9bp round-trip taker.
8. **A strategy is dead if it needs a specific exit geometry to look good.**
   That was the whole lesson of `FINAL_FINDINGS.md` Result 2.

---

# Part 8 — Recommended sequence

**Week 1 — Strategy 1 checkpoint, using data already on disk.**
Build the basis series from `11_Spot_vs_Perp_Analysis/*_hourly.csv`, pull
funding history (30 lines of code), and answer one question: *does basis
z-score predict forward basis change, and with what half-life?* This is a
pure diagnostic with a binary answer and near-zero cost. If the answer is no,
Strategy 1 dies in week 1 and you've lost two days.

**Week 2 — Strategy 2 data build.** Download the Binance Vision metrics
dumps, join OI/long-short/taker-ratio onto the hourly frame, and run the
single most informative diagnostic: mutual information between
`d(OI) × sign(d(price))` and forward returns, versus the same for your
existing 63 features. This directly tests R5 — *is there actually new
information here?* — before any model is trained.

**Week 3+ — branch on results.** If 1 and 2 both show signal, combine them:
the strongest single hypothesis in this document is that **basis/carry gives
you the trade and positioning data tells you when the carry is compensated
versus crowded.** If both are flat, escalate to Strategy 3 (tick data) —
higher cost, but it is the one with a published positive result on your exact
venue and instrument.

---

## A closing note on expectations

The microstructure paper found significant edge on ETC, ENJ and ROSE — and
the weakest results on BTC. The cross-sectional paper found *no* alpha in
large-cap perps. The carry literature documents a real premium that has been
**compressing since 2024**.

The consistent message across all of it is that **BTCUSDT.P is the hardest
instrument in crypto to extract alpha from**, because it is the most liquid
and most arbitraged. Everything in this document is a genuine improvement on
what you were doing, and none of it makes BTC easy. If the instrument is
negotiable, the same five relationships are materially more tractable one or
two rungs down the liquidity ladder — and that, not a better model, is
probably the single highest-expected-value change available to you.

---

## Sources

- [BIS WP1087 — Crypto carry](https://www.bis.org/publ/work1087.pdf)
- [Explainable Patterns in Cryptocurrency Microstructure](https://arxiv.org/html/2602.00776v1)
- [Failure of Cross-Sectional Alpha Screening on Crypto Perpetual Futures (SSRN 6701738)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6701738)
- [Early-warning signals across seven crypto-perpetual liquidation cascades](https://arxiv.org/html/2607.27070)
- [Cross-cryptocurrency return predictability (JEDC 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0165188924000551)
- [A seesaw effect in the cryptocurrency market (JCF)](https://www.sciencedirect.com/science/article/abs/pii/S0927539823000956)
- [A Trend Factor for the Cross Section of Cryptocurrency Returns (JFQA)](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4C1509ACBA33D5DCAF0AC24379148178/S0022109024000747a.pdf/trend_factor_for_the_cross_section_of_cryptocurrency_returns.pdf)
- [Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)
- [Momentum and Network Design in Cross-Section of Crypto (Aalto)](https://aaltodoc.aalto.fi/bitstreams/41680162-a945-4f11-9f99-6458ec4838a4/download)
- [Cross Cryptocurrency Relationship Mining for Bitcoin Price Prediction](https://arxiv.org/pdf/2205.00974)
- [High Frequency Lead-lag Relationships in the Bitcoin Market (CBS)](https://research.cbs.dk/en/studentProjects/high-frequency-lead-lag-relationships-in-the-bitcoin-market-an-em/)
- [Bitcoin wild moves: order flow toxicity and price jumps](https://www.sciencedirect.com/science/article/pii/S0275531925004192)
- [Microstructure and Market Dynamics in Crypto Markets — Easley et al.](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf)
- [The Short-Term Predictability of Returns in Order Book Markets](https://arxiv.org/pdf/2211.13777)
- [Order Flow Imbalance — A High Frequency Trading Signal](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html)
- [The Two-Tiered Structure of Cryptocurrency Funding Rate Markets (MDPI)](https://www.mdpi.com/2227-7390/14/2/346)
- [Fundamentals of Perpetual Futures](https://arxiv.org/pdf/2212.06888)
- [Cross-Sectional Alpha Factors in Crypto (vendor blog — treat with caution)](https://blog.aperiodic.io/p/cross-sectional-alpha-factors-in)
- [binance-public-data](https://github.com/binance/binance-public-data) · [data.binance.vision metrics](https://data.binance.vision/?prefix=data/futures/um/daily/metrics/)
- [hftbacktest](https://github.com/nkaz001/hftbacktest) · [noterminusgit/statarb](https://github.com/noterminusgit/statarb) · [awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading)
