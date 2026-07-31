"""
Faithful bar-by-bar simulator of the uploaded pinescript strategy, so that
"win rate" means the REAL thing the strategy would have realized -- actual
entries, actual 3%-fixed / 1%-trailing / opposite-signal exits, actual
commission -- not a proxy label like a triple barrier.

Faithfulness notes (matched against pinescript.txt):
  - process_orders_on_close=true -> all fills at the bar CLOSE
  - Stops are CLOSE-based (`close <= longFixedSL`), NOT intrabar-touch.
    This is exactly what the pine code does and it conveniently removes all
    intrabar path ambiguity, so the sim needs no high/low fill assumptions.
  - Peak/trough tracking and trailing ACTIVATION use the bar's high/low
    (`buyPeak := max(buyPeak, high)`, `high >= entry*(1+gainThreshold)`)
  - Opposite signal closes the position, and (per the entry block firing on
    every signal bar) immediately reverses into the opposite side
  - commission 0.03% per side, per `commission_value=0.03`

Order of operations per bar, matching the pine script's semantics:
  1. if in position: update peak/trough + trailing-active flag from this bar
  2. if in position: evaluate exits against this bar's close
  3. if flat (incl. just-exited): enter on this bar's signal at close
"""
import numpy as np
from numba import njit

COMMISSION = 0.0003  # 0.03% per side


@njit(cache=True)
def simulate(close, high, low, buy_sig, sell_sig,
             fixed_sl_pct, trail_sl_pct, gain_thresh_pct):
    """
    Returns (trade_pnls, trade_dirs, entry_idx, exit_idx) as arrays.
    pnl is fractional return net of round-trip commission.
    """
    n = len(close)
    max_trades = n
    pnls = np.zeros(max_trades)
    dirs = np.zeros(max_trades, dtype=np.int64)
    entry_i = np.zeros(max_trades, dtype=np.int64)
    exit_i = np.zeros(max_trades, dtype=np.int64)
    n_trades = 0

    pos = 0            # 0 flat, 1 long, -1 short
    entry_price = 0.0
    peak = 0.0
    trough = 0.0
    trailing_active = False
    entry_bar = 0

    fixed_sl = fixed_sl_pct / 100.0
    trail_sl = trail_sl_pct / 100.0
    gain_th = gain_thresh_pct / 100.0

    for i in range(n):
        # ---- 1. update excursion tracking on the open position ----
        if pos == 1:
            if high[i] > peak:
                peak = high[i]
            if (not trailing_active) and high[i] >= entry_price * (1.0 + gain_th):
                trailing_active = True
        elif pos == -1:
            if low[i] < trough:
                trough = low[i]
            if (not trailing_active) and low[i] <= entry_price * (1.0 - gain_th):
                trailing_active = True

        # ---- 2. exits, evaluated on this bar's close ----
        if pos == 1:
            exit_now = False
            if sell_sig[i]:
                exit_now = True
            elif close[i] <= entry_price * (1.0 - fixed_sl):
                exit_now = True
            elif trailing_active and close[i] <= peak * (1.0 - trail_sl):
                exit_now = True
            if exit_now:
                gross = (close[i] - entry_price) / entry_price
                pnls[n_trades] = gross - 2 * COMMISSION
                dirs[n_trades] = 1
                entry_i[n_trades] = entry_bar
                exit_i[n_trades] = i
                n_trades += 1
                pos = 0
        elif pos == -1:
            exit_now = False
            if buy_sig[i]:
                exit_now = True
            elif close[i] >= entry_price * (1.0 + fixed_sl):
                exit_now = True
            elif trailing_active and close[i] >= trough * (1.0 + trail_sl):
                exit_now = True
            if exit_now:
                gross = (entry_price - close[i]) / entry_price
                pnls[n_trades] = gross - 2 * COMMISSION
                dirs[n_trades] = -1
                entry_i[n_trades] = entry_bar
                exit_i[n_trades] = i
                n_trades += 1
                pos = 0

        # ---- 3. entries (only when flat, incl. immediately after a reversal exit) ----
        if pos == 0:
            if buy_sig[i]:
                pos = 1
                entry_price = close[i]
                peak = high[i]
                trailing_active = False
                entry_bar = i
            elif sell_sig[i]:
                pos = -1
                entry_price = close[i]
                trough = low[i]
                trailing_active = False
                entry_bar = i

    return pnls[:n_trades], dirs[:n_trades], entry_i[:n_trades], exit_i[:n_trades]


@njit(cache=True)
def simulate_exits(close, high, low, buy_sig, sell_sig,
                   fixed_sl_pct, trail_sl_pct, gain_thresh_pct, tp_pct):
    """
    Same entry logic as simulate(), but with a generalized EXIT structure so
    we can test whether the strategy's losses come from the entry trigger or
    from the exit design:
      fixed_sl_pct : >=1000 disables the fixed stop
      trail_sl_pct : <=0    disables trailing entirely
      tp_pct       : <=0    disables the take-profit
    All stops remain CLOSE-based, matching the pinescript.
    """
    n = len(close)
    pnls = np.zeros(n)
    n_trades = 0
    pos = 0
    entry_price = 0.0
    peak = 0.0
    trough = 0.0
    trailing_active = False

    fixed_sl = fixed_sl_pct / 100.0
    trail_sl = trail_sl_pct / 100.0
    gain_th = gain_thresh_pct / 100.0
    tp = tp_pct / 100.0

    for i in range(n):
        if pos == 1:
            if high[i] > peak:
                peak = high[i]
            if (not trailing_active) and high[i] >= entry_price * (1.0 + gain_th):
                trailing_active = True
        elif pos == -1:
            if low[i] < trough:
                trough = low[i]
            if (not trailing_active) and low[i] <= entry_price * (1.0 - gain_th):
                trailing_active = True

        if pos == 1:
            exit_now = False
            if sell_sig[i]:
                exit_now = True
            elif fixed_sl_pct < 1000.0 and close[i] <= entry_price * (1.0 - fixed_sl):
                exit_now = True
            elif tp_pct > 0.0 and close[i] >= entry_price * (1.0 + tp):
                exit_now = True
            elif trail_sl_pct > 0.0 and trailing_active and close[i] <= peak * (1.0 - trail_sl):
                exit_now = True
            if exit_now:
                pnls[n_trades] = (close[i] - entry_price) / entry_price - 2 * COMMISSION
                n_trades += 1
                pos = 0
        elif pos == -1:
            exit_now = False
            if buy_sig[i]:
                exit_now = True
            elif fixed_sl_pct < 1000.0 and close[i] >= entry_price * (1.0 + fixed_sl):
                exit_now = True
            elif tp_pct > 0.0 and close[i] <= entry_price * (1.0 - tp):
                exit_now = True
            elif trail_sl_pct > 0.0 and trailing_active and close[i] >= trough * (1.0 + trail_sl):
                exit_now = True
            if exit_now:
                pnls[n_trades] = (entry_price - close[i]) / entry_price - 2 * COMMISSION
                n_trades += 1
                pos = 0

        if pos == 0:
            if buy_sig[i]:
                pos = 1
                entry_price = close[i]
                peak = high[i]
                trailing_active = False
            elif sell_sig[i]:
                pos = -1
                entry_price = close[i]
                trough = low[i]
                trailing_active = False

    return pnls[:n_trades]


def summarize(pnls, n_days):
    if len(pnls) == 0:
        return dict(n_trades=0, win_rate=np.nan, trades_per_day=0.0,
                    mean_pnl=np.nan, total_pnl=np.nan, profit_factor=np.nan,
                    expectancy=np.nan)
    wins = pnls > 0
    gross_win = pnls[wins].sum()
    gross_loss = -pnls[~wins].sum()
    return dict(
        n_trades=len(pnls),
        win_rate=float(wins.mean()),
        trades_per_day=len(pnls) / n_days if n_days > 0 else np.nan,
        mean_pnl=float(pnls.mean()),
        total_pnl=float(pnls.sum()),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        expectancy=float(pnls.mean()),
    )
