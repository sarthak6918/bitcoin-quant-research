"""
Marcos Lopez de Prado, "Advances in Financial Machine Learning" (2018)
-- implementation of the cross-validation leakage fixes.

These are original implementations written against the algorithms described
in the book (chapter/section references given per function), not copies of
the book's code listings. Where the book's approach is O(n^2) and would be
impractical on 78k hourly bars, a vectorized equivalent is used and the
equivalence is noted and unit-tested in test_afml.py.

--------------------------------------------------------------------------
THE PROBLEM (AFML Ch.7, "Cross-Validation in Finance", sec 7.2-7.3)
--------------------------------------------------------------------------
Standard k-fold CV -- especially with shuffling / stratified shuffling --
assumes observations are IID. Financial data breaks that assumption twice:

  1. OVERLAPPING LABELS. A label for a bar at time t is derived from bars
     [t, t+n]. The label at t and the label at t+1 share n-1 of the same
     future bars. If t lands in train and t+1 in test, the training set
     literally contains the test set's outcome.

  2. SERIAL CORRELATION of features. Adjacent bars have near-identical
     feature vectors, so a shuffled split puts near-duplicates of every
     test row into the training set.

Shuffling maximizes both problems by scattering test rows uniformly among
train rows. The result is inflated, unreproducible CV scores -- exactly the
"great backtest, dead live" failure mode.

--------------------------------------------------------------------------
THE SOLUTION (AFML sec 7.4)
--------------------------------------------------------------------------
  PURGING  (sec 7.4.1): remove from the TRAINING set every observation
           whose label interval [t0, t1] overlaps any test label interval.

  EMBARGO  (sec 7.4.2): additionally drop training observations occurring
           in a short window immediately AFTER the test set. Purging alone
           does not fully neutralize leakage when features are built from
           serially-correlated data, because a train observation starting
           just after the test set ends is still informationally entangled
           with it. Book suggests h = 0.01 * n_obs (1%) as a rule of thumb.

  Both are combined in the PurgedKFold class (sec 7.4.3).

--------------------------------------------------------------------------
THE COMPANION FIX (AFML Ch.4, "Sample Weights")
--------------------------------------------------------------------------
Purging fixes the *evaluation*. It does not fix the fact that overlapping
labels mean the training set has far fewer independent observations than
rows. Ch.4 addresses that with:
  - concurrency / number of co-events (sec 4.3)
  - average uniqueness per label (sec 4.4)
  - sequential bootstrap (sec 4.5)
  - sample weights by return attribution (sec 4.6) and time decay (sec 4.7)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ==========================================================================
# Ch.7 sec 7.4.1 -- PURGING
# ==========================================================================
def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """
    Purge the training set (AFML snippet 7.1).

    t1:         Series indexed by each observation's START time (t0), whose
                VALUE is that observation's label END time (t1).
    test_times: same format, but describing the test set's label intervals.

    Returns the subset of `t1` safe to train on -- i.e. every observation
    whose label interval does NOT overlap any test label interval.

    Three overlap cases are removed, per the book:
      (a) train label STARTS inside a test interval
      (b) train label ENDS inside a test interval
      (c) train label ENVELOPS a test interval
    """
    trn = t1.copy(deep=True)
    for test_start, test_end in test_times.items():
        starts_within = trn.index[(test_start <= trn.index) & (trn.index <= test_end)]
        ends_within = trn.index[(test_start <= trn.values) & (trn.values <= test_end)]
        envelops = trn.index[(trn.index <= test_start) & (test_end <= trn.values)]
        trn = trn.drop(starts_within.union(ends_within).union(envelops))
    return trn


# ==========================================================================
# Ch.7 sec 7.4.2 -- EMBARGO
# ==========================================================================
def get_embargo_times(bar_times: pd.Index, pct_embargo: float) -> pd.Series:
    """
    Build the embargo lookup (AFML snippet 7.2).

    Returns a Series mapping each bar time -> the time up to which training
    observations must be excluded if the test set ends at that bar. With
    pct_embargo = 0.01, the embargo spans the next 1% of the sample.
    """
    step = int(len(bar_times) * pct_embargo)
    if step == 0:
        return pd.Series(bar_times, index=bar_times)
    emb = pd.Series(bar_times[step:], index=bar_times[:-step])
    # tail bars embargo through the end of the sample
    emb = pd.concat([emb, pd.Series(bar_times[-1], index=bar_times[-step:])])
    return emb


# ==========================================================================
# Ch.7 sec 7.4.3 -- PurgedKFold
# ==========================================================================
class PurgedKFold(KFold):
    """
    K-fold CV where test folds are CONTIGUOUS blocks (never shuffled), and
    the training set is purged of label-overlap plus embargoed after each
    test block. AFML snippet 7.3.

    This is the drop-in replacement for KFold(shuffle=True) that removes the
    leakage described at the top of this module.

    t1: Series indexed by observation start time, values = label end time.
        Must be aligned 1:1 with X and sorted ascending.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 pct_embargo: float = 0.0):
        if t1 is not None and not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series")
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if self.t1 is None:
            raise ValueError("PurgedKFold requires t1")
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError("X and t1 must have the same length")
        if not self.t1.index.equals(pd.Index(X.index)):
            raise ValueError("X and t1 must share the same index")

        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo)
        # contiguous test blocks
        bounds = [(b[0], b[-1] + 1) for b in np.array_split(indices, self.n_splits)]

        t0_values = self.t1.index.values
        t1_values = self.t1.values

        for start, end in bounds:
            test_indices = indices[start:end]
            test_start_time = t0_values[start]
            # last bar touched by ANY test label -- the test block's true reach
            max_test_t1 = t1_values[test_indices].max()

            # --- purge BEFORE the test block ---
            # Keep only observations whose label ended STRICTLY before the test
            # block starts. NOTE: the book's snippet 7.3 uses `t1 <= t0`, which
            # still admits a training label ending exactly ON the test's first
            # bar -- those two labels then share that bar. We use `<` so no bar
            # is ever shared. Verified by test_purgedkfold_no_leakage.
            left_mask = t1_values < test_start_time

            # --- purge + embargo AFTER the test block ---
            right_start = self.t1.index.searchsorted(max_test_t1, side="right") + embargo
            right_mask = indices >= right_start

            train_indices = indices[left_mask | right_mask]
            yield train_indices, test_indices


# ==========================================================================
# Ch.12 -- Combinatorial Purged Cross-Validation (CPCV)
# ==========================================================================
def cpcv_splits(n_obs: int, t1: pd.Series, n_groups: int = 6, n_test_groups: int = 2,
                pct_embargo: float = 0.01):
    """
    Combinatorial Purged CV (AFML Ch.12, sec 12.4).

    Splits the sample into `n_groups` contiguous groups and tests on every
    combination of `n_test_groups` of them, purging+embargoing each time.
    This yields C(n_groups, n_test_groups) train/test splits instead of k,
    producing many backtest paths rather than a single one -- the book's
    remedy for backtest overfitting from a single CV path.

    Yields (train_indices, test_indices).
    """
    from itertools import combinations

    indices = np.arange(n_obs)
    groups = np.array_split(indices, n_groups)
    embargo = int(n_obs * pct_embargo)

    for combo in combinations(range(n_groups), n_test_groups):
        test_indices = np.concatenate([groups[g] for g in combo])
        test_indices.sort()

        # purge: drop any train obs whose label interval overlaps ANY test group
        drop = np.zeros(n_obs, dtype=bool)
        drop[test_indices] = True
        for g in combo:
            g_start_time = t1.index[groups[g][0]]
            g_max_t1 = t1.iloc[groups[g]].max()
            # train obs whose label ends at/after the group start AND which
            # start at/before the group's label end -> overlapping
            overlaps = (t1.values >= g_start_time) & (t1.index.values <= g_max_t1)
            drop |= overlaps
            # embargo the window right after this test group
            right = t1.index.searchsorted(g_max_t1, side="right")
            drop[right:right + embargo] = True

        train_indices = indices[~drop]
        yield train_indices, test_indices


# ==========================================================================
# Ch.4 sec 4.3 -- CONCURRENCY (number of co-events)
# ==========================================================================
def num_co_events(n_bars: int, start_idx: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
    """
    For each bar, how many label intervals span it (AFML snippet 4.1).

    The book iterates per-label; this is the vectorized difference-array
    equivalent (O(n) instead of O(n*avg_span)), verified identical in
    test_afml.py.
    """
    diff = np.zeros(n_bars + 1, dtype=np.int64)
    np.add.at(diff, start_idx, 1)
    np.add.at(diff, np.minimum(end_idx + 1, n_bars), -1)
    return np.cumsum(diff)[:n_bars]


# ==========================================================================
# Ch.4 sec 4.4 -- AVERAGE UNIQUENESS
# ==========================================================================
def get_avg_uniqueness(n_bars: int, start_idx: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
    """
    Average uniqueness of each label (AFML snippet 4.2).

    uniqueness of label i = mean over bars t in [t0_i, t1_i] of 1/concurrency[t]

    A label that shares all its bars with 20 other labels has uniqueness
    ~1/20 -- it contributes only ~5% of an independent observation. Summing
    uniqueness over the sample gives the EFFECTIVE number of independent
    observations, which is what your error bars should really be based on.
    """
    conc = num_co_events(n_bars, start_idx, end_idx).astype(float)
    conc[conc == 0] = 1.0
    inv = 1.0 / conc
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    span = (end_idx - start_idx + 1).astype(float)
    return (cum[end_idx + 1] - cum[start_idx]) / span


# ==========================================================================
# Ch.4 sec 4.5 -- SEQUENTIAL BOOTSTRAP
# ==========================================================================
def seq_bootstrap(start_idx: np.ndarray, end_idx: np.ndarray, n_bars: int,
                  size: int | None = None, random_state: int | None = None) -> np.ndarray:
    """
    Sequential bootstrap (AFML snippets 4.4-4.5).

    Standard bootstrap draws IID, which over-samples redundant overlapping
    observations. Sequential bootstrap draws each new sample with probability
    proportional to how UNIQUE it is given what's already been drawn, so the
    resulting sample is much closer to IID.

    O(size * n_obs); intended for moderate samples. Returns drawn indices.
    """
    rng = np.random.default_rng(random_state)
    n_obs = len(start_idx)
    if size is None:
        size = n_obs

    drawn: list[int] = []
    conc = np.zeros(n_bars, dtype=float)  # concurrency from already-drawn samples

    cum_template = np.arange(n_bars + 1, dtype=float)
    for _ in range(size):
        # avg uniqueness of each candidate GIVEN what's drawn so far
        inv = 1.0 / (conc + 1.0)
        cum = np.concatenate([[0.0], np.cumsum(inv)])
        span = (end_idx - start_idx + 1).astype(float)
        avg_u = (cum[end_idx + 1] - cum[start_idx]) / span

        prob = avg_u / avg_u.sum()
        pick = rng.choice(n_obs, p=prob)
        drawn.append(int(pick))
        conc[start_idx[pick]:end_idx[pick] + 1] += 1.0

    _ = cum_template
    return np.array(drawn)


# ==========================================================================
# Ch.4 sec 4.6 -- SAMPLE WEIGHTS BY RETURN ATTRIBUTION
# ==========================================================================
def sample_weights_by_return(n_bars: int, start_idx: np.ndarray, end_idx: np.ndarray,
                             log_returns: np.ndarray) -> np.ndarray:
    """
    Weight each observation by the absolute sum of its uniqueness-adjusted
    log returns (AFML snippet 4.10), then normalize to average 1.

    Rationale from the book: a label spanning a large, mostly-unique price
    move carries more information than one spanning noise, and should
    influence the fit proportionally.
    """
    conc = num_co_events(n_bars, start_idx, end_idx).astype(float)
    conc[conc == 0] = 1.0
    attrib = log_returns / conc
    cum = np.concatenate([[0.0], np.cumsum(attrib)])
    w = np.abs(cum[end_idx + 1] - cum[start_idx])
    return w * len(w) / w.sum() if w.sum() > 0 else np.ones_like(w)


# ==========================================================================
# Ch.4 sec 4.7 -- TIME DECAY
# ==========================================================================
def time_decay_weights(avg_uniqueness: np.ndarray, last_weight: float = 1.0) -> np.ndarray:
    """
    Piecewise-linear time decay on CUMULATIVE UNIQUENESS (AFML snippet 4.11),
    not on clock time -- the book's point is that decay should track how much
    independent information has accrued, not how many calendar days passed.

    last_weight (c): weight of the OLDEST observation.
       c = 1   -> no decay
       0<c<1   -> linear decay to c
       c = 0   -> oldest observation gets zero weight
       c < 0   -> the oldest -c fraction of observations get zero weight
    """
    cum = avg_uniqueness.cumsum()
    total = cum.iloc[-1] if isinstance(cum, pd.Series) else cum[-1]

    if last_weight >= 0:
        slope = (1.0 - last_weight) / total
    else:
        slope = 1.0 / ((last_weight + 1) * total)
    const = 1.0 - slope * total
    w = const + slope * cum
    w[w < 0] = 0
    return w


# ==========================================================================
# helper: build t1 for a fixed-horizon label
# ==========================================================================
def build_t1(timestamps: pd.Series, horizon: int) -> pd.Series:
    """
    For a label defined as sign(close[t+n] - close[t]), the label interval is
    [t, t+n]. Returns a Series indexed by t0 with values t1, as PurgedKFold
    and the Ch.4 functions expect.
    """
    ts = pd.Series(timestamps.values, index=timestamps.values)
    end = ts.shift(-horizon)
    end = end.fillna(ts.iloc[-1])
    return end
