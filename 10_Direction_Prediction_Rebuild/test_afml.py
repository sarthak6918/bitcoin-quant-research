"""
Correctness tests for afml.py -- verifying the implementations actually match
the definitions in Lopez de Prado's Advances in Financial Machine Learning,
rather than merely running without error.
"""
import numpy as np
import pandas as pd

from afml import (
    get_train_times, get_embargo_times, PurgedKFold, num_co_events,
    get_avg_uniqueness, seq_bootstrap, time_decay_weights, build_t1,
)


def test_num_co_events_matches_naive_loop():
    """Vectorized concurrency == the book's per-label accumulation loop."""
    rng = np.random.default_rng(0)
    n_bars = 500
    start = np.sort(rng.integers(0, n_bars - 30, size=200))
    end = np.minimum(start + rng.integers(1, 25, size=200), n_bars - 1)

    naive = np.zeros(n_bars, dtype=np.int64)
    for s, e in zip(start, end):
        naive[s:e + 1] += 1

    fast = num_co_events(n_bars, start, end)
    assert np.array_equal(naive, fast), "vectorized concurrency != naive loop"
    print("PASS: num_co_events matches naive per-label loop")


def test_avg_uniqueness_matches_naive():
    rng = np.random.default_rng(1)
    n_bars = 300
    start = np.sort(rng.integers(0, n_bars - 20, size=120))
    end = np.minimum(start + rng.integers(1, 15, size=120), n_bars - 1)

    conc = num_co_events(n_bars, start, end).astype(float)
    conc[conc == 0] = 1.0
    naive = np.array([np.mean(1.0 / conc[s:e + 1]) for s, e in zip(start, end)])
    fast = get_avg_uniqueness(n_bars, start, end)
    assert np.allclose(naive, fast), "avg uniqueness mismatch"
    print("PASS: get_avg_uniqueness matches naive per-label mean")


def test_uniqueness_bounds_and_intuition():
    """Non-overlapping labels -> uniqueness 1. Fully stacked -> 1/k."""
    n_bars = 100
    # 10 disjoint labels of width 10
    start = np.arange(0, 100, 10)
    end = start + 9
    u = get_avg_uniqueness(n_bars, start, end)
    assert np.allclose(u, 1.0), f"disjoint labels should be fully unique, got {u}"

    # 5 identical labels spanning the same bars
    start2 = np.zeros(5, dtype=int)
    end2 = np.full(5, 49)
    u2 = get_avg_uniqueness(n_bars, start2, end2)
    assert np.allclose(u2, 1 / 5), f"5 identical labels should each be 1/5 unique, got {u2}"
    print("PASS: uniqueness = 1.0 when disjoint, 1/k when k labels coincide")


def test_purging_removes_all_overlap():
    """After get_train_times, NO training label may overlap a test label."""
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    t1 = pd.Series(idx[np.minimum(np.arange(200) + 5, 199)], index=idx)

    test_times = pd.Series([t1.iloc[120]], index=[idx[100]])  # test spans 100..125
    trn = get_train_times(t1, test_times)

    ts, te = idx[100], t1.iloc[120]
    for t0, t1_val in trn.items():
        overlap = (t0 <= te) and (ts <= t1_val)
        assert not overlap, f"purge failed: train label [{t0},{t1_val}] overlaps [{ts},{te}]"
    print(f"PASS: purging removed all overlap ({len(t1)} -> {len(trn)} train obs)")


def test_purgedkfold_no_leakage():
    """
    The real test: for every fold, assert no training observation's label
    interval overlaps any test observation's label interval, and that the
    embargo gap is actually respected.
    """
    n = 1000
    horizon = 10
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"f": np.arange(n)}, index=idx)
    t1 = build_t1(pd.Series(idx), horizon)

    pct_embargo = 0.01
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=pct_embargo)
    embargo_bars = int(n * pct_embargo)

    for k, (tr, te) in enumerate(cv.split(X)):
        assert len(np.intersect1d(tr, te)) == 0, "train/test index overlap!"

        test_start_time = t1.index[te[0]]
        test_end_label = t1.iloc[te].max()

        for i in tr:
            a0, a1 = t1.index[i], t1.iloc[i]
            overlap = (a0 <= test_end_label) and (test_start_time <= a1)
            if overlap:
                raise AssertionError(
                    f"fold {k}: train label [{a0},{a1}] overlaps test [{test_start_time},{test_end_label}]"
                )

        right = tr[tr > te[-1]]
        if len(right):
            gap_start = t1.index.searchsorted(test_end_label, side="right")
            assert right.min() >= gap_start + embargo_bars, (
                f"fold {k}: embargo not respected -- first right-side train idx "
                f"{right.min()} < {gap_start + embargo_bars}"
            )
    print("PASS: PurgedKFold produces zero label overlap and respects the embargo")


def test_embargo_increases_gap():
    """More embargo -> strictly fewer training observations."""
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"f": np.arange(n)}, index=idx)
    t1 = build_t1(pd.Series(idx), 10)

    sizes = []
    for pct in (0.0, 0.01, 0.05):
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=pct)
        sizes.append(sum(len(tr) for tr, _ in cv.split(X)))
    assert sizes[0] > sizes[1] > sizes[2], f"embargo should shrink train set, got {sizes}"
    print(f"PASS: embargo shrinks training set monotonically {sizes}")


def test_seq_bootstrap_beats_iid_uniqueness():
    """
    The book's claim: sequential bootstrap yields a sample with HIGHER average
    uniqueness than a standard IID bootstrap. Verify it empirically.
    """
    rng = np.random.default_rng(7)
    n_bars, n_obs = 200, 100
    start = np.sort(rng.integers(0, n_bars - 20, size=n_obs))
    end = np.minimum(start + 15, n_bars - 1)

    seq = seq_bootstrap(start, end, n_bars, size=50, random_state=3)
    iid = rng.integers(0, n_obs, size=50)

    def mean_uniqueness(draw):
        return get_avg_uniqueness(n_bars, start[draw], end[draw]).mean()

    u_seq, u_iid = mean_uniqueness(seq), mean_uniqueness(iid)
    assert u_seq > u_iid, f"sequential bootstrap {u_seq:.4f} should exceed IID {u_iid:.4f}"
    print(f"PASS: sequential bootstrap uniqueness {u_seq:.4f} > IID bootstrap {u_iid:.4f}")


def test_time_decay():
    u = pd.Series(np.ones(100))
    w1 = time_decay_weights(u, last_weight=1.0)
    assert np.allclose(w1, 1.0), "c=1 should mean no decay"

    w2 = time_decay_weights(u, last_weight=0.0)
    assert w2.iloc[-1] > w2.iloc[0] and np.isclose(w2.iloc[0], 0, atol=0.02), "c=0 decays oldest to ~0"

    w3 = time_decay_weights(u, last_weight=-0.5)
    assert (w3 == 0).sum() > 0, "c<0 should zero out the oldest portion"
    print("PASS: time decay behaves per book spec for c=1, c=0, c<0")


def test_embargo_times_helper():
    idx = pd.date_range("2020-01-01", periods=100, freq="h")
    emb = get_embargo_times(idx, 0.05)
    assert len(emb) == 100
    assert emb.iloc[0] == idx[5], "1% embargo should map bar 0 -> bar 5"
    print("PASS: get_embargo_times maps forward by the embargo width")


if __name__ == "__main__":
    test_num_co_events_matches_naive_loop()
    test_avg_uniqueness_matches_naive()
    test_uniqueness_bounds_and_intuition()
    test_purging_removes_all_overlap()
    test_purgedkfold_no_leakage()
    test_embargo_increases_gap()
    test_seq_bootstrap_beats_iid_uniqueness()
    test_time_decay()
    test_embargo_times_helper()
    print("\nAll AFML implementation tests passed.")
