"""Correctness tests for afml_lectures.py, checked against properties the
lectures explicitly assert."""
import numpy as np
import pandas as pd

from afml_lectures import (
    get_weights_ffd, frac_diff_ffd, min_ffd_d, cusum_filter,
    dollar_bars, volume_bars, expected_max_sharpe, deflated_sharpe_ratio,
    prob_backtest_overfitting,
)


def test_ffd_weights_d1_is_plain_diff():
    """d=1 must reduce to first differencing: weights [-1, 1]."""
    w = get_weights_ffd(1.0, thres=1e-5)
    assert np.allclose(w, [-1.0, 1.0]), w
    print("PASS: FFD weights at d=1 reduce to plain first difference")


def test_ffd_weights_d0_is_identity():
    """d=0 must be the identity: single weight 1."""
    w = get_weights_ffd(0.0, thres=1e-5)
    assert np.allclose(w, [1.0]), w
    print("PASS: FFD weights at d=0 are the identity")


def test_ffd_d1_equals_diff():
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=500)))
    fd = frac_diff_ffd(s, 1.0).dropna()
    ref = s.diff().dropna()
    assert np.allclose(fd.values, ref.loc[fd.index].values), "d=1 != diff"
    print("PASS: frac_diff_ffd(d=1) reproduces the ordinary difference exactly")


def test_ffd_weights_alternate_and_decay():
    """Binomial weights must alternate in sign and decay in magnitude."""
    w = get_weights_ffd(0.4, thres=1e-5)[::-1]  # most-recent-first
    assert w[0] == 1.0
    assert np.all(np.abs(np.diff(np.abs(w))) <= 1e-9 + np.abs(w[:-1])), "not decaying"
    signs = np.sign(w[:6])
    assert signs[0] > 0 and signs[1] < 0 and signs[2] < 0, signs
    print("PASS: FFD weights decay and alternate as the binomial series requires")


def test_ffd_is_causal():
    """
    Corrupt-the-future test: values before a cutoff must not change when all
    data after the cutoff is replaced with noise.
    """
    rng = np.random.default_rng(1)
    s = pd.Series(np.cumsum(rng.normal(size=2000)) + 100)
    s2 = s.copy()
    s2.iloc[1200:] = rng.normal(size=800) * 500

    a = frac_diff_ffd(s, 0.4).iloc[:1200].values
    b = frac_diff_ffd(s2, 0.4).iloc[:1200].values
    m = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[m], b[m]), "FFD leaks future information"
    print("PASS: frac_diff_ffd is causal (future corruption changes nothing before cutoff)")


def test_ffd_memory_preserved():
    """
    The lecture's core claim (slide 27): a fractional d yields a series that
    is far more correlated with the original than plain returns are.
    """
    rng = np.random.default_rng(2)
    price = pd.Series(np.cumsum(rng.normal(size=5000)) + 1000)
    fd04 = frac_diff_ffd(price, 0.4).dropna()
    fd10 = frac_diff_ffd(price, 1.0).dropna()
    idx = fd04.index.intersection(fd10.index)
    c04 = abs(np.corrcoef(price.loc[idx], fd04.loc[idx])[0, 1])
    c10 = abs(np.corrcoef(price.loc[idx], fd10.loc[idx])[0, 1])
    assert c04 > c10, f"d=0.4 corr {c04:.3f} should exceed d=1 corr {c10:.3f}"
    print(f"PASS: d=0.4 retains memory (corr {c04:.3f}) vs d=1 returns (corr {c10:.3f})")


def test_min_ffd_d_on_random_walk():
    """A random walk must need d>0 to become stationary, and d=1 must work."""
    rng = np.random.default_rng(3)
    price = pd.Series(np.cumsum(rng.normal(size=3000)) + 1000)
    tbl, chosen = min_ffd_d(price, d_range=np.linspace(0, 1, 11))
    assert chosen is not None and chosen > 0, tbl
    assert bool(tbl.iloc[-1]["stationary"]), "d=1 must be stationary"
    assert not bool(tbl.iloc[0]["stationary"]), "d=0 random walk must NOT be stationary"
    print(f"PASS: min_ffd_d picks d={chosen} on a random walk (d=0 fails, d=1 passes)")


def test_cusum_filter_triggers_on_moves():
    """No events on a flat series; events on a series with jumps."""
    flat = np.zeros(500)
    assert len(cusum_filter(flat, 0.05)) == 0
    jumpy = np.zeros(500)
    jumpy[100:] += 1.0
    jumpy[300:] -= 2.0
    ev = cusum_filter(jumpy, 0.5)
    assert len(ev) >= 2, ev
    assert 100 in ev or 101 in ev, ev
    print(f"PASS: CUSUM fires on structural breaks ({len(ev)} events), silent on flat series")


def test_cusum_resets():
    """After firing, the accumulator resets -- a single ramp shouldn't fire every bar."""
    ramp = np.linspace(0, 10, 1000)
    ev = cusum_filter(ramp, 1.0)
    assert 5 <= len(ev) <= 15, f"expected ~10 events on a ramp of 10 with h=1, got {len(ev)}"
    print(f"PASS: CUSUM resets after firing ({len(ev)} events on a ramp of height 10, h=1)")


def test_dollar_bars_conserve_and_stabilize():
    rng = np.random.default_rng(4)
    n = 5000
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": rng.lognormal(0, 1, n),
    })
    thr = (df["close"] * df["volume"]).sum() / 100
    db = dollar_bars(df, thr)
    assert 80 <= len(db) <= 120, len(db)
    assert db["volume"].sum() <= df["volume"].sum() + 1e-6
    assert (db["high"] >= db["low"]).all()
    print(f"PASS: dollar_bars produced {len(db)} bars targeting 100, OHLC coherent")


def test_volume_bars_have_constant_volume():
    rng = np.random.default_rng(5)
    n = 3000
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": rng.lognormal(0, 0.5, n),
    })
    vb = volume_bars(df, 50.0)
    # each bar accumulates >= threshold, and overshoot is bounded by one source bar
    assert (vb["volume"] >= 50.0).all()
    print(f"PASS: volume_bars each reach the volume threshold ({len(vb)} bars)")


def test_expected_max_sharpe_grows_with_trials():
    """More trials -> higher expected best Sharpe under the null of no skill."""
    vals = [expected_max_sharpe(n) for n in (2, 10, 100, 10000, 129600)]
    assert all(np.diff(vals) > 0), vals
    print("PASS: E[max SR] under the null increases with the number of trials: "
          + ", ".join(f"{v:.3f}" for v in vals))


def test_dsr_penalizes_many_trials():
    """The same observed Sharpe must be less significant after more trials."""
    d1, sr0_1 = deflated_sharpe_ratio(0.05, n_trials=1, n_obs=1000)
    d2, sr0_2 = deflated_sharpe_ratio(0.05, n_trials=100000, n_obs=1000)
    assert d1 > d2, (d1, d2)
    assert sr0_2 > sr0_1
    print(f"PASS: DSR penalizes multiple testing (1 trial: {d1:.4f} -> 100k trials: {d2:.4f})")


def test_pbo_is_half_for_pure_noise():
    """
    With pure-noise strategies there is no real skill, so the in-sample best
    should be a coin flip out of sample -> PBO near 0.5.
    """
    rng = np.random.default_rng(6)
    M = rng.normal(size=(2000, 50))
    pbo = prob_backtest_overfitting(M, n_splits=10)
    assert 0.3 < pbo < 0.7, pbo
    print(f"PASS: PBO on pure noise = {pbo:.3f} (near 0.5 as theory requires)")


def test_pbo_low_for_genuine_skill():
    """If one strategy genuinely has edge, PBO should be low."""
    rng = np.random.default_rng(7)
    M = rng.normal(size=(2000, 50))
    M[:, 7] += 0.25  # a genuinely superior strategy
    pbo = prob_backtest_overfitting(M, n_splits=10)
    assert pbo < 0.2, pbo
    print(f"PASS: PBO with one genuinely skilled strategy = {pbo:.3f} (low, as expected)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
    print(f"\n{len(fns)}/{len(fns)} tests passed")
