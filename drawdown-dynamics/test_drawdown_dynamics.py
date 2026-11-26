"""Offline tests on synthetic prices:  python test_drawdown_dynamics.py"""
import contextlib, importlib.util, io, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "dd", Path(__file__).with_name("drawdown-dynamics.py"))
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


def synthetic(n=4000, seed=1):
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0.0003, 0.017, n)))
    prev = np.r_[100, close[:-1]]
    open_ = prev * np.exp(r.normal(0, 0.008, n))
    high = np.maximum(open_, close) * np.exp(np.abs(r.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(r.normal(0, 0.006, n)))
    return pd.DataFrame(dict(Open=open_, High=high, Low=low, Close=close),
                        index=pd.bdate_range("2000-01-03", periods=n))


def manual(o, h, l, c):
    a = [np.array([x], float) for x in (o, h, l, c)]
    return dict(zip("ohlc", a), length=np.array([len(o)]))


def test_entry_day_and_gap_fills():
    rows = np.array([0])
    run = lambda p, s=0.10: dd.simulate_stop(p, rows, s, 2, 0.0, 0.0)
    # Entry-day low triggers; fill at the stop level.
    p = manual([1, 1, 1], [1, 1, 1], [.85, 1, 1], [.95, 1, 1])
    w, t, f, b = run(p)
    assert t[0] and b[0] == 0 and np.isclose(w[0], .90)
    # Gap through the stop fills at the open.
    p = manual([1, .80, 1], [1, .82, 1], [.99, .78, 1], [.99, .81, 1.1])
    w = run(p)[0]
    assert np.isclose(w[0], .80)
    # Not triggered equals Hold.
    assert np.isclose(run(p, 0.30)[0][0], 1.1)
    # Slippage and cash interest.
    w = dd.simulate_stop(p, rows, .10, 2, .01, .05)[0]
    assert np.isclose(w[0], .80 * .99 * 1.05 ** (1 / 252))
    # Entry-day recovery is deliberately not counted.
    p = manual([1, .9, .9], [1.2, .95, .95], [.85, .88, .88], [.9, .9, .9])
    t = dd.recovery_table(p, [.1], {"a": 1}, {"b": 1})
    assert t.StopHits[0] == 1 and t.Recovered[0] == 0


def test_recovery_matches_brute_force():
    d = synthetic(900)
    e = dd.select_entries(d, 3, 7)
    w = {"s": 5, "m": 21, "x": 63}
    p = dd.historical_paths(d, e, 127)
    o, h, l, c = (d[k].to_numpy() for k in dd.OHLC)
    for eligibility in ["per-window", "common"]:
        table = dd.recovery_table(p, [.03, .10, .60], w, w, eligibility)
        for r in table.itertuples():
            w1, w2 = w[r.W1], w[r.W2]
            need = 126 if eligibility == "common" else w1 + w2
            n = hits = rec = 0
            worst = []
            for i in e:
                if i + need >= len(d):
                    continue
                n += 1
                k = next((j for j in range(i, i + w1 + 1)
                          if l[j] <= o[i] * (1 - r.StopPct)), None)
                if k is None:
                    continue
                hits += 1
                if (h[k + 1:k + w2 + 1] >= o[i]).any():
                    rec += 1
                else:
                    worst.append(l[k:k + w2 + 1].min() / o[i] - 1)
            assert (n, hits, rec) == (r.Entries, r.StopHits, r.Recovered)
            if worst:
                assert np.isclose(np.median(worst),
                                  r.MedianWorstDropNonRecovered)


def test_monthly_entries():
    d = synthetic(500)
    e = dd.select_entries(d, 5, 42)
    sizes = d.index.to_period("M").value_counts().sort_index()
    got = d.index[e].to_period("M").value_counts().sort_index()
    assert (got == sizes.clip(upper=5)).all() and len(set(e)) == len(e)
    assert len(dd.select_entries(d, None, 1)) == len(d)


def test_bootstrap_structure():
    rng = np.random.default_rng(0)
    n = 1000
    i = dd.bootstrap_indices(n, 200, 600, 20, "circular", rng)
    step = (np.diff(i, axis=1) % n) == 1
    inside = np.arange(1, 600) % 20 != 0
    assert step[:, inside].all()
    assert step[:, ~inside].mean() < .01
    i = dd.bootstrap_indices(n, 2000, 600, 20, "stationary", rng)
    breaks = ((np.diff(i, axis=1) % n) != 1).mean()
    assert abs(breaks - (1 / 20) * (1 - 1 / n)) < .002
    for m in ["circular", "stationary"]:          # uniform day weights
        i = dd.bootstrap_indices(50, 4000, 100, 10, m, rng)
        freq = np.bincount(i.ravel(), minlength=50) / i.size
        assert np.abs(freq - .02).max() < .002


def test_reconstruction():
    d = synthetic(300)
    rel = dd.daily_relatives(d)[None]
    p = dd.relatives_to_paths(rel)
    for k, col in zip("ohlc", dd.OHLC):
        assert np.allclose(p[k][0], d[col].to_numpy()[1:] / d.Open.iloc[1])


def test_fhs():
    with contextlib.redirect_stdout(io.StringIO()):
        m = dd.fit_fhs(synthetic(2000))
    p = dd.fhs_paths(m, 300, 200, 3)
    q = dd.fhs_paths(m, 300, 200, 3)
    assert np.array_equal(p["c"], q["c"]) and np.isfinite(p["c"]).all()
    assert (p["l"] <= np.minimum(p["o"], p["c"]) + 1e-9).all()
    assert (p["h"] >= np.maximum(p["o"], p["c"]) - 1e-9).all()


def test_walk_forward_has_no_lookahead():
    d = synthetic(3500)
    altered = d.copy()
    cut = altered.index.year >= 2011
    noise = np.exp(np.cumsum(
        np.random.default_rng(9).normal(-.002, .03, cut.sum())))
    altered.loc[cut, dd.OHLC] = altered.loc[cut, dd.OHLC].mul(noise, axis=0)
    for method in ["historical", "stationary", "circular"]:
        args = dd.parse_args(["--method", method, "--block-lengths", "21",
                              "--paths", "300", "--min-train-years", "5"])
        out = []
        for frame in (d, altered):
            with contextlib.redirect_stdout(io.StringIO()):
                out.append(dd.walk_forward(
                    frame, dd.select_entries(frame, 5, 1),
                    dd.build_scenarios(args), args)[0])
        a, b = (x[x.TestYear <= 2011].ChosenStopPct.tolist() for x in out)
        assert a == b and a


def test_command_line_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "px.csv"
        synthetic(3500).to_csv(csv, index_label="Date")
        with contextlib.redirect_stdout(io.StringIO()):
            dd.main(["--csv", str(csv), "--output-dir", tmp, "--method", "all",
                     "--analysis", "all", "--paths", "300", "--n-seeds", "2",
                     "--block-lengths", "21", "auto"])
        for name in ["recovery_summary", "stop_vs_hold", "best_stop",
                     "by_period", "walk_forward", "walk_forward_summary"]:
            assert len(pd.read_csv(Path(tmp) / f"qqq_{name}.csv")), name


def test_decision_table_and_charts():
    ps = importlib.util.spec_from_file_location(
        "pl", Path(__file__).with_name("plot_results.py"))
    pl = importlib.util.module_from_spec(ps)
    ps.loader.exec_module(pl)
    # Hand-checked table: two stationary rows form ONE family.
    rows = []
    for name, gain in [("historical", .03), ("stationary_L21", -.01),
                       ("stationary_L63", -.01)]:
        for stop, g in [(0.0, 0.0), (0.1, gain)]:
            rows.append(dict(Scenario=name, Seed=1, StopPct=stop,
                             MeanLogGrowth=.05 + g, ES=-.4 + stop,
                             TriggerRate=stop))
    t = pl.decision_table(pd.DataFrame(rows), "mean_log", -.35).iloc[1]
    assert np.isclose(t.BalancedGain, .01) and np.isclose(t.WorstGain, -.01)
    assert np.isclose(t.ShareBeatsHold, 1 / 3)
    assert np.isclose(t.MaxRegret, .01) and t.ShareMeetsMinES == 1
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "px.csv"
        synthetic(3500).to_csv(csv, index_label="Date")
        with contextlib.redirect_stdout(io.StringIO()):
            dd.main(["--csv", str(csv), "--output-dir", tmp, "--method", "all",
                     "--analysis", "all", "--paths", "200",
                     "--block-lengths", "21"])
            pl.main(["--results-dir", tmp])
        assert len(list((Path(tmp) / "charts").glob("*.png"))) == 8
        # A later compare-only run must not redraw stale walk-forward output.
        with contextlib.redirect_stdout(io.StringIO()):
            dd.main(["--csv", str(csv), "--output-dir", tmp,
                     "--analysis", "compare"])
            pl.main(["--results-dir", tmp])
        assert len(list((Path(tmp) / "charts").glob("*.png"))) == 6


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
