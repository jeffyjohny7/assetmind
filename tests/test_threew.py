import numpy as np
import pandas as pd
from assetmind.ingestion.threew import make_windows

IDX = pd.date_range("2020-01-01", periods=600, freq="s")
RNG = np.random.default_rng(0)


def _signal(values, label=0):
    return pd.DataFrame({"P": values, "label": label}, index=IDX)


def _calm():
    return 100 + 0.01 * RNG.standard_normal(600)   # steady, with realistic sensor noise


def test_window_count_and_end_label():
    df = _signal(_calm())
    df.loc[IDX[400]:, "label"] = 106
    w = make_windows(df, ["P"], window=300, step=60)
    assert len(w) == 6
    assert w.loc[1, "label"] == 0 and w.loc[2, "label"] == 106
    assert w.loc[2, "phase"] == "transient"


def test_oscillation_visible_in_std_not_mean():
    t = np.arange(600)
    steady = make_windows(_signal(_calm()), ["P"])
    slugging = make_windows(_signal(100 + 8 * np.sin(2 * np.pi * t / 120), label=3), ["P"])
    assert abs(slugging["P_mean"].mean() - 100) < 2
    assert steady["P_std"].max() < 0.1
    assert slugging["P_std"].min() > 4


def test_sensor_status_rules():
    assert set(make_windows(_signal(np.zeros(600)), ["P"])["P_status"]) == {"zero"}
    assert set(make_windows(_signal(np.full(600, 16.259e6)), ["P"])["P_status"]) == {"frozen"}
    assert set(make_windows(_signal(_calm()), ["P"])["P_status"]) == {"ok"}


def test_invalid_windows_have_no_features():
    w = make_windows(_signal(np.zeros(600)), ["P"])
    assert w[["P_mean", "P_std", "P_change"]].isna().all().all()