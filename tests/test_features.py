import numpy as np
import pandas as pd
from assetmind.features.cmapss import CmapssFeatureBuilder


def _toy():
    cycles = np.arange(1, 31)
    return pd.DataFrame({
        "unit": 1, "cycle": cycles, "rul": 30 - cycles,
        "s_1": 5.0,                    # constant -> must be dropped
        "s_2": 100.0 + 2.0 * cycles,   # perfect line, slope 2
    })


def test_constant_sensor_dropped():
    fb = CmapssFeatureBuilder().fit(_toy())
    assert fb.sensors_ == ["s_2"] and fb.dropped_ == ["s_1"]


def test_slope_recovers_true_slope():
    out = CmapssFeatureBuilder(window=5).fit(_toy()).transform(_toy())
    assert np.allclose(out["s_2_slope5"].dropna(), 2.0)


def test_no_future_leakage():
    full, cut = _toy(), _toy().iloc[:15]
    fb = CmapssFeatureBuilder(baseline_n=5).fit(full)
    a = fb.transform(full).iloc[:15].drop(columns="rul_capped")
    b = fb.transform(cut).drop(columns="rul_capped")
    pd.testing.assert_frame_equal(a, b)