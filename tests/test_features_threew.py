import numpy as np
import pandas as pd
import pytest
from assetmind.features.threew import add_hour_features


def _instance(iid, n, slope):
    t = pd.date_range("2020-01-01", periods=n, freq="min")
    return pd.DataFrame({"instance_id": iid, "window_end": t, "P_mean": 100 + slope * np.arange(n)})


def test_level_change_uses_fixed_one_hour_lookback():
    out = add_hour_features(pd.concat([_instance("a", 120, 2.0), _instance("b", 120, -1.0)]), ["P"])
    a, b = out[out["instance_id"] == "a"], out[out["instance_id"] == "b"]
    assert a["P_lvl1h"].iloc[:60].isna().all()            # first hour: nothing to look back to
    assert np.allclose(a["P_lvl1h"].iloc[60:], 120.0)     # +2 per minute × 60 minutes
    assert np.allclose(b["P_lvl1h"].iloc[60:], -60.0)
    assert b["P_lvl1h"].iloc[:60].isna().all()            # instance b doesn't borrow from a


def test_irregular_spacing_is_rejected():
    gappy = _instance("a", 120, 1.0).drop(index=50)
    with pytest.raises(AssertionError):
        add_hour_features(gappy, ["P"])