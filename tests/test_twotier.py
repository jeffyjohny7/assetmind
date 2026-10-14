import numpy as np
import pandas as pd
from assetmind.models.anomaly_twotier import TwoTierDetector

RNG = np.random.default_rng(0)


def _windows(well, inst, start, n=200):
    t = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({"well": well, "instance_id": inst, "window_end": t, "phase": "normal",
                         "P_std": 100 + RNG.normal(0, 5, n), "P_change": RNG.normal(0, 1, n)})


def test_two_tier_fits_and_alarms():
    train = pd.concat([_windows(w, f"{w}_{i}", f"2020-0{i+1}-01") for w in ["A", "B"] for i in range(3)])
    det = TwoTierDetector(["P"], min_baseline=120).fit(train)
    assert det.calibration_used_ == "out_of_sample"

    test = _windows("C", "C_1", "2020-06-01")
    alarm, tier = det.alarms_instance(test, history=test.iloc[:0])   # no history -> global
    assert tier == "global" and len(alarm) == len(test)