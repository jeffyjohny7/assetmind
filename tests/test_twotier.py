import numpy as np
import pandas as pd
from assetmind.models.anomaly_twotier import TwoTierDetector
from assetmind.models.eval_anomaly_twotier import evaluate

RNG = np.random.default_rng(0)


def _windows(well, inst, start, n=200):
    t = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({"well": well, "instance_id": inst, "window_end": t, "phase": "normal",
                         "P_std": 100 + RNG.normal(0, 5, n), "P_change": RNG.normal(0, 1, n)})

def test_evaluate_runs_end_to_end():
    # 3 wells × 3 files; each file's name carries the same start date as its windows
    w = pd.concat([_windows(wl, f"{wl}_20200{i+1}01000000", f"2020-0{i+1}-01")
                   for wl in ["A", "B", "C"] for i in range(3)], ignore_index=True)
    w["event_class"] = 0
    meta = w.groupby("instance_id").agg(well=("well", "first"), end=("window_end", "max"))
    meta["start"] = pd.to_datetime(meta.index.str.split("_").str[-1], format="%Y%m%d%H%M%S")
    rep = evaluate(w, meta, 0.995, 3, "out_of_sample", sensors=["P"])
    assert len(rep) == 9 and set(rep["tier"]) <= {"well", "global"}

def test_scale_floor_prevents_exploding_threshold():
    normal = pd.concat([_windows(wl, f"{wl}_{i}", f"2020-0{i+1}-01") for wl in ["A", "B"] for i in range(3)])
    calm = pd.concat([_windows("D", f"D_{i}", f"2020-0{i+1}-01") for i in range(3)])
    # Well D: almost perfectly constant in its early files, then a small step in its last file
    calm["P_std"] = np.where(calm["instance_id"] == "D_2", 105.0, 100 + 1e-6 * RNG.standard_normal(len(calm)))
    train = pd.concat([normal, calm])

    no_floor = TwoTierDetector(["P"], min_baseline=120, scale_floor=0.0).fit(train)
    floored = TwoTierDetector(["P"], min_baseline=120, scale_floor=0.1).fit(train)
    assert no_floor.well_threshold_ > 1000   # the bug: one near-constant well blows up the threshold
    assert floored.well_threshold_ < 100     # the fix keeps it in a sensible range

def test_two_tier_fits_and_alarms():
    train = pd.concat([_windows(w, f"{w}_{i}", f"2020-0{i+1}-01") for w in ["A", "B"] for i in range(3)])
    det = TwoTierDetector(["P"], min_baseline=120).fit(train)
    assert det.calibration_used_ == "out_of_sample"

    test = _windows("C", "C_1", "2020-06-01")
    alarm, tier = det.alarms_instance(test, history=test.iloc[:0])   # no history -> global
    assert tier == "global" and len(alarm) == len(test)