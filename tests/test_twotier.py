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

def test_percentile_scoring_gives_features_equal_weight():
    train = pd.concat([_windows(wl, f"{wl}_{i}", f"2020-0{i+1}-01") for wl in ["A", "B"] for i in range(3)])
    train["P_change"] = RNG.standard_cauchy(len(train))   # a heavy-tailed, noisy feature
    det = TwoTierDetector(["P"], min_baseline=120, scale_floor=0.1, scoring="percentile").fit(train)

    assert 0 < det.well_threshold_ < 1
    # The most extreme calibration value of each feature gets the same percentile,
    # however different their raw sizes are
    top = {c: det.cal_sorted_[c][-1] for c in ["P_logstd", "P_change"]}
    p_std = det._combine(pd.DataFrame({"P_logstd": [top["P_logstd"]]})).iloc[0]
    p_chg = det._combine(pd.DataFrame({"P_change": [top["P_change"]]})).iloc[0]
    assert top["P_change"] > 10 * top["P_logstd"]   # raw sizes are very different...
    assert np.isclose(p_std, p_chg)                  # ...but the ranks are equal

def test_two_tier_fits_and_alarms():
    train = pd.concat([_windows(w, f"{w}_{i}", f"2020-0{i+1}-01") for w in ["A", "B"] for i in range(3)])
    det = TwoTierDetector(["P"], min_baseline=120).fit(train)
    assert det.calibration_used_ == "out_of_sample"

    test = _windows("C", "C_1", "2020-06-01")
    alarm, tier = det.alarms_instance(test, history=test.iloc[:0])   # no history -> global
    assert tier == "global" and len(alarm) == len(test)

def test_level_is_per_well_only_and_catches_a_shift():
    train = pd.concat([_windows(wl, f"{wl}_{i}", f"2020-0{i+1}-01") for wl in ["A", "B"] for i in range(3)])
    train["P_mean"] = np.where(train["well"] == "A", 1000.0, 5000.0) + RNG.normal(0, 1, len(train))
    det = TwoTierDetector(["P"], min_baseline=120, scale_floor=0.1, scoring="percentile",
                          features=("logstd", "change", "level")).fit(train)
    assert "P_level" not in det.global_._features(train).columns   # global tier never sees level

    hist = _windows("C", "C_0", "2020-06-01")
    hist["P_mean"] = 3000 + RNG.normal(0, 1, len(hist))            # well C's normal level
    shifted = _windows("C", "C_1", "2020-07-01")
    shifted["P_mean"] = 3020 + RNG.normal(0, 1, len(shifted))      # level 20 spreads higher
    alarm, tier = det.alarms_instance(shifted, hist)
    assert tier == "well" and alarm.mean() > 0.9


def test_without_level_the_floor_reference_is_unchanged():
    train = pd.concat([_windows(wl, f"{wl}_{i}", f"2020-0{i+1}-01") for wl in ["A", "B"] for i in range(3)])
    det = TwoTierDetector(["P"], min_baseline=120, scale_floor=0.1, scoring="percentile").fit(train)
    pd.testing.assert_series_equal(det.floor_scale_, det.global_.scale_)

def test_iforest_scoring_flags_a_clear_anomaly_but_not_normal():
    train = pd.concat([_windows(wl, f"{wl}_{i}", f"2020-0{i+1}-01") for wl in ["A", "B"] for i in range(3)])
    det = TwoTierDetector(["P"], min_baseline=120, scale_floor=0.1, scoring="iforest").fit(train)
    assert np.isfinite(det.well_threshold_)

    hist = _windows("C", "C_0", "2020-06-01")
    odd = _windows("C", "C_1", "2020-07-01")
    odd["P_std"] = 300.0 + RNG.normal(0, 5, len(odd))       # movement 3× this well's normal
    alarm_odd, tier = det.alarms_instance(odd, hist)
    assert tier == "well" and alarm_odd.mean() > 0.9

    calm = _windows("C", "C_2", "2020-08-01")               # ordinary operation for this well
    alarm_calm, _ = det.alarms_instance(calm, hist)
    assert alarm_calm.mean() < 0.1

def test_ensemble_is_calibrated_and_flags_a_clear_anomaly():
    train = pd.concat([_windows(wl, f"{wl}_{i}", f"2020-0{i+1}-01") for wl in ["A", "B"] for i in range(3)])
    det = TwoTierDetector(["P"], min_baseline=120, scale_floor=0.1, scoring="ensemble").fit(train)
    assert 0 < det.well_threshold_ < 1        # ensemble scores are ranks, so the threshold is too

    hist = _windows("C", "C_0", "2020-06-01")
    odd = _windows("C", "C_1", "2020-07-01")
    odd["P_std"] = 300.0 + RNG.normal(0, 5, len(odd))
    alarm_odd, tier = det.alarms_instance(odd, hist)
    assert tier == "well" and alarm_odd.mean() > 0.9

    calm = _windows("C", "C_2", "2020-08-01")
    alarm_calm, _ = det.alarms_instance(calm, hist)
    assert alarm_calm.mean() < 0.1