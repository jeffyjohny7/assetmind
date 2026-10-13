import numpy as np
import pandas as pd
from assetmind.evaluation.event_metrics import instance_report, warning_curve, false_alarms_per_day


def _instance(phases, alarms):
    t = pd.date_range("2020-01-01", periods=len(phases), freq="min")
    return pd.DataFrame({"window_end": t, "phase": phases, "alarm": alarms})


PHASES = ["normal"] * 10 + ["transient"] * 10 + ["event"] * 10


def test_early_detection_gives_positive_lead_time():
    r = instance_report(_instance(PHASES, [False] * 15 + [True] * 15))
    assert r["detected"] is True
    assert np.isclose(r["delay_h"], 5 / 60)       # 5 min after onset
    assert np.isclose(r["lead_time_h"], 5 / 60)   # 5 min before the event
    assert r["false_alarms"] == 0


def test_alarm_only_in_normal_is_a_false_alarm_not_a_detection():
    alarms = [False] * 30
    alarms[2] = alarms[3] = True                  # one episode, two windows
    r = instance_report(_instance(PHASES, alarms))
    assert r["detected"] is False
    assert r["false_alarms"] == 1
    assert np.isnan(r["lead_time_h"])


def test_normal_file_counts_only_false_alarms():
    alarms = [False] * 30
    alarms[1] = alarms[5] = True                  # two separate episodes
    r = instance_report(_instance(["normal"] * 30, alarms))
    assert r["has_event"] is False
    assert r["false_alarms"] == 2
    assert np.isclose(r["normal_hours"], 0.5)


def test_warning_curve_and_false_alarm_rate():
    reports = pd.DataFrame({
        "has_event": [True, True, True, False],
        "lead_time_h": [8.0, 2.0, np.nan, np.nan],   # third event was missed
        "false_alarms": [0, 1, 0, 1],
        "normal_hours": [12, 12, 0, 24],
    })
    curve = warning_curve(reports, thresholds_h=(0, 6))
    assert np.isclose(curve[0], 2 / 3) and np.isclose(curve[6], 1 / 3)
    assert np.isclose(false_alarms_per_day(reports), 1.0)   # 2 alarms over 48 h