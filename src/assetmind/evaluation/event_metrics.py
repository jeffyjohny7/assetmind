import numpy as np
import pandas as pd


def _hours(delta) -> float:
    return delta / pd.Timedelta(hours=1) if pd.notna(delta) else np.nan


def instance_report(w: pd.DataFrame, alarm_col: str = "alarm", step_minutes: float = 1.0) -> dict:
    """Event-level summary for ONE instance. Needs columns: window_end, phase, alarm."""
    w = w.sort_values("window_end")
    t = w["window_end"]
    phase = w["phase"].astype(str)
    alarm = w[alarm_col].astype(bool)

    def first_time(mask):
        return t[mask].iloc[0] if mask.any() else pd.NaT

    onset = first_time(phase.ne("normal"))
    event_start = first_time(phase.eq("event"))
    first_alarm = first_time(alarm & phase.ne("normal"))
    alarm_starts = alarm & ~alarm.shift(fill_value=False)   # count each alarm episode once
    normal = phase.eq("normal")

    has_event = pd.notna(onset)
    fa_times = t[alarm_starts & normal]
    fa_before = (((onset - fa_times) / pd.Timedelta(hours=1)).tolist()
                 if has_event else [])
    return {
        "has_event": has_event,
        "detected": pd.notna(first_alarm) if has_event else np.nan,
        "delay_h": _hours(first_alarm - onset),
        "lead_time_h": _hours(event_start - first_alarm),
        "false_alarms": int((alarm_starts & normal).sum()),
        "normal_hours": float(normal.sum() * step_minutes / 60),
        "fa_hours_before_onset": fa_before,
    }


def warning_curve(reports: pd.DataFrame, thresholds_h=(0, 1, 6, 12, 24)) -> pd.Series:
    """Share of event instances caught with at least X hours of warning.
    Missed events count as failures at every threshold."""
    events = reports[reports["has_event"]]
    return pd.Series({h: float((events["lead_time_h"] >= h).mean()) for h in thresholds_h},
                     name="share_with_warning")


def false_alarms_per_day(reports: pd.DataFrame) -> float:
    days = reports["normal_hours"].sum() / 24
    return float(reports["false_alarms"].sum() / days) if days > 0 else np.nan