from pathlib import Path
import mlflow
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS
from assetmind.models.anomaly_twotier import TwoTierDetector
from assetmind.evaluation.event_metrics import instance_report, warning_curve, false_alarms_per_day

WINDOWS = Path("data/gold/3w_windows_real.parquet")
REPORTS = Path("reports")
MIN_BASELINE = 120
TRANSIENT_CLASSES = [1, 2, 5, 6, 7, 8, 9]


def load_windows(path=WINDOWS):
    w = pd.read_parquet(path).sort_values(["instance_id", "window_end"]).reset_index(drop=True)
    for col in ["phase", "instance_id", "well"]:
        w[col] = w[col].astype(str)
    meta = w.groupby("instance_id").agg(well=("well", "first"), end=("window_end", "max"))
    meta["start"] = pd.to_datetime(meta.index.str.split("_").str[-1], format="%Y%m%d%H%M%S")
    return w, meta


def evaluate(w, meta, quantile, k, calibration, features=("logstd", "change"),
             sensors=MODEL_SENSORS, scale_floor=0.0, scoring="zmax", random_state=42) -> pd.DataFrame:
    """Leave-one-well-out evaluation with the no-future rule. One row per instance."""
    rows = []
    for well in w["well"].unique():
        det = TwoTierDetector(sensors, quantile, k, MIN_BASELINE, calibration,
                              features, scale_floor, scoring, random_state).fit(
            w[(w["well"] != well) & (w["phase"] == "normal")])
        test = w[w["well"] == well]
        well_meta = meta[meta["well"] == well]
        for inst, g in test.groupby("instance_id"):
            earlier = well_meta.index[well_meta["end"] < meta.loc[inst, "start"]]   # no-future rule
            history = test[test["instance_id"].isin(earlier) & (test["phase"] == "normal")]
            alarm, tier = det.alarms_instance(g, history)
            r = instance_report(g.assign(alarm=alarm.to_numpy()))
            r.update(instance_id=inst, well=well, event_class=int(g["event_class"].iloc[0]), tier=tier)
            rows.append(r)
    return pd.DataFrame(rows)


def summarize(rep: pd.DataFrame) -> dict:
    ev = rep[rep["has_event"]].assign(detected=lambda d: d["detected"].astype(bool))
    curve = warning_curve(rep[rep["event_class"].isin(TRANSIENT_CLASSES)])
    return {
        "detection_rate": float(ev["detected"].mean()),
        "detection_macro": float(ev.groupby("event_class")["detected"].mean().mean()),
        "false_alarms_per_day": false_alarms_per_day(rep),
        "fa_day_well": false_alarms_per_day(rep[rep["tier"] == "well"]),
        "fa_day_global": false_alarms_per_day(rep[rep["tier"] == "global"]),
        "fa_day_normal_files": false_alarms_per_day(rep[rep["event_class"] == 0]),
        "fa_day_event_files": false_alarms_per_day(rep[rep["event_class"] != 0]),
        **{f"warn_ge_{h}h": v for h, v in curve.items()},
    }


def log_run(params: dict, metrics: dict):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("3w-anomaly-lowo")
    name = (f"twotier_{params.get('sensors', 'core')}_{params.get('features', 'v3')}"
            f"_f{params.get('scale_floor', 0.0)}_{params.get('scoring', 'zmax')}"
            f"_{params['calibration']}_q{params['quantile']}_k{params['k']}")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": "two_tier_robust_z", "min_baseline": MIN_BASELINE,
                           "sensor_list": ",".join(MODEL_SENSORS), **params})
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    params = {"quantile": 0.995, "k": 3, "calibration": "out_of_sample"}
    w, meta = load_windows()
    rep = evaluate(w, meta, **params)
    rep.to_parquet(REPORTS / f"3w_lowo_twotier_{params['calibration']}.parquet", index=False)
    metrics = summarize(rep)
    log_run(params, metrics)
    for key, value in metrics.items():
        print(f"{key:22s} {value:.3f}")