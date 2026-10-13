from pathlib import Path
import mlflow
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS
from assetmind.models.anomaly_baseline import RobustZDetector
from assetmind.evaluation.event_metrics import instance_report, warning_curve, false_alarms_per_day

WINDOWS = Path("data/gold/3w_windows_real.parquet")
REPORTS = Path("reports")
TRANSIENT_CLASSES = [1, 2, 5, 6, 7, 8, 9]   # event types that have a transient phase
QUANTILE, K = 0.995, 3

if __name__ == "__main__":
    w = pd.read_parquet(WINDOWS).sort_values(["instance_id", "window_end"]).reset_index(drop=True)
    w["phase"] = w["phase"].astype(str)

    rows = []
    for well in w["well"].unique():
        train_normal = w[(w["well"] != well) & (w["phase"] == "normal")]
        test = w[w["well"] == well].copy()
        det = RobustZDetector(MODEL_SENSORS, QUANTILE, K).fit(train_normal)
        test["alarm"] = det.alarms(test)
        test["scored"] = det.score(test).notna()

        for inst, g in test.groupby("instance_id", observed=True):
            r = instance_report(g)
            r.update(instance_id=inst, well=well, event_class=int(g["event_class"].iloc[0]),
                     scored_share=float(g["scored"].mean()))
            rows.append(r)

    rep = pd.DataFrame(rows)
    rep.to_parquet(REPORTS / "3w_lowo_baseline.parquet", index=False)

    ev = rep[rep["has_event"]].assign(detected=lambda d: d["detected"].astype(bool))
    per_class = ev.groupby("event_class").agg(
        files=("instance_id", "size"), wells=("well", "nunique"),
        detected=("detected", "mean"), median_delay_h=("delay_h", "median"),
        median_lead_h=("lead_time_h", "median"), scored=("scored_share", "mean"))
    curve = warning_curve(rep[rep["event_class"].isin(TRANSIENT_CLASSES)])
    fa_day = false_alarms_per_day(rep)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("3w-anomaly-lowo")
    with mlflow.start_run(run_name=f"robust_z_q{QUANTILE}_k{K}_v2"):
        mlflow.log_params({"model": "robust_z", "quantile": QUANTILE, "k": K,
                           "sensors": ",".join(MODEL_SENSORS), "features": "logstd,change"})
        mlflow.log_metrics({"detection_rate": float(ev["detected"].mean()),
                            "false_alarms_per_day": fa_day,
                            **{f"warn_ge_{h}h": v for h, v in curve.items()}})

    print(per_class.round(2).to_string())
    print("\nWarning-time curve:\n", curve.round(2).to_string())
    print(f"\nFalse alarms per well per day: {fa_day:.2f}")