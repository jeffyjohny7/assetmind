from pathlib import Path
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS, MODEL_SENSORS_EXT
from assetmind.models.eval_anomaly_twotier import load_windows, evaluate, summarize, log_run, REPORTS

WINDOWS = Path("data/gold/3w_windows_real_ext_h.parquet")
V4_FEATURES = ("logstd", "change", "lvl1h", "logwav1h")
SCALE_FLOOR = 0.1
CONFIGS = {"v4_core_floor": ("core", MODEL_SENSORS),
           "v5a_ext_floor": ("ext", MODEL_SENSORS_EXT)}
QUANTILES = [0.99, 0.995, 0.999]
K = 5
CALIBRATION = "out_of_sample"

if __name__ == "__main__":
    w, meta = load_windows(WINDOWS)
    results = []
    for name, (sens_label, sensors) in CONFIGS.items():
        for q in QUANTILES:
            rep = evaluate(w, meta, q, K, CALIBRATION, V4_FEATURES, sensors, SCALE_FLOOR)
            rep.to_parquet(REPORTS / f"3w_lowo_{name}_q{q}_k{K}.parquet", index=False)
            metrics = summarize(rep)
            params = {"config": name, "sensors": sens_label, "sensor_list": ",".join(sensors),
                      "features": "v4", "scale_floor": SCALE_FLOOR,
                      "quantile": q, "k": K, "calibration": CALIBRATION}
            log_run(params, metrics)
            results.append({**params, **metrics})
            print(f"{name} q={q}: macro detection {metrics['detection_macro']:.2f}, "
                  f"false alarms/day {metrics['false_alarms_per_day']:.2f}")

    out = pd.DataFrame(results)
    out.to_parquet(REPORTS / "3w_sweep_floor.parquet", index=False)
    cols = ["config", "quantile", "detection_macro", "false_alarms_per_day",
            "fa_day_normal_files", "fa_day_event_files", "warn_ge_1h"]
    print("\n" + out[cols].round(3).to_string(index=False))