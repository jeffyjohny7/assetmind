from pathlib import Path
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS_EXT
from assetmind.models.eval_anomaly_twotier import load_windows, evaluate, summarize, log_run, REPORTS

WINDOWS = Path("data/gold/3w_windows_real_ext_h.parquet")
FEATURES = ("logstd", "change", "lvl1h", "logwav1h", "level")
SCALE_FLOOR = 0.1
SCORING = "percentile"
NAME = "v5b_ext"
QUANTILES = [0.99, 0.995, 0.999]
K = 5
CALIBRATION = "out_of_sample"

if __name__ == "__main__":
    w, meta = load_windows(WINDOWS)
    results = []
    for q in QUANTILES:
        rep = evaluate(w, meta, q, K, CALIBRATION, FEATURES, MODEL_SENSORS_EXT, SCALE_FLOOR, SCORING)
        rep.to_parquet(REPORTS / f"3w_lowo_{NAME}_q{q}_k{K}.parquet", index=False)
        metrics = summarize(rep)
        params = {"config": NAME, "sensors": "ext", "sensor_list": ",".join(MODEL_SENSORS_EXT),
                  "features": "v4+level", "scale_floor": SCALE_FLOOR, "scoring": SCORING,
                  "quantile": q, "k": K, "calibration": CALIBRATION}
        log_run(params, metrics)
        results.append({**params, **metrics})
        print(f"{NAME} q={q}: macro detection {metrics['detection_macro']:.2f}, "
              f"false alarms/day {metrics['false_alarms_per_day']:.2f}")

    out = pd.DataFrame(results)
    out.to_parquet(REPORTS / "3w_sweep_v5b.parquet", index=False)
    cols = ["config", "quantile", "detection_macro", "false_alarms_per_day",
            "fa_day_well", "fa_day_global", "warn_ge_1h"]
    print("\n" + out[cols].round(3).to_string(index=False))