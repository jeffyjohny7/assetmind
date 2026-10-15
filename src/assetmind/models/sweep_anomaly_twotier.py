from pathlib import Path
import pandas as pd
from assetmind.models.eval_anomaly_twotier import load_windows, evaluate, summarize, log_run, REPORTS

WINDOWS_V4 = Path("data/gold/3w_windows_real_v4.parquet")
FEATURE_SETS = {"v3": ("logstd", "change"),
                "v4": ("logstd", "change", "lvl1h", "logwav1h")}
QUANTILES = [0.99, 0.995, 0.999]
K = 5
CALIBRATION = "out_of_sample"

if __name__ == "__main__":
    w, meta = load_windows(WINDOWS_V4)
    results = []
    for name, feats in FEATURE_SETS.items():
        for q in QUANTILES:
            metrics = summarize(evaluate(w, meta, q, K, CALIBRATION, feats))
            params = {"features": name, "quantile": q, "k": K, "calibration": CALIBRATION}
            log_run(params, metrics)
            results.append({**params, **metrics})
            print(f"{name} q={q}: macro detection {metrics['detection_macro']:.2f}, "
                  f"false alarms/day {metrics['false_alarms_per_day']:.2f}")

    out = pd.DataFrame(results)
    out.to_parquet(REPORTS / "3w_sweep_v3_vs_v4.parquet", index=False)
    cols = ["features", "quantile", "detection_macro", "false_alarms_per_day",
            "fa_day_well", "fa_day_global", "warn_ge_1h"]
    print("\n" + out[cols].round(3).to_string(index=False))