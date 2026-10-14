import pandas as pd
from assetmind.models.eval_anomaly_twotier import load_windows, evaluate, summarize, log_run, REPORTS

QUANTILES = [0.99, 0.995, 0.999]
KS = [1, 3, 5]
CALIBRATION = "out_of_sample"

if __name__ == "__main__":
    w, meta = load_windows()
    results = []
    for q in QUANTILES:
        for k in KS:
            params = {"quantile": q, "k": k, "calibration": CALIBRATION}
            metrics = summarize(evaluate(w, meta, **params))
            log_run(params, metrics)
            results.append({**params, **metrics})
            print(f"q={q} k={k}: macro detection {metrics['detection_macro']:.2f}, "
                  f"false alarms/day {metrics['false_alarms_per_day']:.2f}")

    out = pd.DataFrame(results)
    out.to_parquet(REPORTS / "3w_sweep_twotier.parquet", index=False)
    cols = ["quantile", "k", "detection_rate", "detection_macro",
            "false_alarms_per_day", "fa_day_well", "fa_day_global", "warn_ge_1h"]
    print("\n" + out[cols].round(3).to_string(index=False))