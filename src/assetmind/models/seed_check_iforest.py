from pathlib import Path
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS_EXT
from assetmind.models.eval_anomaly_twotier import load_windows, evaluate, summarize, REPORTS

WINDOWS = Path("data/gold/3w_windows_real_ext_h.parquet")
FEATURES = ("logstd", "change", "lvl1h", "logwav1h")
SCALE_FLOOR, K = 0.1, 5
# (name, scoring, quantile, saved report for seed 42)
CONFIGS = [("v7_iforest", "iforest", 0.995, "3w_lowo_v7_ext_iforest_q0.995_k5.parquet"),
           ("v7_iforest", "iforest", 0.999, "3w_lowo_v7_ext_iforest_q0.999_k5.parquet"),
           ("v8_ensemble", "ensemble", 0.99, "3w_lowo_v8_ext_ensemble_q0.99_k5.parquet")]
NEW_SEEDS = [7, 123]


def row(name, q, seed, rep):
    m = summarize(rep)
    ev = rep[rep["has_event"]].assign(detected=lambda d: d["detected"].astype(bool))
    hits = ev.groupby("event_class")["detected"].sum()
    return {"config": name, "quantile": q, "seed": seed,
            "macro": m["detection_macro"], "fa_day": m["false_alarms_per_day"],
            "instability_files": int(hits.get(4, 0)), "restriction_files": int(hits.get(6, 0)),
            "scaling_files": int(hits.get(7, 0))}


if __name__ == "__main__":
    w, meta = load_windows(WINDOWS)
    rows = []
    for name, scoring, q, saved in CONFIGS:
        rows.append(row(name, q, 42, pd.read_parquet(REPORTS / saved)))
        for seed in NEW_SEEDS:
            rep = evaluate(w, meta, q, K, "out_of_sample", FEATURES, MODEL_SENSORS_EXT,
                           SCALE_FLOOR, scoring, seed)
            rows.append(row(name, q, seed, rep))
            print(f"{name} q={q} seed={seed}: macro {rows[-1]['macro']:.3f}, FA/day {rows[-1]['fa_day']:.3f}")

    out = pd.DataFrame(rows)
    out.to_parquet(REPORTS / "3w_seed_check_iforest.parquet", index=False)
    print("\n" + out.round(3).to_string(index=False))
    print("\nSpread across seeds (min, max):")
    print(out.groupby(["config", "quantile"])[["macro", "fa_day", "instability_files"]]
             .agg(["min", "max"]).round(3).to_string())