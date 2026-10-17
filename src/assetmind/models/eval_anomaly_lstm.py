import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS_EXT
from assetmind.models.anomaly_twotier import TwoTierDetector
from assetmind.models.anomaly_lstm import to_input, make_sequences, train_lstm, seq_errors, window_scores
from assetmind.models.eval_anomaly_twotier import load_windows, summarize, log_run, REPORTS, MIN_BASELINE
from assetmind.evaluation.event_metrics import instance_report

WINDOWS = Path("data/gold/3w_windows_real_ext_h.parquet")
FEATURES = ("logstd", "change", "lvl1h", "logwav1h")
SCALE_FLOOR = 0.1
QUANTILES = [0.99, 0.995, 0.999, 0.9995, 0.9999]
K = 5
CAL_SHARE = 0.3   # share of training wells held out to set the threshold
SEEDS = [int(s) for s in os.environ.get("LSTM_SEEDS", "42").split(",")]


def run_name(seed: int) -> str:
    return "v9_ext_lstm" if seed == 42 else f"v9_ext_lstm_s{seed}"


def score_fold(w, meta, well, seed):
    """Train on all wells except `well`; return per-window scores for `well` plus threshold references."""
    train_normal = w[(w["well"] != well) & (w["phase"] == "normal")]
    det = TwoTierDetector(MODEL_SENSORS_EXT, 0.995, K, MIN_BASELINE, "out_of_sample",
                          FEATURES, SCALE_FLOOR, "zmax").fit(train_normal)
    cols = det.cal_columns_

    # Normal z-vectors per training well: later files scored against earlier files
    cal = {}
    for wname, g in train_normal.groupby("well"):
        if len(g) >= MIN_BASELINE:
            z = det._cal_out_of_sample(g)
            if z is not None:
                cal[wname] = z

    def sequences(z):
        inst = w.loc[z.index, "instance_id"].to_numpy()
        return [make_sequences(to_input(part, cols)) for _, part in z.groupby(inst)]

    # 70% of training wells train the LSTM; 30% (unseen by it) set the threshold
    names = list(cal)
    np.random.default_rng(seed).shuffle(names)
    n_ref = max(1, int(len(names) * CAL_SHARE))
    ref_names, fit_names = names[:n_ref], names[n_ref:]
    model = train_lstm(np.concatenate([s for n in fit_names for s in sequences(cal[n])]), seed=seed)
    ref_err = np.sort(np.concatenate([seq_errors(model, s) for n in ref_names for s in sequences(cal[n])]))
    global_ref = np.sort(det.global_.score(train_normal).dropna().to_numpy())

    rows = []
    test, wm = w[w["well"] == well], meta[meta["well"] == well]
    for inst, g in test.groupby("instance_id"):
        earlier = wm.index[wm["end"] < meta.loc[inst, "start"]]   # no-future rule
        hist = test[test["instance_id"].isin(earlier) & (test["phase"] == "normal")]
        tier, s = "global", None
        if len(hist) >= MIN_BASELINE:
            Z = det._well_z(det._well_X(g), *det._well_stats(det._well_X(hist)))
            s = window_scores(model, to_input(Z, cols))
            s[Z.isna().all(axis=1).to_numpy()] = np.nan
            if np.isfinite(s).any():
                tier = "well"
        if tier == "global":
            s = det.global_.score(g).to_numpy()
        rows.append(pd.DataFrame({"instance_id": inst, "well": well,
                                  "event_class": g["event_class"].to_numpy(),
                                  "window_end": g["window_end"].to_numpy(),
                                  "phase": g["phase"].to_numpy(), "score": s, "tier": tier}))
    return pd.concat(rows), ref_err, global_ref


def apply_threshold(folds, q, k) -> pd.DataFrame:
    reports = []
    for scores, ref_err, global_ref in folds:
        thr = {"well": np.quantile(ref_err, q) if len(ref_err) else np.inf,
               "global": np.quantile(global_ref, q)}
        for inst, g in scores.groupby("instance_id"):
            tier = g["tier"].iloc[0]
            exceed = (g["score"] > thr[tier]).astype(int)
            alarm = exceed.rolling(k, min_periods=k).sum().eq(k)
            r = instance_report(g.assign(alarm=alarm.to_numpy()))
            r.update(instance_id=inst, well=g["well"].iloc[0],
                     event_class=int(g["event_class"].iloc[0]), tier=tier)
            reports.append(r)
    return pd.DataFrame(reports)


def run_seed(w, meta, seed) -> pd.DataFrame:
    name = run_name(seed)
    cache = REPORTS / f"{name}_fold_scores.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            folds = pickle.load(f)
        print(f"[seed {seed}] loaded cached fold scores")
    else:
        wells = w["well"].unique()
        folds = []
        for i, well in enumerate(wells, 1):
            folds.append(score_fold(w, meta, well, seed))
            print(f"[seed {seed}] {i}/{len(wells)} wells scored")
        with open(cache, "wb") as f:
            pickle.dump(folds, f)

    results = []
    for q in QUANTILES:
        rep = apply_threshold(folds, q, K)
        rep.to_parquet(REPORTS / f"3w_lowo_{name}_q{q}_k{K}.parquet", index=False)
        metrics = summarize(rep)
        params = {"config": name, "sensors": "ext", "sensor_list": ",".join(MODEL_SENSORS_EXT),
                  "features": "v4", "scale_floor": SCALE_FLOOR, "scoring": "lstm_ae", "seed": seed,
                  "quantile": q, "k": K, "calibration": "held_out_wells"}
        log_run(params, metrics)
        results.append({**params, **metrics})
        print(f"[seed {seed}] q={q}: macro detection {metrics['detection_macro']:.3f}, "
              f"false alarms/day {metrics['false_alarms_per_day']:.3f}")

    out = pd.DataFrame(results)
    out.to_parquet(REPORTS / f"3w_sweep_{name}.parquet", index=False)
    return out


if __name__ == "__main__":
    w, meta = load_windows(WINDOWS)
    all_results = pd.concat([run_seed(w, meta, seed) for seed in SEEDS], ignore_index=True)

    cols = ["seed", "quantile", "detection_macro", "false_alarms_per_day", "fa_day_well", "warn_ge_1h"]
    print("\n" + all_results[cols].round(3).assign(quantile=all_results["quantile"]).to_string(index=False))