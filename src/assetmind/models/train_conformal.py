import mlflow
import numpy as np
import pandas as pd
from assetmind.models.train_baselines import GOLD, NON_FEATURES
from assetmind.models.train_quantile import quantile_model

TARGET, N_CAL, MIN_HISTORY, SEED = 0.80, 30, 30, 3


def random_cut(df: pd.DataFrame, rng) -> pd.DataFrame:
    """One random cut point per engine, mimicking how test engines are truncated."""
    rows = []
    for _, g in df.groupby("unit"):
        life = g["cycle"].max()
        cut = rng.integers(min(MIN_HISTORY, life), life + 1)
        rows.append(g[g["cycle"] == cut])
    return pd.concat(rows)


def coverage_and_width(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cmapss-fd001-baselines")

    train = pd.read_parquet(GOLD / "cmapss_train_FD001.parquet")
    test = pd.read_parquet(GOLD / "cmapss_test_FD001.parquet")
    features = [c for c in train.columns if c not in NON_FEATURES]

    # 1. Split training engines: 70 to fit, 30 to calibrate
    cal_units = rng.choice(train["unit"].unique(), N_CAL, replace=False)
    fit_df = train[~train["unit"].isin(cal_units)]
    cal_df = random_cut(train[train["unit"].isin(cal_units)], rng)

    # 2. Train P10 and P90 on the 70 engines only
    lo_model = quantile_model(0.1).fit(fit_df[features], fit_df["rul_capped"])
    hi_model = quantile_model(0.9).fit(fit_df[features], fit_df["rul_capped"])

    # 3. Miss scores on the calibration engines
    y_cal = cal_df["rul_capped"].to_numpy()
    scores = np.maximum(lo_model.predict(cal_df[features]) - y_cal,
                        y_cal - hi_model.predict(cal_df[features]))
    n = len(scores)
    k = min(int(np.ceil((n + 1) * TARGET)), n)
    widen = float(np.sort(scores)[k - 1])

    # 4. Final exam: compare raw vs calibrated intervals on the test set
    test_last = test.loc[test.groupby("unit")["cycle"].idxmax()]
    y_test = test_last["rul_capped"].to_numpy()
    raw_lo, raw_hi = lo_model.predict(test_last[features]), hi_model.predict(test_last[features])

    raw_cov, raw_w = coverage_and_width(y_test, raw_lo, raw_hi)
    cal_cov, cal_w = coverage_and_width(y_test, raw_lo - widen, raw_hi + widen)

    with mlflow.start_run(run_name="lightgbm_conformal_p10_p90"):
        mlflow.log_params({"target_coverage": TARGET, "n_cal_engines": N_CAL, "seed": SEED})
        mlflow.log_metrics({"widen_cycles": widen, "raw_coverage": raw_cov, "raw_width": raw_w,
                            "conformal_coverage": cal_cov, "conformal_width": cal_w})

    print(f"Widen each side by: {widen:.2f} cycles")
    print(f"Raw intervals:       coverage={raw_cov:.2f}, width={raw_w:.1f}")
    print(f"Conformal intervals: coverage={cal_cov:.2f}, width={cal_w:.1f}")