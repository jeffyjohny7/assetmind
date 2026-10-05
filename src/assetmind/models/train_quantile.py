import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from assetmind.evaluation.metrics import rmse, nasa_score
from assetmind.models.train_baselines import GOLD, REPORTS, NON_FEATURES, RUL_CAP

QUANTILES = [0.1, 0.3, 0.5, 0.9]


def quantile_model(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cmapss-fd001-baselines")

    train = pd.read_parquet(GOLD / "cmapss_train_FD001.parquet")
    test = pd.read_parquet(GOLD / "cmapss_test_FD001.parquet")
    features = [c for c in train.columns if c not in NON_FEATURES]
    test_last = test.loc[test.groupby("unit")["cycle"].idxmax()]
    truth = test_last["rul"].to_numpy()

    preds = {q: np.clip(quantile_model(q).fit(train[features], train["rul_capped"])
                        .predict(test_last[features]), 0, None) for q in QUANTILES}

    capped = np.minimum(truth, RUL_CAP)
    coverage = float(np.mean((capped >= preds[0.1]) & (capped <= preds[0.9])))
    width = float(np.mean(preds[0.9] - preds[0.1]))

    rows = []
    for q in [0.3, 0.5]:
        p = preds[q]
        metrics = {"test_rmse": rmse(truth, p), "test_nasa": nasa_score(truth, p),
                   "late_count": int((p > truth).sum()),
                   "interval_coverage_p10_p90": coverage, "interval_width": width}
        with mlflow.start_run(run_name=f"lightgbm_q{int(q*100)}"):
            mlflow.log_params({"model": "lightgbm_quantile", "alpha": q, "rul_cap": RUL_CAP})
            mlflow.log_metrics(metrics)
        rows.append({"point": f"P{int(q*100)}", **metrics})

    print(pd.DataFrame(rows).round(2).to_string(index=False))

    out = test_last[["unit", "cycle", "rul"]].reset_index(drop=True)
    for q in QUANTILES:
        out[f"p{int(q*100)}"] = preds[q]
    out.to_parquet(REPORTS / "quantile_predictions_FD001.parquet", index=False)