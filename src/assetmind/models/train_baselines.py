from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from assetmind.evaluation.metrics import rmse, nasa_score

GOLD, REPORTS = Path("data/gold"), Path("reports")
NON_FEATURES = {"unit", "rul", "rul_capped"}
RUL_CAP = 125


def build_models():
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "lightgbm": LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        ),
    }


def simple_params(model):
    """Keep only plain values; skip nested objects that bloat the log."""
    return {k: v for k, v in model.get_params().items()
            if isinstance(v, (int, float, str, bool, type(None)))}


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cmapss-fd001-baselines")

    train = pd.read_parquet(GOLD / "cmapss_train_FD001.parquet")
    test = pd.read_parquet(GOLD / "cmapss_test_FD001.parquet")
    features = [c for c in train.columns if c not in NON_FEATURES]
    X, y, groups = train[features], train["rul_capped"], train["unit"]
    test_last = test.loc[test.groupby("unit")["cycle"].idxmax()]

    preds = test_last[["unit", "cycle", "rul"]].rename(
        columns={"cycle": "n_cycles", "rul": "true_rul"}).reset_index(drop=True)
    results = []

    for name, model in build_models().items():
        with mlflow.start_run(run_name=name):
            cv_scores = []
            for tr_idx, va_idx in GroupKFold(n_splits=5).split(X, y, groups):
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                cv_scores.append(rmse(y.iloc[va_idx], model.predict(X.iloc[va_idx])))

            model.fit(X, y)
            pred = np.clip(model.predict(test_last[features]), 0, None)
            preds[f"pred_{name}"] = pred

            metrics = {
                "cv_rmse": float(np.mean(cv_scores)),
                "test_rmse": rmse(test_last["rul"], pred),
                "test_rmse_capped": rmse(test_last["rul"].clip(upper=RUL_CAP), pred),
                "test_nasa": nasa_score(test_last["rul"], pred),
            }
            mlflow.log_params({"model": name, "rul_cap": RUL_CAP,
                               "n_features": len(features), **simple_params(model)})
            mlflow.log_metrics(metrics)
            results.append({"model": name, **metrics})

    out = REPORTS / "predictions_FD001.parquet"
    preds.to_parquet(out, index=False)
    print(pd.DataFrame(results).round(2).to_string(index=False))
    print(f"Predictions saved to {out}")