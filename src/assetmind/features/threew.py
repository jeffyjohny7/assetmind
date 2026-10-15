import pandas as pd


def add_hour_features(w: pd.DataFrame, sensors, lag: int = 60) -> pd.DataFrame:
    """Fixed 1-hour look-back features, computed within each instance.
    Assumes windows are spaced exactly 1 minute apart inside an instance."""
    w = w.sort_values(["instance_id", "window_end"]).reset_index(drop=True)
    steps = w.groupby("instance_id", observed=True)["window_end"].diff().dropna()
    assert (steps == pd.Timedelta(minutes=1)).all(), "expected a 1-minute step between windows"

    by_inst = w.groupby("instance_id", observed=True)
    for s in sensors:
        means = by_inst[f"{s}_mean"]
        w[f"{s}_lvl1h"] = w[f"{s}_mean"] - means.shift(lag)
        w[f"{s}_wav1h"] = means.transform(lambda x: x.rolling(lag, min_periods=lag).std())
    return w