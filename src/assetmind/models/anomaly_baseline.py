import numpy as np
import pandas as pd

FEATURE_BUILDERS = {
    "logstd":   lambda w, s: np.log1p(w[f"{s}_std"]),
    "change":   lambda w, s: w[f"{s}_change"],
    "lvl1h":    lambda w, s: w[f"{s}_lvl1h"],
    "logwav1h": lambda w, s: np.log1p(w[f"{s}_wav1h"]),
}

class RobustZDetector:
    """Per-sensor anomaly scores from movement features (std, change).
    Window score = most abnormal valid sensor feature. Alarm = k consecutive windows above threshold."""

    def __init__(self, sensors, quantile=0.995, k=3, features=("logstd", "change")):
        self.sensors, self.quantile, self.k = sensors, quantile, k
        self.features = tuple(features)

    def _features(self, w: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f"{s}_{f}": FEATURE_BUILDERS[f](w, s)
                             for s in self.sensors for f in self.features}, index=w.index)

    def fit(self, normal: pd.DataFrame) -> "RobustZDetector":
        X = self._features(normal)
        self.median_ = X.median()
        self.scale_ = 1.4826 * (X - self.median_).abs().median()   # robust spread (MAD)
        self.scale_ = self.scale_.replace(0, np.nan)
        self.threshold_ = float(np.nanquantile(self.score(normal), self.quantile))
        return self

    def score(self, w: pd.DataFrame) -> pd.Series:
        z = ((self._features(w) - self.median_) / self.scale_).abs()
        return z.max(axis=1, skipna=True)   # NaN when no sensor is valid

    def alarms(self, w: pd.DataFrame) -> pd.Series:
        """w must be sorted by instance_id, window_end."""
        exceed = (self.score(w) > self.threshold_).astype(int)
        run = exceed.groupby(w["instance_id"], observed=True).transform(
            lambda s: s.rolling(self.k, min_periods=self.k).sum())
        return run.eq(self.k)