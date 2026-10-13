import numpy as np
import pandas as pd


class RobustZDetector:
    """Per-sensor anomaly scores from movement features (std, change).
    Window score = most abnormal valid sensor feature. Alarm = k consecutive windows above threshold."""

    def __init__(self, sensors, quantile=0.995, k=3):
        self.sensors, self.quantile, self.k = sensors, quantile, k

    def _features(self, w: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for s in self.sensors:
            cols[f"{s}_logstd"] = np.log1p(w[f"{s}_std"])   # std is skewed; log tames the long tail
            cols[f"{s}_change"] = w[f"{s}_change"]
        return pd.DataFrame(cols, index=w.index)

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