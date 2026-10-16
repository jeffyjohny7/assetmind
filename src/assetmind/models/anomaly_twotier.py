import numpy as np
import pandas as pd
from assetmind.models.anomaly_baseline import RobustZDetector


def robust_stats(X: pd.DataFrame):
    med = X.median()
    scale = (1.4826 * (X - med).abs().median()).replace(0, np.nan)
    return med, scale


class TwoTierDetector:
    """Per-well baseline when the well has enough earlier normal history, else the global detector.
    scale_floor: a well's spread per feature is at least scale_floor × the global spread (0 = no floor)."""

    def __init__(self, sensors, quantile=0.995, k=3, min_baseline=120,
                 calibration="out_of_sample", features=("logstd", "change"), scale_floor=0.0):
        self.global_ = RobustZDetector(sensors, quantile, k, features)
        self.quantile, self.k, self.min_baseline = quantile, k, min_baseline
        self.calibration, self.scale_floor = calibration, scale_floor

    # ---------- per-well statistics with a minimum spread ----------
    def _well_stats(self, X: pd.DataFrame):
        med = X.median()
        scale = (1.4826 * (X - med).abs().median()).fillna(0)
        floor = (self.scale_floor * self.global_.scale_).fillna(0)
        return med, np.maximum(scale, floor).replace(0, np.nan)

    def _z(self, X: pd.DataFrame, med, scale) -> pd.Series:
        return ((X - med) / scale).abs().max(axis=1, skipna=True)

    # ---------- calibration of the per-well threshold ----------
    def _well_scores_in_sample(self, g):
        X = self.global_._features(g)
        return self._z(X, *self._well_stats(X))

    def _well_scores_out_of_sample(self, g):
        """Baseline from the well's earlier files, scores on its later files."""
        span = g.groupby("instance_id")["window_end"].agg(["min", "max"]).sort_values("min")
        if len(span) < 2:
            return None
        cut = span["min"].iloc[len(span) // 2]
        early = g[g["instance_id"].isin(span.index[span["max"] < cut])]
        late = g[g["instance_id"].isin(span.index[span["min"] >= cut])]
        if len(early) < self.min_baseline or late.empty:
            return None
        return self._z(self.global_._features(late), *self._well_stats(self.global_._features(early)))

    def fit(self, train_normal: pd.DataFrame) -> "TwoTierDetector":
        self.global_.fit(train_normal)
        wells = [g for _, g in train_normal.groupby("well", observed=True) if len(g) >= self.min_baseline]

        scores = []
        if self.calibration == "out_of_sample":
            scores = [s for s in map(self._well_scores_out_of_sample, wells) if s is not None]
        self.calibration_used_ = "out_of_sample" if scores else "in_sample"
        if not scores:   # fallback: no well could be split in time
            scores = [self._well_scores_in_sample(g) for g in wells]

        self.well_threshold_ = float(np.nanquantile(pd.concat(scores), self.quantile))
        return self

    # ---------- scoring ----------
    def score_instance(self, inst: pd.DataFrame, history: pd.DataFrame):
        """Returns (scores, threshold, tier). `history` = this well's earlier normal windows."""
        if len(history) >= self.min_baseline:
            z = self._z(self.global_._features(inst), *self._well_stats(self.global_._features(history)))
            if z.notna().any():
                return z, self.well_threshold_, "well"
        return self.global_.score(inst), self.global_.threshold_, "global"

    def alarms_instance(self, inst: pd.DataFrame, history: pd.DataFrame):
        score, thr, tier = self.score_instance(inst, history)
        exceed = (score > thr).astype(int)
        alarm = exceed.rolling(self.k, min_periods=self.k).sum().eq(self.k)
        return alarm, tier