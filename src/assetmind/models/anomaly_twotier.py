import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from assetmind.models.anomaly_baseline import RobustZDetector, FEATURE_BUILDERS

WELL_ONLY = {"level"}   # features that only make sense against a well's own history


def robust_stats(X: pd.DataFrame):
    med = X.median()
    scale = (1.4826 * (X - med).abs().median()).replace(0, np.nan)
    return med, scale


def _rank(ref_sorted: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Position of each value within a sorted reference, as a fraction in (0, 1). NaN stays NaN."""
    r = np.searchsorted(ref_sorted, values, side="right") / (len(ref_sorted) + 1)
    return np.where(np.isnan(values), np.nan, r)


class TwoTierDetector:
    """Per-well baseline when the well has enough earlier normal history, else the global detector.
    scale_floor: a well's spread per feature is at least scale_floor × a reference spread (0 = no floor).
    scoring (per-well tier):
      "zmax"       = max raw z over features
      "percentile" = max per-feature percentile
      "iforest"    = Isolation Forest on the per-feature z vector
      "ensemble"   = max of the percentile and iforest scores, each ranked against calibration
    Features in WELL_ONLY are used by the per-well tier only."""

    def __init__(self, sensors, quantile=0.995, k=3, min_baseline=120, calibration="out_of_sample",
                 features=("logstd", "change"), scale_floor=0.0, scoring="zmax", random_state=42):
        self.sensors = sensors
        self.well_features = tuple(features)
        global_features = tuple(f for f in features if f not in WELL_ONLY)
        self.global_ = RobustZDetector(sensors, quantile, k, global_features)
        self.quantile, self.k, self.min_baseline = quantile, k, min_baseline
        self.calibration, self.scale_floor, self.scoring = calibration, scale_floor, scoring
        self.random_state = random_state

    # ---------- per-well features and statistics ----------
    def _well_X(self, w: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f"{s}_{f}": FEATURE_BUILDERS[f](w, s)
                             for s in self.sensors for f in self.well_features}, index=w.index)

    def _well_stats(self, X: pd.DataFrame):
        med = X.median()
        scale = (1.4826 * (X - med).abs().median()).fillna(0)
        floor = (self.scale_floor * self.floor_scale_.reindex(X.columns)).fillna(0)
        return med, np.maximum(scale, floor).replace(0, np.nan)

    @staticmethod
    def _well_z(X: pd.DataFrame, med, scale) -> pd.DataFrame:
        return ((X - med) / scale).abs()

    # ---------- the scoring methods ----------
    def _score_percentile(self, Z: pd.DataFrame) -> pd.Series:
        cols = {}
        for c in Z.columns:
            ref = self.cal_sorted_.get(c)
            v = Z[c].to_numpy(dtype=float)
            cols[c] = np.full(len(v), np.nan) if ref is None or len(ref) == 0 else _rank(ref, v)
        return pd.DataFrame(cols, index=Z.index).max(axis=1, skipna=True)

    def _score_iforest(self, Z: pd.DataFrame) -> pd.Series:
        Z = Z[self.cal_columns_]
        score = pd.Series(-self.iforest_.score_samples(Z.fillna(0).to_numpy()), index=Z.index)
        score[Z.isna().all(axis=1)] = np.nan   # nothing valid in this window: no score
        return score

    def _combine(self, Z: pd.DataFrame) -> pd.Series:
        """Turn per-feature z-scores into one window score (higher = more anomalous)."""
        if self.scoring == "zmax":
            return Z.max(axis=1, skipna=True)
        if self.scoring == "percentile":
            return self._score_percentile(Z)
        if self.scoring == "iforest":
            return self._score_iforest(Z)
        # ensemble: rank each method's score against its own calibration distribution, take the max
        r_pct = _rank(self.ens_ref_["percentile"], self._score_percentile(Z).to_numpy(dtype=float))
        r_if = _rank(self.ens_ref_["iforest"], self._score_iforest(Z).to_numpy(dtype=float))
        return pd.Series(np.fmax(r_pct, r_if), index=Z.index)

    # ---------- calibration ----------
    def _cal_in_sample(self, g):
        X = self._well_X(g)
        return self._well_z(X, *self._well_stats(X))

    def _cal_out_of_sample(self, g):
        """Baseline from the well's earlier files, z-scores on its later files."""
        span = g.groupby("instance_id")["window_end"].agg(["min", "max"]).sort_values("min")
        if len(span) < 2:
            return None
        cut = span["min"].iloc[len(span) // 2]
        early = g[g["instance_id"].isin(span.index[span["max"] < cut])]
        late = g[g["instance_id"].isin(span.index[span["min"] >= cut])]
        if len(early) < self.min_baseline or late.empty:
            return None
        return self._well_z(self._well_X(late), *self._well_stats(self._well_X(early)))

    def fit(self, train_normal: pd.DataFrame) -> "TwoTierDetector":
        self.global_.fit(train_normal)
        wells = [g for _, g in train_normal.groupby("well", observed=True) if len(g) >= self.min_baseline]

        # Floor reference: pooled spread for movement features; typical within-well spread for level
        self.floor_scale_ = self.global_.scale_.copy()
        level_cols = [f"{s}_level" for s in self.sensors if "level" in self.well_features]
        if level_cols and wells:
            mads = pd.DataFrame([1.4826 * (X - X.median()).abs().median()
                                 for X in (self._well_X(g)[level_cols] for g in wells)])
            self.floor_scale_ = pd.concat([self.floor_scale_, mads.median()])

        parts = []
        if self.calibration == "out_of_sample":
            parts = [z for z in map(self._cal_out_of_sample, wells) if z is not None]
        self.calibration_used_ = "out_of_sample" if parts else "in_sample"
        if not parts:   # fallback: no well could be split in time
            parts = [self._cal_in_sample(g) for g in wells]

        Z = pd.concat(parts)
        self.cal_columns_ = list(Z.columns)
        if self.scoring in ("percentile", "ensemble"):
            self.cal_sorted_ = {c: np.sort(Z[c].dropna().to_numpy()) for c in Z.columns}
        if self.scoring in ("iforest", "ensemble"):
            self.iforest_ = IsolationForest(n_estimators=200, random_state=self.random_state).fit(
                Z.fillna(0).to_numpy())
        if self.scoring == "ensemble":
            self.ens_ref_ = {
                "percentile": np.sort(self._score_percentile(Z).dropna().to_numpy()),
                "iforest": np.sort(self._score_iforest(Z).dropna().to_numpy()),
            }
        self.well_threshold_ = float(np.nanquantile(self._combine(Z), self.quantile))
        return self

    # ---------- scoring ----------
    def score_instance(self, inst: pd.DataFrame, history: pd.DataFrame):
        """Returns (scores, threshold, tier). `history` = this well's earlier normal windows."""
        if len(history) >= self.min_baseline:
            Z = self._well_z(self._well_X(inst), *self._well_stats(self._well_X(history)))
            score = self._combine(Z)
            if score.notna().any():
                return score, self.well_threshold_, "well"
        return self.global_.score(inst), self.global_.threshold_, "global"

    def alarms_instance(self, inst: pd.DataFrame, history: pd.DataFrame):
        score, thr, tier = self.score_instance(inst, history)
        exceed = (score > thr).astype(int)
        alarm = exceed.rolling(self.k, min_periods=self.k).sum().eq(self.k)
        return alarm, tier