import numpy as np
import pandas as pd


class CmapssFeatureBuilder:
    """Learns which sensors to keep from training data, then builds features."""

    def __init__(self, rul_cap=125, window=10, baseline_n=20, std_tol=1e-8):
        self.rul_cap = rul_cap
        self.window = window
        self.baseline_n = baseline_n
        self.std_tol = std_tol

    def fit(self, train: pd.DataFrame) -> "CmapssFeatureBuilder":
        candidates = [c for c in train.columns if c.startswith("s_")]
        self.sensors_ = [c for c in candidates if train[c].std() > self.std_tol]
        self.dropped_ = sorted(set(candidates) - set(self.sensors_))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        parts = [self._engine_features(g) for _, g in df.groupby("unit", sort=False)]
        out = pd.concat([df[["unit", "cycle", "rul"]], pd.concat(parts)], axis=1)
        out["rul_capped"] = out["rul"].clip(upper=self.rul_cap)
        return out

    def _engine_features(self, g: pd.DataFrame) -> pd.DataFrame:
        w, feats = self.window, {}
        t = g["cycle"].astype(float)
        t_mean = t.rolling(w, min_periods=2).mean()
        t_var = (t * t).rolling(w, min_periods=2).mean() - t_mean**2

        for s in self.sensors_:
            y = g[s]
            baseline = y.iloc[: self.baseline_n].mean()
            y_mean = y.rolling(w, min_periods=1).mean()
            cov = (t * y).rolling(w, min_periods=2).mean() - t_mean * y.rolling(w, min_periods=2).mean()

            feats[s] = y
            feats[f"{s}_dev"] = y - baseline
            feats[f"{s}_absdev"] = (y - baseline).abs()
            feats[f"{s}_mean{w}"] = y_mean
            feats[f"{s}_slope{w}"] = cov / t_var
        return pd.DataFrame(feats, index=g.index)