from pathlib import Path
import numpy as np
import pandas as pd

CORE = ["T-TPT", "P-TPT", "P-ANULAR", "P-PDG"]
MODEL_SENSORS = ["P-TPT", "T-TPT", "P-ANULAR"]   # P-PDG excluded: usable in only 14% of files
NEVER_PRESENT = ["T-MON-CKP", "P-MON-SDV-P", "P-JUS-BS", "QBS", "PT-P"]
TRANSIENT_OFFSET = 100


def parse_path(path: Path) -> dict:
    """Metadata from a 3W file path: class folder, source type and well."""
    stem = path.stem
    if stem.startswith("WELL-"):
        source, well = "real", stem.split("_")[0]
    elif stem.startswith("SIMULATED"):
        source, well = "simulated", None
    else:
        source, well = "drawn", None
    return {"instance_id": f"{path.parent.name}/{stem}",
            "event_class": int(path.parent.name), "source": source, "well": well}


def read_instance(path: Path, sensors=CORE) -> pd.DataFrame:
    """Read one instance: selected sensors plus label, unlabeled rows removed."""
    df = pd.read_parquet(path, columns=list(sensors) + ["class"])
    df = df[df["class"].notna()]
    df = df.astype({s: "float32" for s in sensors})
    df["label"] = df.pop("class").astype(int)
    return df


def make_windows(df: pd.DataFrame, sensors, window=300, step=60) -> pd.DataFrame:
    """Sliding windows over a 1 Hz instance. Each window is labeled by its last second.
    Each sensor gets a status per window; features from non-ok windows are set to NaN."""
    roll = df[sensors].rolling(window, min_periods=window)
    feats = pd.concat({"mean": roll.mean(), "std": roll.std(),
                       "min": roll.min(), "max": roll.max()}, axis=1)
    feats.columns = [f"{s}_{stat}" for stat, s in feats.columns]
    change = df[sensors] - df[sensors].shift(window - 1)
    for s in sensors:
        feats[f"{s}_change"] = change[s]

    for s in sensors:
        lo, hi = feats[f"{s}_min"], feats[f"{s}_max"]
        status = np.select(
            [feats[f"{s}_mean"].isna(), lo.eq(0) & hi.eq(0), lo.eq(hi)],
            ["missing", "zero", "frozen"], default="ok")
        feats[f"{s}_status"] = status
        cols = [f"{s}_{k}" for k in ("mean", "std", "min", "max", "change")]
        feats.loc[status != "ok", cols] = np.nan

    feats["label"] = df["label"]
    feats["phase"] = np.select([df["label"].eq(0), df["label"].ge(TRANSIENT_OFFSET)],
                               ["normal", "transient"], default="event")
    return feats.iloc[window - 1 :: step].reset_index(names="window_end")