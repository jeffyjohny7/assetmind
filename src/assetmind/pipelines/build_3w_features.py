from pathlib import Path
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS_EXT
from assetmind.features.threew import add_hour_features


def build(sensors, src: Path, out: Path):
    w = add_hour_features(pd.read_parquet(src), sensors)
    w.to_parquet(out, index=False)
    new = [c for c in w.columns if c.endswith(("_lvl1h", "_wav1h"))]
    print("Shape:", w.shape)
    print("Share of windows with each hour feature:\n", w[new].notna().mean().round(2).to_string())


if __name__ == "__main__":
    build(MODEL_SENSORS_EXT, Path("data/gold/3w_windows_real_ext.parquet"),
          Path("data/gold/3w_windows_real_ext_h.parquet"))