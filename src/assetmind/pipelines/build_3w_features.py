from pathlib import Path
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS
from assetmind.features.threew import add_hour_features

IN = Path("data/gold/3w_windows_real.parquet")
OUT = Path("data/gold/3w_windows_real_v4.parquet")

if __name__ == "__main__":
    w = add_hour_features(pd.read_parquet(IN), MODEL_SENSORS)
    w.to_parquet(OUT, index=False)
    new = [c for c in w.columns if c.endswith(("_lvl1h", "_wav1h"))]
    print("Shape:", w.shape)
    print("Share of windows with each new feature:\n", w[new].notna().mean().round(2).to_string())