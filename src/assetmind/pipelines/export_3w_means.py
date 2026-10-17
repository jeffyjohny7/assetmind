from pathlib import Path
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS_EXT

IN = Path("data/gold/3w_windows_real_ext_h.parquet")
OUT = Path("data/gold/3w_means_ext.parquet")

if __name__ == "__main__":
    cols = ["instance_id", "window_end"] + [f"{s}_mean" for s in MODEL_SENSORS_EXT]
    w = pd.read_parquet(IN, columns=cols)
    w["instance_id"] = w["instance_id"].astype(str)
    w = w.sort_values(["instance_id", "window_end"])
    w.to_parquet(OUT, index=False)
    print(w.shape, f"{OUT.stat().st_size / 1e6:.1f} MB")