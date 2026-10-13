from pathlib import Path
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS, parse_path, read_instance, make_windows

W3 = Path("../3W/dataset")
OUT = Path("data/gold/3w_windows_real.parquet")

if __name__ == "__main__":
    files = sorted(W3.glob("[0-9]/WELL-*.parquet"))
    parts = []
    for i, path in enumerate(files, 1):
        w = make_windows(read_instance(path, MODEL_SENSORS), MODEL_SENSORS)
        for key, value in parse_path(path).items():
            w[key] = value
        parts.append(w)
        if i % 100 == 0:
            print(f"{i}/{len(files)} files")

    out = pd.concat(parts, ignore_index=True)
    text_cols = ["phase", "source", "well", "instance_id"] + [f"{s}_status" for s in MODEL_SENSORS]
    out[text_cols] = out[text_cols].astype("category")   # repeated text stored compactly
    out.to_parquet(OUT, index=False)

    print("Shape:", out.shape)
    print(out.groupby("phase", observed=True).size().to_string())