from pathlib import Path
import pandas as pd
from assetmind.ingestion.threew import MODEL_SENSORS_EXT, parse_path, read_instance, make_windows

W3 = Path("../3W/dataset")


def build(sensors, out: Path):
    files = sorted(W3.glob("[0-9]/WELL-*.parquet"))
    parts = []
    for i, path in enumerate(files, 1):
        w = make_windows(read_instance(path, sensors), sensors)
        for key, value in parse_path(path).items():
            w[key] = value
        parts.append(w)
        if i % 100 == 0:
            print(f"{i}/{len(files)} files")
    out_df = pd.concat(parts, ignore_index=True)
    text_cols = ["phase", "source", "well", "instance_id"] + [f"{s}_status" for s in sensors]
    out_df[text_cols] = out_df[text_cols].astype("category")
    out_df.to_parquet(out, index=False)
    print("Shape:", out_df.shape)


if __name__ == "__main__":
    build(MODEL_SENSORS_EXT, Path("data/gold/3w_windows_real_ext.parquet"))