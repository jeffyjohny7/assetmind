from pathlib import Path
import pandas as pd
from assetmind.features.cmapss import CmapssFeatureBuilder

SILVER, GOLD = Path("data/silver"), Path("data/gold")

if __name__ == "__main__":
    train = pd.read_parquet(SILVER / "cmapss_train_FD001.parquet")
    test = pd.read_parquet(SILVER / "cmapss_test_FD001.parquet")

    fb = CmapssFeatureBuilder().fit(train)   # rules learned from train only
    for name, df in [("train", train), ("test", test)]:
        out = fb.transform(df)
        out.to_parquet(GOLD / f"cmapss_{name}_FD001.parquet", index=False)
        print(f"{name}: {out.shape[0]:,} rows, {out.shape[1]} columns")
    print("Dropped sensors:", fb.dropped_)