from pathlib import Path
import pandas as pd

WIN = Path("data/gold/3w_windows_real_ext_h.parquet")
FM = Path("data/gold/3w_fm_scores.parquet")
OUT = Path("data/gold/3w_windows_real_ext_h_fm.parquet")

if __name__ == "__main__":
    w = pd.read_parquet(WIN)
    w["instance_id"] = w["instance_id"].astype(str)
    w["window_end"] = w["window_end"].astype("datetime64[ns]")

    fm = pd.read_parquet(FM)
    fm["window_end"] = fm["window_end"].astype("datetime64[ns]")
    wide = fm.pivot_table(index=["instance_id", "window_end"], columns="sensor", values=["fmerr", "fmout"])
    wide.columns = [f"{sensor}_{score}" for score, sensor in wide.columns]
    wide = wide.reset_index()

    out = w.merge(wide, on=["instance_id", "window_end"], how="left")
    assert len(out) == len(w), "merge changed the number of windows"
    out.to_parquet(OUT, index=False)

    new = [c for c in out.columns if c.endswith("_fmerr")]
    print("Shape:", out.shape)
    print("Share of windows with a forecast score:\n", out[new].notna().mean().round(2).to_string())