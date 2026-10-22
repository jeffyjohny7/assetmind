import pandas as pd
from assetmind.ingestion.phmsa import mask_giveaways, SPLITS



def test_mask_replaces_giveaways_only():
    out = mask_giveaways(pd.Series(["EXTERNAL CORROSION FOUND", "Pump seal failed", "Excavator struck line"]))
    assert out.tolist() == ["EXTERNAL [MASK] FOUND", "Pump seal failed", "[MASK] struck line"]

def test_time_splits_do_not_overlap():
    ranges = sorted(SPLITS.values())
    assert all(a_hi < b_lo for (_, a_hi), (b_lo, _) in zip(ranges, ranges[1:]))