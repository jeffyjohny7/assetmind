from pathlib import Path
from assetmind.ingestion.cmapss import load_train

RAW = Path("data/raw/cmapss")

def test_rul_is_zero_at_failure():
    df = load_train(RAW)
    last_rows = df.loc[df.groupby("unit")["cycle"].idxmax()]
    assert (last_rows["rul"] == 0).all()

def test_rul_decreases_by_one_each_cycle():
    df = load_train(RAW).sort_values(["unit", "cycle"])
    diffs = df.groupby("unit")["rul"].diff().dropna()
    assert (diffs == -1).all()