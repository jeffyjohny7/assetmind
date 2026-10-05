from pathlib import Path
import pandas as pd
import pandera.pandas as pa

COLS = (["unit", "cycle"] + [f"op_{i}" for i in range(1, 4)]
        + [f"s_{i}" for i in range(1, 22)])

SCHEMA = pa.DataFrameSchema(
    {
        "unit": pa.Column(int, pa.Check.ge(1)),
        "cycle": pa.Column(int, pa.Check.ge(1)),
        **{c: pa.Column(float, nullable=False) for c in COLS[2:]},
        "rul": pa.Column(int, pa.Check.ge(0)),
    },
    strict=True,
)

def _read(path: Path) -> pd.DataFrame:
    # sep=r"\s+" absorbs the trailing spaces that create phantom columns
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    measurement_cols = COLS[2:]  # op settings + all 21 sensors
    return df.astype({"unit": int, "cycle": int, **{c: float for c in measurement_cols}})

def load_train(raw_dir: Path, subset: str = "FD001") -> pd.DataFrame:
    df = _read(raw_dir / f"train_{subset}.txt")
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return SCHEMA.validate(df)

def load_test(raw_dir: Path, subset: str = "FD001") -> pd.DataFrame:
    df = _read(raw_dir / f"test_{subset}.txt")
    true_rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None, names=["final_rul"])
    true_rul["unit"] = range(1, len(true_rul) + 1)
    df = df.merge(true_rul, on="unit")
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = df["final_rul"] + (last - df["cycle"])
    return SCHEMA.validate(df.drop(columns="final_rul"))

if __name__ == "__main__":
    raw, out = Path("data/raw/cmapss"), Path("data/silver")
    for name, loader in [("train", load_train), ("test", load_test)]:
        df = loader(raw)
        df.to_parquet(out / f"cmapss_{name}_FD001.parquet", index=False)
        print(f"{name}: {len(df):,} rows, {df['unit'].nunique()} engines")