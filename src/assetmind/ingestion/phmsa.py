import re
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/phmsa/accident_hazardous_liquid_jan2010_present.txt")
OUT = Path("data/gold/phmsa_cause.parquet")
SPLITS = {"train": (2010, 2021), "val": (2022, 2023), "test": (2024, 2100)}

# Words that give a cause away, replaced by [MASK] in the shortcut test
MASK_PATTERNS = [r"corro\w*", r"excavat\w*", r"lightning", r"flood\w*", r"earthquake\w*",
                 r"freez\w*", r"hurricane\w*", r"root cause", r"operator error", r"human error",
                 r"incorrect\w*", r"procedur\w*"]
_MASK_RE = re.compile("|".join(MASK_PATTERNS), flags=re.IGNORECASE)

# Keyword rule for "investigation pending" (a baseline for the pending-detection task)
PENDING_PATTERN = (r"UNDER INVESTIGATION|INVESTIGATION IS ONGOING|INVESTIGATION IS STILL|"
                   r"SUPPLEMENTAL REPORT WILL|INVESTIGATION INTO THE CAUSE")


def load_phmsa(path: Path = RAW) -> pd.DataFrame:
    first_line = Path(path).open(encoding="latin-1").readline()
    sep = max(["\t", "|", ","], key=first_line.count)
    raw = pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)

    df = pd.DataFrame({
        "year": pd.to_numeric(raw["IYEAR"], errors="coerce"),
        "text": raw["NARRATIVE"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip(),
        "label": raw["CAUSE"],
        "detail": raw["CAUSE_DETAILS"],
    })
    df = df[(df["text"] != "") & df["label"].notna() & df["year"].notna()].reset_index(drop=True)
    df["year"] = df["year"].astype(int)

    df["split"] = None
    for name, (lo, hi) in SPLITS.items():
        df.loc[df["year"].between(lo, hi), "split"] = name

    df["pending"] = df["detail"].eq("UNKNOWN")                     # side task: cause not yet determined
    df["main_track"] = df["label"].ne("OTHER ACCIDENT CAUSE")      # main task: 7 physical causes
    return df


def mask_giveaways(text: pd.Series) -> pd.Series:
    return text.str.replace(_MASK_RE, "[MASK]", regex=True)


if __name__ == "__main__":
    df = load_phmsa()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("Saved", OUT, df.shape)
    main = df[df["main_track"]]
    print("\nMain track (7 physical causes):")
    print(pd.crosstab(main["label"], main["split"]).loc[:, ["train", "val", "test"]].to_string())
    print("\nPending (UNKNOWN) cases by split:")
    print(df.groupby("split")["pending"].sum().to_string())