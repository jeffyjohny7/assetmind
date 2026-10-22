from pathlib import Path
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import FeatureUnion, Pipeline
from assetmind.ingestion.phmsa import mask_giveaways, PENDING_PATTERN

GOLD = Path("data/gold/phmsa_cause.parquet")
REPORTS = Path("reports")
CS = [0.3, 1.0, 3.0, 10.0]


def build(C: float) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
    ])


def run_main(df: pd.DataFrame, masked: bool) -> dict:
    """Main track: 7 physical causes. C chosen on validation, test scored once."""
    if masked:
        df = df.assign(text=mask_giveaways(df["text"]))
    tr, va, te = (df[df["split"] == s] for s in ["train", "val", "test"])

    val_f1 = {C: f1_score(va["label"], build(C).fit(tr["text"], tr["label"]).predict(va["text"]), average="macro")
              for C in CS}
    best_C = max(val_f1, key=val_f1.get)
    model = build(best_C).fit(pd.concat([tr["text"], va["text"]]), pd.concat([tr["label"], va["label"]]))
    pred = model.predict(te["text"])

    tag = "masked" if masked else "original"
    pd.DataFrame(classification_report(te["label"], pred, output_dict=True, zero_division=0)).T.to_csv(
        REPORTS / f"phmsa_main_tfidf_{tag}_report.csv")
    pd.DataFrame({"label": te["label"].values, "pred": pred}, index=te.index).to_parquet(
        REPORTS / f"phmsa_main_tfidf_{tag}_predictions.parquet")

    metrics = {"test_macro_f1": f1_score(te["label"], pred, average="macro"),
               "test_accuracy": accuracy_score(te["label"], pred),
               "val_macro_f1_best": val_f1[best_C]}
    with mlflow.start_run(run_name=f"main_tfidf_{tag}"):
        mlflow.log_params({"task": "main_7_causes", "model": "tfidf_word_char_logreg", "masked": masked, "C": best_C})
        mlflow.log_metrics(metrics)
    return {"task": "main", "model": f"tfidf_{tag}", **metrics}


def pending_metrics(y, pred) -> dict:
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


def run_pending(df: pd.DataFrame) -> list:
    """Side task: is the cause still undetermined? Too few positives to tune, so C is fixed."""
    tr, te = df[df["split"] != "test"], df[df["split"] == "test"]
    results = []

    rule = te["text"].str.contains(PENDING_PATTERN, regex=True)
    results.append({"task": "pending", "model": "keyword_rule", **pending_metrics(te["pending"], rule)})

    model = build(1.0).fit(tr["text"], tr["pending"])
    pred = model.predict(te["text"])
    results.append({"task": "pending", "model": "tfidf", **pending_metrics(te["pending"], pred)})

    for r in results:
        with mlflow.start_run(run_name=f"pending_{r['model']}"):
            mlflow.log_params({"task": "pending_detection", "model": r["model"]})
            mlflow.log_metrics({k: v for k, v in r.items() if k in ("precision", "recall", "f1")})
    return results


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("phmsa-text-cause")
    df = pd.read_parquet(GOLD)
    main = df[df["main_track"]]

    tr, te = main[main["split"] != "test"], main[main["split"] == "test"]
    dummy = DummyClassifier(strategy="most_frequent").fit(tr["text"], tr["label"]).predict(te["text"])
    print(f"Main track dummy: macro-F1 {f1_score(te['label'], dummy, average='macro'):.3f}, "
          f"accuracy {accuracy_score(te['label'], dummy):.3f}")
    print("Pending positives: train+val", int(df[df['split'] != 'test']['pending'].sum()),
          "| test", int(df[df['split'] == 'test']['pending'].sum()))

    main_results = pd.DataFrame([run_main(main, masked=False), run_main(main, masked=True)])
    pending_results = pd.DataFrame(run_pending(df))
    print("\n" + main_results.round(3).to_string(index=False))
    print("\n" + pending_results.round(3).to_string(index=False))