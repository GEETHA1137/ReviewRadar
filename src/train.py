"""Train + evaluate the review sentiment model.

Protocols (all out-of-fold; the final model is fit on everything afterwards):
  A. repeated stratified 5-fold CV (3 repeats), C chosen by inner 3-fold grid search
  B. leave-one-category-out (does it generalise to an unseen product domain?)
Baselines: majority class, lexicon rule (total > 0), lexicon-LR, TF-IDF-LR.
"""
from __future__ import annotations
import argparse, json, os, platform, time
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from .classifier import LexiconFeatures, build_pipeline, save_model
from .data import load_all
from .lexicon import lexicon_fingerprint
from .ontology import ONTOLOGY_VERSION

GRID = [0.3, 1.0, 3.0, 10.0]
SEED = 42


def _fit_predict(kind, Xtr, ytr, Xte):
    if kind == "majority":
        p = float(np.mean(ytr))
        return np.full(len(Xte), p)
    if kind == "lexicon_rule":
        tot = LexiconFeatures().transform(Xte)[:, 2]
        return 1 / (1 + np.exp(-2.0 * tot))  # monotone map; decision at total == 0
    gs = GridSearchCV(build_pipeline(kind, seed=SEED), {"clf__C": GRID}, scoring="f1_macro",
                      cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1)
    gs.fit(Xtr, ytr)
    return gs.predict_proba(Xte)[:, list(gs.classes_).index(1)]


def metrics(y, p, thr=0.5):
    yh = (np.asarray(p) >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yh, labels=[0, 1], zero_division=0)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = None
    return {"accuracy": float(accuracy_score(y, yh)), "macro_f1": float(f1_score(y, yh, average="macro", zero_division=0)),
            "negative": {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f1[0])},
            "positive": {"precision": float(pr[1]), "recall": float(rc[1]), "f1": float(f1[1])},
            "roc_auc": auc, "confusion_matrix_rows_true_cols_pred_[neg,pos]": confusion_matrix(y, yh, labels=[0, 1]).tolist()}


def _mean_std(rows, keys):
    return {k: {"mean": float(np.mean([r[k] for r in rows if r[k] is not None])),
                "std": float(np.std([r[k] for r in rows if r[k] is not None]))} for k in keys}


def evaluate(texts, y, cats, kinds, repeats=3, folds=5):
    y, cats = np.asarray(y), np.asarray(cats)
    out = {"protocol_A_repeated_stratified_cv": {}, "protocol_B_leave_one_category_out": {}}
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    splits = list(rskf.split(texts, y))
    for kind in kinds:
        per_rep = []
        for r in range(repeats):
            p = np.zeros(len(y))
            for tr, te in splits[r * folds:(r + 1) * folds]:
                p[te] = _fit_predict(kind, [texts[i] for i in tr], y[tr], [texts[i] for i in te])
            per_rep.append(metrics(y, p))
        out["protocol_A_repeated_stratified_cv"][kind] = {
            "repeats": repeats, "folds": folds,
            **_mean_std(per_rep, ["accuracy", "macro_f1", "roc_auc"]),
            "negative_recall": _mean_std([r["negative"] for r in per_rep], ["recall"])["recall"],
            "negative_f1": _mean_std([r["negative"] for r in per_rep], ["f1"])["f1"],
            "last_repeat_detail": per_rep[-1]}
        folds_b = {}
        for c in sorted(set(cats)):
            te = np.where(cats == c)[0]
            tr = np.where(cats != c)[0]
            p = _fit_predict(kind, [texts[i] for i in tr], y[tr], [texts[i] for i in te])
            m = metrics(y[te], p)
            folds_b[str(c)] = {"n": int(len(te)), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                               "negative_recall": m["negative"]["recall"]}
        out["protocol_B_leave_one_category_out"][kind] = {
            "per_category": folds_b,
            "mean_macro_f1": float(np.mean([v["macro_f1"] for v in folds_b.values()])),
            "mean_accuracy": float(np.mean([v["accuracy"] for v in folds_b.values()]))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="models")
    ap.add_argument("--quick", action="store_true", help="1 CV repeat (faster)")
    a = ap.parse_args(argv)
    t0 = time.time()
    reviews = load_all(a.data)
    texts = [r.text for r in reviews]
    y = np.array([1 if r.label == "positive" else 0 for r in reviews])
    cats = [r.category for r in reviews]
    kinds = ["majority", "lexicon_rule", "lexicon", "tfidf", "hybrid"]
    ev = evaluate(texts, y, cats, kinds, repeats=1 if a.quick else 3)

    gs = GridSearchCV(build_pipeline("hybrid", seed=SEED), {"clf__C": GRID}, scoring="f1_macro",
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1).fit(texts, y)
    final = build_pipeline("hybrid", C=gs.best_params_["clf__C"], seed=SEED).fit(texts, y)
    os.makedirs(a.out, exist_ok=True)
    meta = save_model(final, os.path.join(a.out, "review_sentiment.joblib"), {
        "kind": "hybrid tfidf+lexicon logistic regression", "C": gs.best_params_["clf__C"],
        "n_train": len(y), "label_map": {"negative": 0, "positive": 1}, "seed": SEED,
        "ontology_version": ONTOLOGY_VERSION, "lexicon_sha256": lexicon_fingerprint(),
        "sklearn": sklearn.__version__, "python": platform.python_version()})
    report = {
        "dataset": {"n_reviews": len(y), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
                    "per_category": {c: cats.count(c) for c in sorted(set(cats))},
                    "labels": "binary (positive/negative) from the Sentiment_* columns; star_rating == label so it is NOT a feature"},
        "evaluation": ev, "final_model": meta,
        "limitations": [
            "500 reviews, 5 categories, binary labels only: no neutral class and no aspect-level gold labels.",
            "These numbers measure review-level sentiment only. Aspect extraction/aspect sentiment are rule-based and "
            "are checked only against hand-written fixtures (tests/), not independent gold annotations.",
            "Final model is trained on all 500 reviews; reported scores are out-of-fold estimates from separate fits.",
            'Lexicon weights were ported from the ReviewRadar.html explorer, which was probably tuned while looking at these same 500 reviews. The lexicon is not fit inside CV, so lexicon_rule / lexicon / hybrid scores are likely optimistic versus truly unseen reviews; only protocol B (unseen category) partly mitigates this. Treat them as an upper-ish estimate, not a guarantee.',
            "C is chosen by inner CV inside each outer fold (nested), so the CV scores are not tuned on their own test folds."],
        "elapsed_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: {m: round(v["macro_f1"]["mean"], 3) for m, v in ev["protocol_A_repeated_stratified_cv"].items()}
                      for k in ["macro_f1_cv"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
