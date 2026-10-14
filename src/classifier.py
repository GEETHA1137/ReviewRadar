"""Review-level sentiment classifier (scikit-learn).

Hybrid model = TF-IDF word 1-2grams + TF-IDF char 2-5grams + lexicon features
(pos/neg clause sums, negated-hit counts) -> class-balanced logistic regression.
Loading fails closed: a missing file or hash mismatch raises ModelUnavailable.
"""
from __future__ import annotations
import hashlib, json, os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .lexicon import score_clause
from .text import split_clauses


class ModelUnavailable(RuntimeError):
    pass


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Stateless features from the clause-level lexicon engine."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            pos = neg = 0.0
            n_neg_hits = n_hits = 0
            for c in split_clauses(t):
                s, contribs = score_clause(t[c.start:c.end])
                pos += max(s, 0.0)
                neg += max(-s, 0.0)
                n_hits += len(contribs)
                n_neg_hits += sum(1 for k in contribs if k["value"] < 0)
            rows.append([pos, neg, pos - neg, np.tanh(pos - neg), n_neg_hits, n_hits])
        return np.asarray(rows, dtype=float)


def build_pipeline(kind: str = "hybrid", C: float = 1.0, seed: int = 42) -> Pipeline:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    lex = Pipeline([("f", LexiconFeatures()), ("s", StandardScaler())])
    feats = {"hybrid": [("word", word), ("char", char), ("lex", lex)],
             "tfidf": [("word", word), ("char", char)],
             "lexicon": [("lex", lex)]}[kind]
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed)
    return Pipeline([("features", FeatureUnion(feats)), ("clf", clf)])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(pipe: Pipeline, path: str, meta: dict) -> dict:
    import joblib
    joblib.dump(pipe, path)
    meta = {**meta, "sha256": sha256_file(path), "file": os.path.basename(path)}
    with open(path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return meta


class ReviewSentimentModel:
    """Loaded, hash-verified classifier. P(positive) for a review text."""

    def __init__(self, pipe: Pipeline, meta: dict):
        self.pipe, self.meta = pipe, meta

    @classmethod
    def load(cls, path: str) -> "ReviewSentimentModel":
        import joblib
        meta_path = path + ".meta.json"
        if not (os.path.isfile(path) and os.path.isfile(meta_path)):
            raise ModelUnavailable(f"model or metadata missing: {os.path.basename(path)}")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if sha256_file(path) != meta.get("sha256"):
            raise ModelUnavailable("model file hash does not match its metadata")
        try:
            return cls(joblib.load(path), meta)
        except Exception as exc:  # corrupt pickle / incompatible sklearn
            raise ModelUnavailable(f"model could not be loaded: {type(exc).__name__}") from exc

    def p_positive(self, texts: list[str]) -> list[float]:
        idx = list(self.pipe.classes_).index(1)
        return [float(p) for p in self.pipe.predict_proba(texts)[:, idx]]

    @property
    def sha256(self) -> str:
        return self.meta["sha256"]
