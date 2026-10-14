"""Compose extraction -> clause context -> aspect sentiment -> normalization -> tags.

Aspect sentiment is rule-based (lexicon + negation + clause context); the trained
classifier supplies the review-level `overall` sentiment. Aspect probabilities are a
fixed softmax of the clause score: they sum to 1 but are NOT calibrated.
"""
from __future__ import annotations
import hashlib, json, math, time

from . import __version__
from .aspects import RawMention, extract_mentions
from .classifier import ModelUnavailable, ReviewSentimentModel
from .lexicon import DEFAULTS, lexicon_fingerprint, score_clause
from .ontology import ASPECTS, ONTOLOGY_VERSION
from .tags import make_tag
from .text import MAX_REVIEW_CHARS, Clause, split_clauses, split_on_and

CONTEXT_POLICY = "clause_with_comma_borrow"
SOFTMAX_K = 2.5
MAX_BORROW_HOPS = 2


class InvalidReview(ValueError):
    pass


def _probs(score: float, band: float) -> dict[str, float]:
    z = {"positive": SOFTMAX_K * (score - band), "negative": SOFTMAX_K * (-score - band), "neutral": 0.0}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    t = sum(e.values())
    return {k: e[k] / t for k in ("negative", "neutral", "positive")}


def validate_text(text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise InvalidReview("review_text must be a non-blank string")
    if len(text) > MAX_REVIEW_CHARS:
        raise InvalidReview(f"review_text exceeds {MAX_REVIEW_CHARS} characters")
    return text


class ReviewAnalyzer:
    def __init__(self, model: ReviewSentimentModel | None, allow_lexicon_only: bool = False):
        if model is None and not allow_lexicon_only:
            raise ModelUnavailable("no trained model supplied; pass allow_lexicon_only=True to opt in explicitly")
        if model is not None and model.meta.get("lexicon_sha256") != lexicon_fingerprint():
            raise ModelUnavailable("lexicon changed since the model was trained; retrain (python -m reviewradar.train)")
        self.model = model
        self.params = dict(DEFAULTS)
        self.backend = "lexicon+tfidf-lr" if model else "lexicon-only"
        cfg = {"model": model.sha256 if model else None, "ontology": ONTOLOGY_VERSION,
               "lexicon": lexicon_fingerprint(), "policy": CONTEXT_POLICY, "params": self.params,
               "softmax_k": SOFTMAX_K, "code": __version__}
        self.pipeline_version = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_path(cls, model_path: str, allow_lexicon_only: bool = False) -> "ReviewAnalyzer":
        try:
            return cls(ReviewSentimentModel.load(model_path), allow_lexicon_only)
        except ModelUnavailable:
            if allow_lexicon_only:  # explicit opt-in only
                return cls(None, True)
            raise

    # ---- internals -------------------------------------------------------
    def _clauses(self, text: str, mentions: list[RawMention]) -> list[Clause]:
        def has_m(a, b):
            return any(a <= m.start and m.end <= b for m in mentions)

        def ok(a, b):
            return has_m(a, b) and bool(score_clause(text[a:b], self.params)[1])

        out: list[Clause] = []
        for c in split_clauses(text):
            n = sum(1 for m in mentions if c.start <= m.start < c.end)
            out.extend(split_on_and(text, c, ok, ok) if n >= 2 else [c])
        return out

    def analyze(self, text: str) -> dict:
        t0 = time.perf_counter()
        validate_text(text)
        mentions = extract_mentions(text)
        clauses = self._clauses(text, mentions)
        scored = [score_clause(text[c.start:c.end], self.params) for c in clauses]
        band = self.params["band"]
        out_m = []
        for m in mentions:
            i = next((k for k, c in enumerate(clauses) if c.start <= m.start < c.end), None)
            if i is None:
                continue
            score, contribs = scored[i]
            cs, ce = clauses[i].start, clauses[i].end
            j, hops = i, 0
            while not contribs and j + 1 < len(clauses) and clauses[j + 1].boundary == "comma" and hops < MAX_BORROW_HOPS:
                j += 1
                hops += 1
                score, contribs = scored[j]
                ce = clauses[j].end
            sent = "positive" if score >= band else "negative" if score <= -band else "neutral"
            pr = _probs(score, band)
            conf = pr[sent] if sent in pr else max(pr.values())
            asp = ASPECTS[m.aspect_id]
            out_m.append({
                "mention_id": f"m{m.start}-{m.end}", "raw_aspect": m.raw,
                "start_char": m.start, "end_char": m.end, "aspect_confidence": 1.0,
                "normalized_aspect_id": asp.id, "normalized_aspect": asp.label,
                "normalization": {"method": "alias", "similarity": None, "is_unknown": False},
                "sentiment": sent, "sentiment_confidence": conf, "probabilities": pr,
                "tag": make_tag(asp.id, sent), "confidence": min(1.0, conf),
                "context_start_char": cs, "context_end_char": ce,
                "evidence": [c["term"] for c in contribs], "clause_score": score,
            })
        overall = None
        if self.model:
            p = self.model.p_positive([text])[0]
            overall = {"label": "positive" if p >= 0.5 else "negative", "p_positive": p,
                       "model_sha256": self.model.sha256}
        warnings = [] if self.model else ["lexicon-only mode: no trained review-level model is loaded"]
        return {
            "review_text": text, "mentions": out_m, "overall": overall, "backend": self.backend,
            "pipeline_version": self.pipeline_version, "ontology_version": ONTOLOGY_VERSION,
            "context_policy": CONTEXT_POLICY, "probabilities_calibrated": False,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3), "warnings": warnings,
            "offset_unit": "unicode_code_points",
        }

    def analyze_batch(self, texts: list[str], max_batch: int = 100) -> list[dict]:
        if len(texts) > max_batch:
            raise InvalidReview(f"batch exceeds {max_batch} reviews")
        return [self.analyze(t) for t in texts]
