"""Deterministic aggregation of SAVED analyses (never runs inference).

Rules (v1): one vote per review per aspect; opposing positive+negative mentions in one
review -> neutral vote + contradictory count; support < MIN_SUPPORT distinct reviews ->
insufficient evidence; score = log1p(review_count) * avg_vote_confidence * |pos_ratio - neg_ratio|.
"""
from __future__ import annotations
import math
from collections import defaultdict

from .ontology import ASPECTS
from .tags import make_tag, mixed_tag

AGGREGATE_VERSION = "agg-1"
MIN_SUPPORT = 3
REPS_PER_SENTIMENT = 3
SENTS = ("positive", "neutral", "negative")


def _classify(v: dict, n: int) -> str:
    if v["positive"] + v["negative"] == 0:
        return "neutral"
    pr, nr = v["positive"] / n, v["negative"] / n
    if pr >= 0.80:
        return "excellent"
    if pr >= 0.60:
        return "good"
    if nr >= 0.70:
        return "poor"
    if nr >= 0.50:
        return "weak"
    return "mixed"


def aggregate(analyses: list[dict], top_k: int | None = None, min_support: int = MIN_SUPPORT) -> dict:
    """analyses: [{review_id, helpful_votes, mentions:[{mention_id, normalized_aspect_id, sentiment, confidence}]}]"""
    by_aspect: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    no_aspect = 0
    for a in analyses:
        known = [m for m in a["mentions"] if m.get("normalized_aspect_id")]
        if not known:
            no_aspect += 1
        for m in known:
            by_aspect[m["normalized_aspect_id"]][a["review_id"]].append(m)
    helpful = {a["review_id"]: a.get("helpful_votes", 0) for a in analyses}

    rows, dist = [], {s: 0 for s in SENTS}
    for aid, per_review in by_aspect.items():
        votes = []  # (review_id, sentiment, confidence, contradictory, mention_ids)
        for rid, ms in per_review.items():
            sents = {m["sentiment"] for m in ms}
            contradictory = {"positive", "negative"} <= sents
            vote = "neutral" if contradictory else ("positive" if "positive" in sents else
                                                     "negative" if "negative" in sents else "neutral")
            support = ms if contradictory else [m for m in ms if m["sentiment"] == vote]
            votes.append((rid, vote, min(m["confidence"] for m in ms), contradictory, [m["mention_id"] for m in support]))
        n = len(votes)
        counts = {s: sum(1 for v in votes if v[1] == s) for s in SENTS}
        ratios = {s: counts[s] / n for s in SENTS}
        avg_conf = sum(v[2] for v in votes) / n
        supported = n >= min_support
        cls = _classify(counts, n) if supported else "insufficient_evidence"
        score = math.log1p(n) * avg_conf * abs(ratios["positive"] - ratios["negative"])
        tone = {"excellent": "positive", "good": "positive", "poor": "negative", "weak": "negative"}.get(cls, "neutral")
        tag = (mixed_tag(aid) if cls == "mixed" else make_tag(aid, tone)) if supported else None
        reps = {}
        for s in SENTS:
            chosen = sorted((v for v in votes if v[1] == s), key=lambda v: (-v[2], -helpful[v[0]], v[0]))
            reps[s] = [{"review_id": v[0], "mention_ids": v[4], "confidence": v[2],
                        "mixed_within_review": v[3]} for v in chosen[:REPS_PER_SENTIMENT]]
        if supported:
            for s in SENTS:
                dist[s] += counts[s]
        rows.append({"aspect_id": aid, "label": ASPECTS[aid].label, "counts": counts, "ratios": ratios,
                     "mention_count": sum(len(ms) for ms in per_review.values()), "review_count": n,
                     "contradictory_review_count": sum(1 for v in votes if v[3]),
                     "average_confidence": avg_conf, "score": score, "aggregate_tag": tag,
                     "classification": cls, "rank": None, "representative_reviews": reps})
    sup = sorted((r for r in rows if r["classification"] != "insufficient_evidence"),
                 key=lambda r: (-r["score"], -r["review_count"], r["aspect_id"]))
    for i, r in enumerate(sup, 1):
        r["rank"] = i
    insuff = sorted((r for r in rows if r["classification"] == "insufficient_evidence"), key=lambda r: r["aspect_id"])
    shown = sup[:top_k] if top_k else sup
    ids = lambda cs: [r["aspect_id"] for r in sup if r["classification"] in cs]
    return {
        "aggregate_version": AGGREGATE_VERSION, "min_support": min_support,
        "review_counts": {"analyzed": len(analyses), "no_aspect": no_aspect},
        "sentiment_distribution": dist, "distribution_unit": "review-aspect votes (not star ratings)",
        "aspects": shown + insuff,
        "strengths": [i for i in ids({"excellent", "good"}) if not top_k or i in {r["aspect_id"] for r in shown}],
        "weaknesses": [i for i in ids({"poor", "weak"}) if not top_k or i in {r["aspect_id"] for r in shown}],
        "mixed": ids({"mixed"}), "neutral": ids({"neutral"}),
        "insufficient_evidence": [r["aspect_id"] for r in insuff],
    }
