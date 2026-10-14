"""Deterministic, customer-friendly tags from stable aspect ids + sentiment."""
from __future__ import annotations
from .ontology import ASPECTS, SENTIMENTS


def make_tag(aspect_id: str | None, sentiment: str) -> str | None:
    """Unknown aspect -> None (never invent a feature). Invalid input raises."""
    if sentiment not in SENTIMENTS:
        raise ValueError(f"invalid sentiment: {sentiment!r}")
    if aspect_id is None:
        return None
    if aspect_id not in ASPECTS:
        raise ValueError(f"unknown aspect id: {aspect_id!r}")
    return ASPECTS[aspect_id].tags[sentiment]


def mixed_tag(aspect_id: str) -> str:
    return f"Mixed Feedback on {ASPECTS[aspect_id].label}"
