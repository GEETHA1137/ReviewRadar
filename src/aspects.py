"""Aspect extraction: longest-alias-first matching on the original text."""
from __future__ import annotations
import re
from dataclasses import dataclass

from .ontology import ALIAS_INDEX


@dataclass(frozen=True)
class RawMention:
    start: int
    end: int
    raw: str
    aspect_id: str


def _build() -> re.Pattern:
    # longest first so "battery life" wins over "battery"; whitespace in aliases is flexible
    parts = [re.escape(a).replace(r"\ ", r"\s+") for a in sorted(ALIAS_INDEX, key=lambda s: (-len(s), s))]
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.I)


_PATTERN = _build()


def extract_mentions(text: str) -> list[RawMention]:
    """Non-overlapping mentions sorted by offset; repeated occurrences are all kept."""
    out = []
    for m in _PATTERN.finditer(text):
        raw = text[m.start():m.end()]
        key = re.sub(r"\s+", " ", raw.lower())
        out.append(RawMention(m.start(), m.end(), raw, ALIAS_INDEX[key]))
    return out
