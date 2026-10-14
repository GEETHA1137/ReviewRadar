"""Clause segmentation on the ORIGINAL text. All offsets are Python str indices,
i.e. Unicode code points, and `text[start:end]` always reproduces the clause."""
from __future__ import annotations
import re
from dataclasses import dataclass

MAX_REVIEW_CHARS = 10_000

# kind of boundary: sentence end / comma-like (list continuation allowed) / contrast word
_DELIM = re.compile(
    r"(?P<sent>[.;!?]+(?=\s|$))"
    r"|(?P<comma>,(?=\s)|\s[-\u2013\u2014]+\s)"
    r"|(?P<contrast>\b(?:but|however|although|though|while|except|yet)\b)",
    re.I,
)


@dataclass(frozen=True)
class Clause:
    start: int
    end: int
    boundary: str  # how this clause was introduced: start | sent | comma | contrast | and


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def split_clauses(text: str) -> list[Clause]:
    out: list[Clause] = []
    pos, kind = 0, "start"
    for m in _DELIM.finditer(text):
        a, b = _trim(text, pos, m.start())
        if b > a:
            out.append(Clause(a, b, kind))
        pos, kind = m.end(), m.lastgroup
        if kind == "contrast":
            pos = m.end()
    a, b = _trim(text, pos, len(text))
    if b > a:
        out.append(Clause(a, b, kind))
    return out


_AND = re.compile(r"\s(?:and|&)\s", re.I)


def split_on_and(text: str, clause: Clause, left_ok, right_ok) -> list[Clause]:
    """Split a clause at 'and' only if BOTH sides satisfy the given predicates
    (e.g. each has an aspect mention and polarity evidence). Never a global split."""
    seg = text[clause.start:clause.end]
    for m in _AND.finditer(seg):
        ls, le = _trim(text, clause.start, clause.start + m.start())
        rs, re_ = _trim(text, clause.start + m.end(), clause.end)
        if le > ls and re_ > rs and left_ok(ls, le) and right_ok(rs, re_):
            return [Clause(ls, le, clause.boundary), Clause(rs, re_, "and")]
    return [clause]
