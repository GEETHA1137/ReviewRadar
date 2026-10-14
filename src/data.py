"""Load and clean the Amazon review CSVs (one file per category)."""
from __future__ import annotations
import csv, glob, html, os, re
from dataclasses import dataclass

CATEGORY_BY_FILE = {"books": "Books", "ebook": "eBooks", "grocery": "Grocery",
                    "jwellery": "Jewelry", "pc": "PC & Electronics"}
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Strip wrapping quotes, HTML tags/entities and collapse whitespace."""
    s = (s or "").strip().strip('"')
    s = _TAG.sub(" ", html.unescape(s))
    return _WS.sub(" ", s).strip()


def parse_label(raw: str) -> str:
    raw = (raw or "").strip().lower()
    if raw.startswith("pos"):
        return "positive"
    if raw.startswith("neg"):  # source files spell it 'negaitve'
        return "negative"
    raise ValueError(f"unknown sentiment label: {raw!r}")


@dataclass(frozen=True)
class Review:
    review_id: str
    product_id: str
    product_title: str
    category: str
    headline: str
    body: str
    helpful_votes: int
    label: str  # positive | negative

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.body}" if self.headline else self.body


def _category(path: str) -> str:
    base = os.path.basename(path).lower()
    for key, name in CATEGORY_BY_FILE.items():
        if key in base:
            return name
    return "Other"


def load_csv(path: str) -> list[Review]:
    """Parse one CSV. star_rating is deliberately ignored: it equals the label (leakage)."""
    out, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            lab_col = next(c for c in row if c.lower().startswith("sentiment"))
            body = clean_text(row["review_body"])
            rid = clean_text(row["review_id"])
            if not body or rid in seen:
                continue
            seen.add(rid)
            out.append(Review(rid, clean_text(row["product_id"]), clean_text(row["product_title"]),
                              _category(path), clean_text(row["review_headline"]), body,
                              int(float(row.get("helpful_votes") or 0)), parse_label(row[lab_col])))
    return out


def load_all(folder: str) -> list[Review]:
    reviews: list[Review] = []
    for p in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        reviews.extend(load_csv(p))
    return reviews
