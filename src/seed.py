"""Import the Amazon CSVs through the SAME service path as the API (idempotent).
One 'product' per category: almost every item in the sample has a single review, so
per-item aggregation would never reach the support threshold."""
from __future__ import annotations
import json, uuid

from .data import load_all
from .service import MAX_INLINE_REVIEWS, Service

SLUG = {"Books": "books", "eBooks": "ebooks", "Grocery": "grocery", "Jewelry": "jewelry", "PC & Electronics": "pc"}


def seed(svc: Service, data_dir: str) -> dict:
    reviews = load_all(data_dir)
    by_cat: dict[str, list] = {}
    for r in reviews:
        by_cat.setdefault(r.category, []).append(r)
    summary = {}
    for cat, rs in sorted(by_cat.items()):
        pid = SLUG.get(cat, cat.lower())
        svc.store.upsert_product(pid, f"Amazon {cat} (sample)", cat, "seed")
        added = dups = 0
        for i in range(0, len(rs), MAX_INLINE_REVIEWS):
            chunk = rs[i:i + MAX_INLINE_REVIEWS]
            body = {"reviews": [{"review_text": r.text, "source": "seed", "external_review_id": r.review_id,
                                 "helpful_votes": r.helpful_votes, "item_title": r.product_title} for r in chunk]}
            key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"reviewradar-seed/{pid}/{i}"))
            status, resp, _ = svc.dispatch("POST", f"/api/products/{pid}/reviews", "", json.dumps(body).encode(),
                                           {"Idempotency-Key": key})
            assert status in (200, 201), (status, resp)
            if status == 200:  # idempotent replay of an earlier import: nothing new was written
                dups += len(chunk)
                continue
            dups += resp["duplicate_count"]
            added += len(chunk) - resp["duplicate_count"]
        summary[pid] = {"reviews": len(rs), "new": added, "duplicates": dups}
    return summary


if __name__ == "__main__":
    from .app import ROOT, build_service
    import os
    print(seed(build_service(), os.path.join(ROOT, "data", "raw")))
