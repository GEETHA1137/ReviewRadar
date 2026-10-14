"""HTTP-agnostic service layer. `Service.dispatch` is the single router used by both the
FastAPI adapter and the stdlib server, so what is tested is what is served.
Every non-2xx response uses the envelope:
  {"error": {"code", "message", "details", "retryable"}, "request_id"}
"""
from __future__ import annotations
import json, os, re, uuid
from urllib.parse import parse_qs

from .aggregate import AGGREGATE_VERSION, MIN_SUPPORT, aggregate
from .classifier import ModelUnavailable
from .ontology import ASPECTS, SENTIMENTS
from .pipeline import InvalidReview, ReviewAnalyzer, validate_text
from .store import IdempotencyConflict, Store, sha256_text

MAX_BODY_BYTES = 2_000_000
MAX_INLINE_REVIEWS = 100
PAGE_SIZE, MAX_PAGE_SIZE = 20, 100
DEFAULT_TOP_K, MAX_TOP_K = 10, 50
SOURCES = {"manual", "import", "seed"}


class ApiError(Exception):
    def __init__(self, status, code, message, details=None, retryable=False, headers=None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details, self.retryable, self.headers = details or [], retryable, headers or {}


def _bad(field, msg):
    return ApiError(422, "VALIDATION_ERROR", "Request validation failed.", [{"field": field, "message": msg}])


def _int(q, name, default, lo, hi):
    raw = q.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise _bad(name, "must be an integer")
    if not lo <= v <= hi:
        raise _bad(name, f"must be between {lo} and {hi}")
    return v


def _rating(v):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 5:
        raise _bad("rating", "must be an integer from 1 to 5")
    return v


def _review_text(v, field="review_text"):
    try:
        return validate_text(v)
    except InvalidReview as e:
        raise _bad(field, str(e))


class Service:
    def __init__(self, store: Store, analyzer: ReviewAnalyzer | None, metrics_path: str | None = None):
        self.store, self.analyzer, self.metrics_path = store, analyzer, metrics_path
        self._routes = [
            ("GET", r"^/health$", self.health), ("GET", r"^/ready$", self.ready),
            ("POST", r"^/api/reviews/analyze$", self.preview),
            ("GET", r"^/api/products$", self.products),
            ("POST", r"^/api/products/(?P<pid>[^/]+)/reviews$", self.submit),
            ("GET", r"^/api/products/(?P<pid>[^/]+)/reviews$", self.product_reviews),
            ("GET", r"^/api/products/(?P<pid>[^/]+)/insights$", self.insights),
            ("GET", r"^/api/products/(?P<pid>[^/]+)/tags$", self.tags),
            ("GET", r"^/api/reviews/(?P<rid>[^/]+)$", self.review),
            ("GET", r"^/api/models/metrics$", self.metrics),
        ]

    # ---- router ----------------------------------------------------------
    def dispatch(self, method: str, path: str, query_string: str = "", body: bytes = b"", headers: dict | None = None):
        """Returns (status, json_body, extra_headers)."""
        rid = str(uuid.uuid4())
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        try:
            if len(body) > MAX_BODY_BYTES:
                raise ApiError(413, "PAYLOAD_TOO_LARGE", "Request body is too large.")
            q = {k: v[0] for k, v in parse_qs(query_string, keep_blank_values=True).items()}
            matched_path = False
            for m, rx, fn in self._routes:
                mt = re.match(rx, path)
                if not mt:
                    continue
                matched_path = True
                if m != method:
                    continue
                status, payload = fn(q=q, body=body, headers=headers, **mt.groupdict())
                return status, payload, {"X-Request-ID": rid}
            if matched_path:
                raise ApiError(405, "VALIDATION_ERROR", "Method not allowed.")
            raise ApiError(404, "NOT_FOUND", "Route not found.")
        except ApiError as e:
            return e.status, self._envelope(e, rid), {"X-Request-ID": rid, **e.headers}
        except ModelUnavailable:
            e = ApiError(503, "MODEL_UNAVAILABLE", "Analysis is temporarily unavailable.", retryable=True)
            return 503, self._envelope(e, rid), {"X-Request-ID": rid}
        except Exception:  # never leak tracebacks
            e = ApiError(500, "INTERNAL_ERROR", "Internal error.", retryable=True)
            return 500, self._envelope(e, rid), {"X-Request-ID": rid}

    @staticmethod
    def _envelope(e: ApiError, rid: str) -> dict:
        return {"error": {"code": e.code, "message": e.message, "details": e.details, "retryable": e.retryable},
                "request_id": rid}

    # ---- helpers ---------------------------------------------------------
    def _json(self, body: bytes) -> dict:
        try:
            data = json.loads(body.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(422, "VALIDATION_ERROR", "Malformed JSON body.", [{"field": "body", "message": "invalid JSON"}])
        if not isinstance(data, dict):
            raise _bad("body", "must be a JSON object")
        return data

    def _need_analyzer(self) -> ReviewAnalyzer:
        if self.analyzer is None:
            raise ApiError(503, "MODEL_UNAVAILABLE", "Analysis is temporarily unavailable.", retryable=True)
        return self.analyzer

    def _product(self, pid):
        p = self.store.get_product(pid)
        if not p:
            raise ApiError(404, "NOT_FOUND", "Product not found.")
        return p

    # ---- handlers --------------------------------------------------------
    def health(self, **_):
        return 200, {"status": "ok"}

    def ready(self, **_):
        try:
            db = self.store.ping()
        except Exception:
            db = False
        if not db:
            raise ApiError(503, "DATABASE_UNAVAILABLE", "Database unavailable.", retryable=True)
        a = self._need_analyzer()
        return 200, {"status": "ready", "backend": a.backend, "pipeline_version": a.pipeline_version}

    def preview(self, body, **_):
        data = self._json(body)
        extra = set(data) - {"review_text", "product_id", "rating"}
        if extra:
            raise _bad(sorted(extra)[0], "unknown field")
        text = _review_text(data.get("review_text"))
        _rating(data.get("rating"))
        pid = data.get("product_id")
        if not isinstance(pid, str) or not pid:
            raise _bad("product_id", "required")
        self._product(pid)
        analysis = self._need_analyzer().analyze(text)  # ephemeral: nothing is stored
        return 200, {"preview_id": str(uuid.uuid4()), "status": "completed", "analysis": analysis}

    def products(self, q, **_):
        page = _int(q, "page", 1, 1, 10**6)
        size = _int(q, "page_size", PAGE_SIZE, 1, MAX_PAGE_SIZE)
        items, total = self.store.list_products(q.get("search") or None, page, size)
        return 200, {"items": items, "total": total, "page": page, "page_size": size}

    def submit(self, pid, body, headers, **_):
        self._product(pid)
        key = headers.get("idempotency-key", "")
        try:
            uuid.UUID(key)
            ok = len(key) <= 64
        except ValueError:
            ok = False
        if not ok:
            raise _bad("Idempotency-Key", "header must be a UUID")
        data = self._json(body)
        if set(data) - {"reviews"} or not isinstance(data.get("reviews"), list):
            raise _bad("reviews", "required array")
        if not 1 <= len(data["reviews"]) <= MAX_INLINE_REVIEWS:
            raise _bad("reviews", f"must contain 1..{MAX_INLINE_REVIEWS} reviews")
        rows = []
        for i, r in enumerate(data["reviews"]):
            f = f"reviews[{i}]"
            if not isinstance(r, dict):
                raise _bad(f, "must be an object")
            extra = set(r) - {"review_text", "rating", "source", "external_review_id", "helpful_votes", "item_title"}
            if extra:
                raise _bad(f"{f}.{sorted(extra)[0]}", "unknown field")
            src = r.get("source", "manual")
            hv = r.get("helpful_votes", 0)
            if src not in SOURCES:
                raise _bad(f"{f}.source", f"must be one of {sorted(SOURCES)}")
            if isinstance(hv, bool) or not isinstance(hv, int) or hv < 0:
                raise _bad(f"{f}.helpful_votes", "must be a non-negative integer")
            ext = r.get("external_review_id")
            if ext is not None and (not isinstance(ext, str) or not ext or len(ext) > 128):
                raise _bad(f"{f}.external_review_id", "must be a non-empty string up to 128 chars")
            rows.append({"review_text": _review_text(r.get("review_text"), f"{f}.review_text"),
                         "rating": _rating(r.get("rating")), "source": src, "external_review_id": ext,
                         "helpful_votes": hv, "item_title": r.get("item_title") if isinstance(r.get("item_title"), str) else None,
                         "source_label": None})
        analyzer = self._need_analyzer()
        req_sha = sha256_text(json.dumps(data, sort_keys=True, ensure_ascii=False))
        try:
            status, resp, replayed = self.store.ingest(
                pid, rows, analyzer, f"POST /api/products/{pid}/reviews", key, req_sha,
                lambda ids, dups: (201, {"review_ids": ids, "duplicate_count": dups, "status": "completed",
                                         "pipeline_version": analyzer.pipeline_version}))
        except IdempotencyConflict:
            raise ApiError(409, "IDEMPOTENCY_CONFLICT", "Idempotency-Key was already used with a different request.")
        return (200 if replayed else status), resp

    def _snapshot(self, pid, top_k=None):
        p = self._product(pid)
        a = self._need_analyzer()
        analyses = self.store.analyses_for_product(pid, a.pipeline_version)
        agg = aggregate(analyses, top_k=top_k)
        counts = self.store.product_counts(pid, a.pipeline_version)
        counts["no_aspect"] = agg["review_counts"]["no_aspect"]
        return p, a, analyses, agg, counts

    def insights(self, pid, **_):
        p, a, analyses, agg, counts = self._snapshot(pid)
        return 200, {"product": p, "pipeline_version": a.pipeline_version, "aggregate_version": AGGREGATE_VERSION,
                     "data_revision": counts["analyzed"], "stale": False, "review_counts": counts,
                     **{k: v for k, v in agg.items() if k not in ("review_counts", "aggregate_version")}}

    def tags(self, pid, q, **_):
        k = _int(q, "top_k", DEFAULT_TOP_K, 1, MAX_TOP_K)
        p, a, analyses, agg, counts = self._snapshot(pid, top_k=k)
        ranked = [{"rank": r["rank"], "aspect_id": r["aspect_id"], "label": r["label"], "tag": r["aggregate_tag"],
                   "classification": r["classification"], "review_count": r["review_count"], "score": r["score"]}
                  for r in agg["aspects"] if r["rank"] is not None]
        return 200, {"product_id": pid, "pipeline_version": a.pipeline_version, "aggregate_version": AGGREGATE_VERSION,
                     "data_revision": counts["analyzed"], "min_support": MIN_SUPPORT, "tags": ranked}

    def product_reviews(self, pid, q, **_):
        page = _int(q, "page", 1, 1, 10**6)
        size = _int(q, "page_size", PAGE_SIZE, 1, MAX_PAGE_SIZE)
        aid, sent = q.get("aspect_id"), q.get("sentiment")
        if aid and aid not in ASPECTS:
            raise _bad("aspect_id", "unknown aspect id")
        if sent and sent not in SENTIMENTS:
            raise _bad("sentiment", f"must be one of {list(SENTIMENTS)}")
        p, a, analyses, _agg, _c = self._snapshot(pid)
        items = []
        for an in analyses:
            ms = [m for m in an["mentions"] if (not aid or m["normalized_aspect_id"] == aid) and (not sent or m["sentiment"] == sent)]
            if (aid or sent) and not ms:
                continue
            items.append({"review_id": an["review_id"], "review_text": an["review_text"], "overall": an["overall"],
                          "mentions": ms, "helpful_votes": an["helpful_votes"]})
        start = (page - 1) * size
        return 200, {"items": items[start:start + size], "total": len(items), "page": page, "page_size": size}

    def review(self, rid, **_):
        a = self._need_analyzer()
        r = self.store.get_review(rid, a.pipeline_version)
        if not r:
            raise ApiError(404, "NOT_FOUND", "Review not found.")
        return 200, r

    def metrics(self, **_):
        path = self.metrics_path
        if not path or not os.path.isfile(path):
            raise ApiError(503, "MODEL_UNAVAILABLE", "Evaluation evidence is unavailable.", retryable=False)
        try:
            with open(path, encoding="utf-8") as fh:
                rep = json.load(fh)
        except (OSError, json.JSONDecodeError):
            raise ApiError(503, "MODEL_UNAVAILABLE", "Evaluation evidence is unreadable.")
        return 200, {"evidence_source": os.path.basename(path), "dataset": rep["dataset"],
                     "evaluation": rep["evaluation"], "limitations": rep["limitations"],
                     "not_evaluated": ["aspect extraction vs gold spans", "aspect-level sentiment vs gold labels",
                                       "end-to-end tag F1", "latency benchmark"]}
