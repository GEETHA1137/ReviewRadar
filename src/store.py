"""SQLite persistence (stdlib). Same shape as the planned PostgreSQL schema:
products, reviews (unique per product+dedup_key), analyses (unique per review+pipeline_version),
idempotency_requests. Original review text is stored byte-for-byte; offsets reference it.
"""
from __future__ import annotations
import hashlib, json, sqlite3, threading, uuid
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS products(
  product_id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, source TEXT NOT NULL,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews(
  review_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(product_id),
  source TEXT NOT NULL, external_review_id TEXT, item_title TEXT, review_text TEXT NOT NULL,
  text_sha256 TEXT NOT NULL, dedup_key TEXT NOT NULL, rating INTEGER CHECK(rating BETWEEN 1 AND 5),
  helpful_votes INTEGER NOT NULL DEFAULT 0 CHECK(helpful_votes >= 0), source_label TEXT,
  submitted_at TEXT NOT NULL, analysis_status TEXT NOT NULL DEFAULT 'pending',
  UNIQUE(product_id, dedup_key));
CREATE INDEX IF NOT EXISTS ix_reviews_product ON reviews(product_id, submitted_at, review_id);
CREATE TABLE IF NOT EXISTS analyses(
  review_id TEXT NOT NULL REFERENCES reviews(review_id), pipeline_version TEXT NOT NULL,
  original_text_sha256 TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(review_id, pipeline_version));
CREATE TABLE IF NOT EXISTS idempotency_requests(
  scope TEXT NOT NULL, key TEXT NOT NULL, request_sha256 TEXT NOT NULL, status INTEGER NOT NULL,
  response TEXT NOT NULL, PRIMARY KEY(scope, key));
"""


class IdempotencyConflict(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Store:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        self._db.executescript(SCHEMA)

    def ping(self) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1").fetchone()[0] == 1

    # ---- products --------------------------------------------------------
    def upsert_product(self, product_id, name, category, source="seed"):
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO products VALUES(?,?,?,?,?)",
                             (product_id, name, category, source, now_iso()))

    def get_product(self, product_id):
        with self._lock:
            r = self._db.execute("SELECT * FROM products WHERE product_id=?", (product_id,)).fetchone()
            return dict(r) if r else None

    def list_products(self, search, page, size):
        where, args = "", []
        if search:
            where, args = "WHERE name LIKE ? ESCAPE '\\' OR category LIKE ? ESCAPE '\\'", [f"%{_like_escape(search)}%"] * 2
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM products {where}", args).fetchone()[0]
            rows = self._db.execute(
                f"""SELECT p.*, (SELECT COUNT(*) FROM reviews r WHERE r.product_id=p.product_id) AS review_count
                    FROM products p {where} ORDER BY name, product_id LIMIT ? OFFSET ?""",
                args + [size, (page - 1) * size]).fetchall()
        return [dict(r) for r in rows], total

    # ---- idempotent ingest ----------------------------------------------
    def ingest(self, product_id, rows, analyzer, scope, key, request_sha, build_response):
        """One transaction: idempotency check -> dedup -> analyze -> save. All or nothing.
        Returns (status, response, replayed)."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                prev = self._db.execute("SELECT * FROM idempotency_requests WHERE scope=? AND key=?",
                                        (scope, key)).fetchone()
                if prev:
                    if prev["request_sha256"] != request_sha:
                        raise IdempotencyConflict()
                    self._db.execute("COMMIT")
                    return prev["status"], json.loads(prev["response"]), True
                ids, dups, key_to_id = [], 0, {}
                for row in rows:
                    text = row["review_text"]
                    dk = (f"{row['source']}:{row['external_review_id']}" if row.get("external_review_id")
                          else sha256_text(f"{text}\x00{row.get('rating')}"))
                    existing = self._db.execute("SELECT review_id FROM reviews WHERE product_id=? AND dedup_key=?",
                                                (product_id, dk)).fetchone()
                    if existing or dk in key_to_id:
                        dups += 1
                        ids.append(existing["review_id"] if existing else key_to_id[dk])
                        continue
                    rid = str(uuid.uuid4())
                    key_to_id[dk] = rid
                    self._db.execute(
                        "INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (rid, product_id, row["source"], row.get("external_review_id"), row.get("item_title"),
                         text, sha256_text(text), dk, row.get("rating"), row.get("helpful_votes", 0),
                         row.get("source_label"), now_iso(), "pending"))
                    ids.append(rid)
                    analysis = analyzer.analyze(text)  # raises -> whole request rolls back
                    self._save_analysis(rid, text, analysis)
                    self._db.execute("UPDATE reviews SET analysis_status='completed' WHERE review_id=?", (rid,))
                status, response = build_response(ids, dups)
                self._db.execute("INSERT INTO idempotency_requests VALUES(?,?,?,?,?)",
                                 (scope, key, request_sha, status, json.dumps(response)))
                self._db.execute("COMMIT")
                return status, response, False
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def _save_analysis(self, review_id, text, analysis):
        self._db.execute("INSERT OR REPLACE INTO analyses VALUES(?,?,?,?,?)",
                         (review_id, analysis["pipeline_version"], sha256_text(text), json.dumps(analysis), now_iso()))

    # ---- reads -----------------------------------------------------------
    def product_counts(self, product_id, pipeline_version):
        with self._lock:
            q = lambda sql, *a: self._db.execute(sql, a).fetchone()[0]
            total = q("SELECT COUNT(*) FROM reviews WHERE product_id=?", product_id)
            analyzed = q("""SELECT COUNT(*) FROM reviews r JOIN analyses a ON a.review_id=r.review_id
                            WHERE r.product_id=? AND a.pipeline_version=?""", product_id, pipeline_version)
            failed = q("SELECT COUNT(*) FROM reviews WHERE product_id=? AND analysis_status='failed'", product_id)
        return {"total": total, "analyzed": analyzed, "pending": max(total - analyzed - failed, 0), "failed": failed}

    def analyses_for_product(self, product_id, pipeline_version):
        """Only successful analyses from the ACTIVE pipeline version (old versions never mix in)."""
        with self._lock:
            rows = self._db.execute(
                """SELECT r.review_id, r.helpful_votes, r.review_text, a.payload, a.original_text_sha256
                   FROM reviews r JOIN analyses a ON a.review_id=r.review_id
                   WHERE r.product_id=? AND a.pipeline_version=? ORDER BY r.submitted_at DESC, r.review_id""",
                (product_id, pipeline_version)).fetchall()
        out = []
        for r in rows:
            payload = json.loads(r["payload"])
            assert sha256_text(r["review_text"]) == r["original_text_sha256"], "analysis/text mismatch"
            out.append({"review_id": r["review_id"], "helpful_votes": r["helpful_votes"],
                        "review_text": r["review_text"], "mentions": payload["mentions"], "overall": payload["overall"]})
        return out

    def get_review(self, review_id, pipeline_version):
        with self._lock:
            r = self._db.execute("SELECT * FROM reviews WHERE review_id=?", (review_id,)).fetchone()
            if not r:
                return None
            a = self._db.execute("SELECT payload FROM analyses WHERE review_id=? AND pipeline_version=?",
                                 (review_id, pipeline_version)).fetchone()
        d = dict(r)
        d["analysis"] = json.loads(a["payload"]) if a else None
        return d
