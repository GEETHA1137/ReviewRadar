import json, os, shutil, tempfile, threading, unittest, urllib.request, urllib.error, uuid
from tests.helpers import MODEL, RAW, ROOT
from reviewradar.app import build_service, serve
from reviewradar.classifier import ModelUnavailable, ReviewSentimentModel
from reviewradar.pipeline import ReviewAnalyzer
from reviewradar.seed import seed
from reviewradar.service import Service
from reviewradar.store import Store

HAVE_MODEL = os.path.isfile(MODEL)


def call(svc, method, path, body=None, key=None, qs=""):
    h = {"Idempotency-Key": key} if key else {}
    st, b, _ = svc.dispatch(method, path, qs, json.dumps(body).encode() if body is not None else b"", h)
    return st, b


def mk(lexicon_only=True):
    a = ReviewAnalyzer(None, True) if lexicon_only else ReviewAnalyzer.from_path(MODEL)
    s = Service(Store(":memory:"), a, os.path.join(ROOT, "models", "metrics.json"))
    s.store.upsert_product("p1", "Demo", "PC")
    s.store.upsert_product("p2", "Other", "PC")
    return s


@unittest.skipUnless(HAVE_MODEL, "trained model not present (run python -m reviewradar.train)")
class TestModel(unittest.TestCase):
    def test_loads_and_predicts(self):
        m = ReviewSentimentModel.load(MODEL)
        p = m.p_positive(["Love it, works perfectly, highly recommend", "Terrible junk, broke in a day, waste of money"])
        self.assertGreater(p[0], 0.8)
        self.assertLess(p[1], 0.2)
        self.assertEqual(m.p_positive(["Great"]), m.p_positive(["Great"]))

    def test_tampered_model_fails_closed(self):
        d = tempfile.mkdtemp()
        try:
            for f in ("review_sentiment.joblib", "review_sentiment.joblib.meta.json"):
                shutil.copy(os.path.join(os.path.dirname(MODEL), f), d)
            with open(os.path.join(d, "review_sentiment.joblib"), "ab") as fh:
                fh.write(b"x")
            with self.assertRaises(ModelUnavailable):
                ReviewSentimentModel.load(os.path.join(d, "review_sentiment.joblib"))
        finally:
            shutil.rmtree(d)

    def test_pipeline_end_to_end_requested_example(self):
        r = ReviewAnalyzer.from_path(MODEL).analyze("Amazing display but terrible battery life.")
        self.assertEqual(r["backend"], "lexicon+tfidf-lr")
        got = {(m["normalized_aspect_id"], m["sentiment"]) for m in r["mentions"]}
        self.assertEqual(got, {("display", "positive"), ("battery", "negative")})
        self.assertIn("overall", r)

    def test_reported_metrics_beat_majority_baseline(self):
        with open(os.path.join(ROOT, "models", "metrics.json")) as fh:
            rep = json.load(fh)
        A = rep["evaluation"]["protocol_A_repeated_stratified_cv"]
        self.assertGreater(A["hybrid"]["macro_f1"]["mean"], A["majority"]["macro_f1"]["mean"] + 0.3)
        self.assertEqual(rep["dataset"]["n_reviews"], 500)


class TestApi(unittest.TestCase):
    def test_health_ready_and_503_without_model(self):
        s = mk()
        self.assertEqual(call(s, "GET", "/health")[0], 200)
        self.assertEqual(call(s, "GET", "/ready")[0], 200)
        dead = Service(Store(":memory:"), None)
        dead.store.upsert_product("p1", "Demo", "PC")
        st, b = call(dead, "GET", "/ready")
        self.assertEqual((st, b["error"]["code"], b["error"]["retryable"]), (503, "MODEL_UNAVAILABLE", True))
        st, b = call(dead, "POST", "/api/reviews/analyze", {"review_text": "Good screen", "product_id": "p1"})
        self.assertEqual((st, b["error"]["code"]), (503, "MODEL_UNAVAILABLE"))  # no silent rule fallback

    def test_error_envelope_and_validation(self):
        s = mk()
        for body, field in [({"review_text": "   ", "product_id": "p1"}, "review_text"),
                            ({"review_text": "x" * 10001, "product_id": "p1"}, "review_text"),
                            ({"review_text": "ok screen", "product_id": "p1", "rating": 6}, "rating"),
                            ({"review_text": "ok screen", "product_id": "p1", "rating": True}, "rating"),
                            ({"review_text": "ok screen", "product_id": "p1", "bogus": 1}, "bogus")]:
            st, b = call(s, "POST", "/api/reviews/analyze", body)
            self.assertEqual((st, b["error"]["code"], b["error"]["details"][0]["field"]), (422, "VALIDATION_ERROR", field))
            self.assertIn("request_id", b)
        self.assertEqual(call(s, "POST", "/api/reviews/analyze", {"review_text": "ok screen", "product_id": "nope"})[0], 404)
        self.assertEqual(s.dispatch("POST", "/api/reviews/analyze", "", b"{not json", {})[0], 422)
        self.assertEqual(s.dispatch("POST", "/api/reviews/analyze", "", b"x" * 2_000_001, {})[0], 413)
        self.assertEqual(call(s, "GET", "/api/products", qs="page_size=1000")[0], 422)
        self.assertEqual(call(s, "GET", "/nope")[0], 404)

    def test_preview_is_ephemeral(self):
        s = mk()
        st, b = call(s, "POST", "/api/reviews/analyze", {"review_text": "Great screen.", "product_id": "p1"})
        self.assertEqual(st, 200)
        self.assertEqual(call(s, "GET", "/api/products/p1/insights")[1]["review_counts"]["total"], 0)

    def test_submit_idempotency_and_dedup(self):
        s = mk()
        k = str(uuid.uuid4())
        body = {"reviews": [{"review_text": "Great screen, terrible battery."}]}
        st1, r1 = call(s, "POST", "/api/products/p1/reviews", body, k)
        st2, r2 = call(s, "POST", "/api/products/p1/reviews", body, k)  # lost-response retry
        self.assertEqual((st1, st2, r1["review_ids"]), (201, 200, r2["review_ids"]))
        self.assertEqual(call(s, "GET", "/api/products/p1/insights")[1]["review_counts"]["total"], 1)
        st, b = call(s, "POST", "/api/products/p1/reviews", {"reviews": [{"review_text": "Different text"}]}, k)
        self.assertEqual((st, b["error"]["code"]), (409, "IDEMPOTENCY_CONFLICT"))
        st, r3 = call(s, "POST", "/api/products/p1/reviews", body, str(uuid.uuid4()))  # same text, new key
        self.assertEqual(r3["duplicate_count"], 1)
        self.assertEqual(r3["review_ids"], r1["review_ids"])
        call(s, "POST", "/api/products/p2/reviews", body, str(uuid.uuid4()))  # same text, other product is NOT a dup
        self.assertEqual(call(s, "GET", "/api/products/p2/insights")[1]["review_counts"]["total"], 1)
        self.assertEqual(call(s, "POST", "/api/products/p1/reviews", body, "not-a-uuid")[0], 422)

    def test_failed_batch_rolls_back_everything(self):
        s = mk()
        bad = {"reviews": [{"review_text": "Good screen"}, {"review_text": "   "}]}
        self.assertEqual(call(s, "POST", "/api/products/p1/reviews", bad, str(uuid.uuid4()))[0], 422)
        self.assertEqual(call(s, "GET", "/api/products/p1/insights")[1]["review_counts"]["total"], 0)

    def test_insights_threshold_and_review_detail(self):
        s = mk()
        texts = ["The battery is great.", "Great battery life!", "Battery is excellent."]
        for i, t in enumerate(texts):
            call(s, "POST", "/api/products/p1/reviews", {"reviews": [{"review_text": t}]}, str(uuid.uuid4()))
            ins = call(s, "GET", "/api/products/p1/insights")[1]
            self.assertEqual("battery" in ins["strengths"], i == 2)  # support 2 -> insufficient, 3 -> strength
        tags = call(s, "GET", "/api/products/p1/tags", qs="top_k=5")[1]
        self.assertEqual(tags["tags"][0]["tag"], "Great Battery")
        revs = call(s, "GET", "/api/products/p1/reviews", qs="aspect_id=battery&sentiment=positive")[1]
        self.assertEqual(revs["total"], 3)
        rid = revs["items"][0]["review_id"]
        d = call(s, "GET", f"/api/reviews/{rid}")[1]
        self.assertEqual(d["review_text"], d["analysis"]["review_text"])
        self.assertEqual(call(s, "GET", "/api/reviews/missing")[0], 404)

    def test_pipeline_version_change_does_not_mix(self):
        s = mk()
        call(s, "POST", "/api/products/p1/reviews", {"reviews": [{"review_text": "Great battery."}]}, str(uuid.uuid4()))
        s.analyzer.pipeline_version = "other-version"  # simulate a new active pipeline
        self.assertEqual(call(s, "GET", "/api/products/p1/insights")[1]["review_counts"]["analyzed"], 0)

    def test_metrics_endpoint_and_missing_evidence(self):
        if HAVE_MODEL:
            st, b = call(mk(), "GET", "/api/models/metrics")
            self.assertEqual(st, 200)
            self.assertIn("aspect extraction vs gold spans", b["not_evaluated"])
        s = Service(Store(":memory:"), None, "/nonexistent.json")
        self.assertEqual(call(s, "GET", "/api/models/metrics")[0], 503)

    def test_like_wildcards_escaped(self):
        s = mk()
        self.assertEqual(call(s, "GET", "/api/products", qs="search=%25")[1]["total"], 0)

    def test_seed_is_idempotent(self):
        s = mk()
        first = seed(s, RAW)
        self.assertEqual(sum(v["new"] for v in first.values()), 500)
        again = seed(s, RAW)
        self.assertEqual(sum(v["new"] for v in again.values()), 0)
        self.assertEqual(call(s, "GET", "/api/products/pc/insights")[1]["review_counts"]["total"], 100)


class TestHttp(unittest.TestCase):
    def test_real_http_round_trip(self):
        srv = serve(mk(), "127.0.0.1", 0)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{port}"
            page = urllib.request.urlopen(base + "/").read().decode()
            self.assertIn("<h1>ReviewRadar</h1>", page)
            self.assertNotIn("innerHTML", page)  # DOM built with textContent only
            req = urllib.request.Request(base + "/api/reviews/analyze", method="POST",
                                         data=json.dumps({"review_text": "Great screen", "product_id": "p1"}).encode(),
                                         headers={"Content-Type": "application/json"})
            self.assertEqual(json.load(urllib.request.urlopen(req))["analysis"]["mentions"][0]["tag"], "Great Display")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(base + "/api/products/zzz/insights")
            self.assertEqual(cm.exception.code, 404)
            self.assertEqual(json.load(cm.exception)["error"]["code"], "NOT_FOUND")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
