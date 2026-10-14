import unittest
from tests.helpers import RAW
from reviewradar.data import clean_text, parse_label, load_all
from reviewradar.ontology import ASPECTS, ALIAS_INDEX
from reviewradar.text import split_clauses
from reviewradar.aspects import extract_mentions
from reviewradar.lexicon import score_clause
from reviewradar.pipeline import ReviewAnalyzer, InvalidReview
from reviewradar.classifier import ModelUnavailable
from reviewradar.tags import make_tag


class TestData(unittest.TestCase):
    def test_clean_and_label(self):
        self.assertEqual(clean_text('"Great<br />book &amp; fun"'), "Great book & fun")
        self.assertEqual(parse_label("negaitve"), "negative")  # typo in source files
        with self.assertRaises(ValueError):
            parse_label("meh")

    def test_load_all(self):
        rs = load_all(RAW)
        self.assertEqual(len(rs), 500)
        self.assertEqual(sum(r.label == "negative" for r in rs), 102)
        import re
        self.assertFalse([r for r in rs if re.search(r"<[^>]+>|&amp;|&#", r.body)])  # no markup/entities left


class TestExtraction(unittest.TestCase):
    def test_offsets_exact_with_emoji_and_repeats(self):
        t = "😀 Battery life is great, the battery lasts. Screen!"
        ms = extract_mentions(t)
        self.assertEqual([m.raw for m in ms][:2], ["Battery life", "battery"])  # longest alias wins
        for m in ms:
            self.assertEqual(t[m.start:m.end], m.raw)
        self.assertEqual(sum(m.aspect_id == "battery" for m in ms), 2)  # repeats retained

    def test_adjectives_alone_do_not_create_aspects(self):
        self.assertEqual(extract_mentions("It is fast and good."), [])

    def test_ambiguous_charge_not_mapped(self):
        self.assertEqual(extract_mentions("I had to charge it."), [])

    def test_alias_ownership_unique_and_templates_complete(self):
        self.assertGreater(len(ALIAS_INDEX), 100)
        for a in ASPECTS.values():
            self.assertEqual(set(a.tags), {"positive", "negative", "neutral"})
            self.assertNotEqual(a.tags["positive"], a.tags["negative"])

    def test_clause_spans(self):
        t = "Display is great, but battery dies. OK."
        cl = split_clauses(t)
        self.assertEqual([t[c.start:c.end] for c in cl], ["Display is great", "battery dies", "OK"])
        self.assertEqual([c.boundary for c in cl], ["start", "contrast", "sent"])


class TestLexicon(unittest.TestCase):
    def s(self, t):
        return score_clause(t)[0]

    def test_negation(self):
        self.assertLess(self.s("not good"), 0)
        self.assertGreater(self.s("not bad"), 0)
        self.assertGreater(self.s("very good"), self.s("good"))
        self.assertLess(self.s("it doesn't work"), -1)


GOLD = [  # hand-written sanity fixtures (NOT independent gold annotations)
    ("Amazing display but terrible battery life.", {("display", "positive"), ("battery", "negative")}),
    ("Great screen and terrible battery.", {("display", "positive"), ("battery", "negative")}),
    ("The keyboard is not good.", {("keyboard", "negative")}),
    ("The screen, keyboard and battery are great.", {("display", "positive"), ("keyboard", "positive"), ("battery", "positive")}),
    ("The order arrived late.", {("delivery", "negative")}),
    ("The battery died.", {("battery", "negative")}),
    ("Delicious taste and fresh.", {("taste", "positive"), ("freshness", "positive")}),
    ("The plot was boring but the characters were wonderful.", {("story", "negative"), ("characters", "positive")}),
    ("Not worth the money.", {("price", "negative"), ("value", "negative")}),
    ("The ring fits perfectly.", {("fit", "positive")}),
]


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = ReviewAnalyzer(None, allow_lexicon_only=True)

    def test_gold_fixtures(self):
        for text, exp in GOLD:
            got = {(m["normalized_aspect_id"], m["sentiment"]) for m in self.a.analyze(text)["mentions"]}
            self.assertEqual(got, exp, text)

    def test_spans_probabilities_and_text_unchanged(self):
        t = "  😀 Great screen!  "
        r = self.a.analyze(t)
        self.assertEqual(r["review_text"], t)
        for m in r["mentions"]:
            self.assertEqual(t[m["start_char"]:m["end_char"]], m["raw_aspect"])
            self.assertAlmostEqual(sum(m["probabilities"].values()), 1.0, places=5)
            self.assertTrue(0 <= m["confidence"] <= 1)
        self.assertFalse(r["probabilities_calibrated"])

    def test_no_aspect_is_valid(self):
        r = self.a.analyze("I enjoyed the afternoon.")
        self.assertEqual(r["mentions"], [])

    def test_validation(self):
        for bad in ["", "   ", None, 5, "x" * 10001]:
            with self.assertRaises(InvalidReview):
                self.a.analyze(bad)

    def test_deterministic_version_and_batch_limit(self):
        self.assertEqual(self.a.analyze("Good screen.")["mentions"], self.a.analyze("Good screen.")["mentions"])
        with self.assertRaises(InvalidReview):
            self.a.analyze_batch(["x y"] * 101)

    def test_fails_closed_without_model(self):
        with self.assertRaises(ModelUnavailable):
            ReviewAnalyzer(None)
        with self.assertRaises(ModelUnavailable):
            ReviewAnalyzer.from_path("/nonexistent/model.joblib")
        self.assertEqual(ReviewAnalyzer.from_path("/nonexistent/m.joblib", allow_lexicon_only=True).backend, "lexicon-only")

    def test_tags(self):
        self.assertEqual(make_tag("display", "positive"), "Great Display")
        self.assertEqual(make_tag("battery", "negative"), "Poor Battery")
        self.assertEqual(make_tag("display", "neutral"), "Neutral Display Feedback")
        self.assertIsNone(make_tag(None, "positive"))
        with self.assertRaises(ValueError):
            make_tag("display", "mixed")


if __name__ == "__main__":
    unittest.main()
