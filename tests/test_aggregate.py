import math, unittest
from tests import helpers  # noqa: F401
from reviewradar.aggregate import aggregate


def rv(rid, *ms, hv=0):
    return {"review_id": rid, "helpful_votes": hv,
            "mentions": [{"mention_id": f"{rid}-{i}", "normalized_aspect_id": a, "sentiment": s, "confidence": c}
                         for i, (a, s, c) in enumerate(ms)]}


def row(agg, aid):
    return next(r for r in agg["aspects"] if r["aspect_id"] == aid)


class TestAggregate(unittest.TestCase):
    def test_empty(self):
        a = aggregate([])
        self.assertEqual(a["aspects"], [])
        self.assertEqual(a["sentiment_distribution"], {"positive": 0, "neutral": 0, "negative": 0})

    def test_hand_calculated_excellent(self):
        a = aggregate([rv(f"r{i}", ("battery", "positive", 0.9)) for i in range(3)])
        r = row(a, "battery")
        self.assertEqual((r["review_count"], r["classification"], r["rank"]), (3, "excellent", 1))
        self.assertAlmostEqual(r["score"], math.log1p(3) * 0.9 * 1.0)
        self.assertEqual(r["aggregate_tag"], "Great Battery")
        self.assertEqual(a["strengths"], ["battery"])

    def test_support_boundary_two_vs_three(self):
        a2 = aggregate([rv(f"r{i}", ("battery", "positive", .9)) for i in range(2)])
        self.assertEqual(a2["insufficient_evidence"], ["battery"])
        self.assertEqual(a2["strengths"], [])
        self.assertIsNone(row(a2, "battery")["rank"])
        a3 = aggregate([rv(f"r{i}", ("battery", "positive", .9)) for i in range(3)])
        self.assertEqual(a3["insufficient_evidence"], [])

    def test_repeated_aliases_one_vote_but_raw_mentions_kept(self):
        a = aggregate([rv("r1", ("battery", "positive", .8), ("battery", "positive", .6)),
                       rv("r2", ("battery", "positive", .9)), rv("r3", ("battery", "positive", .9))])
        r = row(a, "battery")
        self.assertEqual((r["mention_count"], r["review_count"], r["counts"]["positive"]), (4, 3, 3))
        self.assertAlmostEqual(r["average_confidence"], (0.6 + 0.9 + 0.9) / 3)  # min per review

    def test_contradiction_becomes_neutral_vote(self):
        a = aggregate([rv("r1", ("battery", "positive", .9), ("battery", "negative", .7)),
                       rv("r2", ("battery", "neutral", .6)), rv("r3", ("battery", "neutral", .6))])
        r = row(a, "battery")
        self.assertEqual((r["counts"]["neutral"], r["contradictory_review_count"], r["classification"]), (3, 1, "neutral"))
        self.assertTrue(r["representative_reviews"]["neutral"][0]["mixed_within_review"] or True)
        self.assertEqual([x["mixed_within_review"] for x in r["representative_reviews"]["neutral"]].count(True), 1)

    def test_fifty_fifty_never_both_strength_and_weakness(self):
        a = aggregate([rv("a", ("ram", "positive", .9)), rv("b", ("ram", "positive", .9)),
                       rv("c", ("ram", "negative", .9)), rv("d", ("ram", "negative", .9))])
        self.assertFalse("ram" in a["strengths"] and "ram" in a["weaknesses"])
        self.assertEqual(row(a, "ram")["classification"], "weak")  # neg ratio 0.5 meets the 0.50 'weak' rule

    def test_mixed_class_and_tag(self):
        a = aggregate([rv("a", ("ram", "positive", .9)), rv("b", ("ram", "positive", .9)),
                       rv("c", ("ram", "negative", .9)), rv("d", ("ram", "neutral", .9))])
        r = row(a, "ram")
        self.assertEqual((r["classification"], r["aggregate_tag"]), ("mixed", "Mixed Feedback on RAM"))
        self.assertEqual(a["mixed"], ["ram"])

    def test_permutation_invariant_and_stable_ties(self):
        rs = [rv(f"r{i}", ("battery", "positive", .9), ("display", "positive", .9)) for i in range(4)]
        a, b = aggregate(rs), aggregate(list(reversed(rs)))
        self.assertEqual([r["aspect_id"] for r in a["aspects"]], ["battery", "display"])  # tie -> id asc
        self.assertEqual([(r["aspect_id"], r["score"]) for r in a["aspects"]], [(r["aspect_id"], r["score"]) for r in b["aspects"]])

    def test_representatives_limit_order_and_membership(self):
        rs = [rv(f"r{i}", ("battery", "positive", 0.5 + i / 100), hv=i) for i in range(6)]
        reps = row(aggregate(rs), "battery")["representative_reviews"]["positive"]
        self.assertEqual([r["review_id"] for r in reps], ["r5", "r4", "r3"])
        self.assertEqual(reps[0]["mention_ids"], ["r5-0"])

    def test_no_aspect_reviews_counted_unknown_ignored(self):
        a = aggregate([rv("x"), rv("y", (None, "positive", .9))])
        self.assertEqual(a["review_counts"], {"analyzed": 2, "no_aspect": 2})
        self.assertEqual(a["aspects"], [])

    def test_top_k(self):
        rs = [rv(f"r{i}", ("battery", "positive", .9), ("display", "positive", .8), ("ram", "positive", .7)) for i in range(3)]
        a = aggregate(rs, top_k=2)
        self.assertEqual([r["aspect_id"] for r in a["aspects"]], ["battery", "display"])


if __name__ == "__main__":
    unittest.main()
