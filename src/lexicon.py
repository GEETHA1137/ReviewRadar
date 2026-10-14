"""Auditable clause-level polarity scoring: lexicon + negation + intensifiers + phrases.

Weights are ported from the lexicon engine in ReviewRadar.html and extended with
a few electronics terms. Every score is a sum of visible contributions.
"""
from __future__ import annotations
import hashlib, json, re


def _parse(s: str) -> dict[str, float]:
    return {w: float(v) for w, v in (p.split(":") for p in s.split())}


POSITIVE = _parse("""
great:1 good:.7 love:1 loved:1 loves:1 excellent:1.2 perfect:1.2 perfectly:1.1 beautiful:1 nice:.7 happy:.9
pleased:.9 amazing:1.2 awesome:1.2 wonderful:1.1 fantastic:1.2 best:1.1 favorite:1 recommend:.9 recommended:.9
comfortable:.8 sturdy:.8 easy:.7 cute:.8 pretty:.8 gorgeous:1.1 delicious:1.1 fresh:.7 smooth:.6 worth:.7
enjoyed:.9 enjoy:.8 impressed:.9 fine:.4 solid:.7 lovely:1 helpful:.8 useful:.8 informative:.8 stunning:1.1
adorable:.9 satisfied:.8 flawless:1.2 reliable:.8 quick:.5 yummy:1 tasty:.9 super:.8 superb:1.2 glad:.7
delighted:1 like:.3 likes:.4 liked:.5 works:.6 worked:.5 well:.4 fast:.5 bargain:.8 compliments:.8 sparkles:.7
thanks:.3 thank:.3 exceptional:1.1 interesting:.6 entertaining:.8 gripping:.9 engaging:.8 unique:.6 charming:.8
gem:.8 incredible:1 wow:.8 neat:.6 handy:.7 clean:.4 sweet:.4 secure:.5 safe:.4 prompt:.5 promptly:.5
inspiring:.9 favorites:.9 adore:1 fun:.7 funny:.5 hilarious:.7 better:.5 stronger:.4 improved:.5
lightweight:.6 durable:.8 crisp:.7 bright:.6 responsive:.7 powerful:.8 affordable:.7 cheaper:.4
""")
NEGATIVE = _parse("""
bad:-1 poor:-1 terrible:-1.3 awful:-1.3 horrible:-1.3 disappointed:-1.1 disappointing:-1.1 disappointment:-1.1
waste:-1.1 broke:-1 broken:-1 cheap:-.6 junk:-1.2 garbage:-1.3 worst:-1.4 hate:-1.2 hated:-1.2 defective:-1.2
fail:-1 failed:-1 failure:-1 fell:-.8 falls:-.8 cracked:-1 stale:-1 rancid:-1.2 disgusting:-1.3 tasteless:-1
bland:-.8 slow:-.7 noisy:-.6 tinny:-.9 flimsy:-.9 fragile:-.6 ugly:-1 useless:-1.2 boring:-.9 predictable:-.7
flat:-.6 forgettable:-.8 overpriced:-1 pricey:-.7 expensive:-.5 small:-.3 smaller:-.4 snug:-.4 wrong:-.9
missing:-.8 damaged:-1 dented:-.8 returned:-.5 refund:-.6 regret:-1 dead:-1 froze:-.8 freezing:-.8 buggy:-1
misleading:-1 nasty:-1.1 lame:-.9 dull:-.7 mediocre:-.7 annoying:-.8 painful:-1 dangerous:-1 stopped:-.7
sorry:-.4 unfortunately:-.5 unhappy:-.9 issue:-.5 issues:-.6 problem:-.6 problems:-.7 complaint:-.5 scam:-1.3
fake:-.9 chipped:-.8 stuck:-.5 loose:-.4 uncomfortable:-.8 overrated:-.8 unreliable:-.9 burnt:-.8 weak:-.7
ripoff:-1.2 lacking:-.5 sloppy:-.8 pathetic:-1.1 trash:-1.2 nonsense:-.9 simplistic:-.6 filler:-.7 late:-.4
drains:-.9 drained:-.8 dies:-.8 overheats:-1 overheating:-1 lags:-.8 laggy:-.9 blurry:-.9 grainy:-.7
unresponsive:-1 crashes:-.9 crashed:-.9 shattered:-1 scratched:-.7 faded:-.6 rusted:-.9 peeling:-.8
tarnished:-.9 heavy:-.4 dim:-.6 typos:-.7 died:-.8 dying:-.8
""")
LEX: dict[str, float] = {**POSITIVE, **NEGATIVE}

NEGATORS = frozenset("""not no never don't doesn't didn't isn't wasn't can't cannot won't wouldn't couldn't
without hardly barely nothing none neither nor aren't weren't haven't hasn't""".split())
BOOST = frozenset("very really extremely absolutely totally highly so incredibly truly especially quite".split())
DAMP = frozenset("slightly somewhat bit little kinda rather fairly mildly".split())

# (regex, weight): matched text is removed before token scoring
PHRASES = [
    (re.compile(p, re.I), w) for p, w in [
        (r"\bwaste of (?:money|time)\b", -1.2), (r"\bnot worth(?: it| the money)?\b", -1.0),
        (r"\btoo (?:high|much|expensive|pricey)\b", -1.0), (r"\bfive stars?\b|\b5 stars?\b", 1.2),
        (r"\bone star\b|\b1 star\b", -1.2), (r"\bso[- ]so\b", -0.5), (r"\bnot (?:bad|terrible)\b", 0.5),
        (r"\bjust (?:ok|okay)\b|\bokay\b|\bok\b", -0.2), (r"\bhighly recommend(?:ed)?\b", 1.2),
        (r"\bwould (?:definitely )?(?:buy|order) again\b", 1.1),
        (r"\b(?:will not|won't|never) (?:buy|order) again\b", -1.2),
        (r"\b(?:doesn't|didn't|does not|did not|won't|will not|stopped|never) work(?:ed|ing)?\b", -1.1),
        (r"\bnone of (?:them|these|it|those) (?:worked|work|works)\b", -1.3),
        (r"\bfell (?:out|apart)\b|\bcame apart\b", -1.0), (r"\bcouldn't put (?:it|this) down\b", 1.1),
    ]
]
_TOKEN = re.compile(r"[a-z][a-z'\-]*|\d+")
DEFAULTS = {"band": 0.4, "negwin": 3, "boost": 1.5, "damp": 0.6, "neg_flip": 0.8}


def lexicon_fingerprint() -> str:
    blob = json.dumps({"lex": LEX, "neg": sorted(NEGATORS), "boost": sorted(BOOST), "damp": sorted(DAMP),
                       "phr": [(p.pattern, w) for p, w in PHRASES], "p": DEFAULTS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def score_clause(text: str, p: dict | None = None) -> tuple[float, list[dict]]:
    """Return (score, contributions). Negation flips and damps; boosters scale the next hit."""
    p = {**DEFAULTS, **(p or {})}
    s = text.lower().replace("\u2019", "'")
    contribs: list[dict] = []
    for rx, w in PHRASES:
        def _sub(m, w=w):
            contribs.append({"term": m.group(0).strip(), "value": w, "kind": "phrase"})
            return " "
        s = rx.sub(_sub, s)
    neg_left, mult, mult_left = 0, 1.0, 0
    for tok in _TOKEN.findall(s):
        if tok in NEGATORS:
            neg_left = p["negwin"]
            continue
        if tok in BOOST:
            mult, mult_left = p["boost"], 2
            continue
        if tok in DAMP:
            mult, mult_left = p["damp"], 2
            continue
        base = LEX.get(tok)
        if base is not None:
            v = base * (mult if mult_left > 0 else 1.0)
            negated = neg_left > 0
            if negated:
                v = -v * p["neg_flip"]
            contribs.append({"term": ("not " if negated else "") + tok, "value": round(v, 3), "kind": "word"})
            mult_left = 0
        neg_left = max(0, neg_left - 1)
        mult_left = max(0, mult_left - 1)
    return round(sum(c["value"] for c in contribs), 3), contribs
