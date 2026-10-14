# ReviewRadar - aspect-based review analysis

Reads a review, finds product aspects (display, battery, taste, plot ...), decides the sentiment
for each aspect, and produces customer-friendly tags ("Great Display", "Poor Battery"). Aggregates
saved analyses into per-collection strengths/weaknesses. Local demo: binds to 127.0.0.1, no auth.

## Quick start
    pip install -r requirements.txt
    python -m reviewradar.train          # trains + evaluates; writes models/ (already included)
    python -m reviewradar.app --seed     # imports data/raw CSVs, serves http://127.0.0.1:8000
    python -m unittest discover -s tests -t .      # 41 tests (set PYTHONPATH=src:.)
`static/ReviewRadar_explorer.html` is your original standalone explorer (client-side lexicon engine), kept unchanged.

## NLP pipeline
| Stage | Method | File |
|---|---|---|
| Cleaning | strip HTML/entities/wrapper quotes; fix `negaitve` label typo; dedup by review id | data.py |
| Segmentation | clause splitter on the ORIGINAL text; offsets are Unicode code points | text.py |
| Aspect extraction | 38-aspect ontology, longest-alias-first matching, exact spans, repeats kept | ontology.py, aspects.py |
| Aspect sentiment | lexicon + negation window + intensifiers + phrase rules over the clause; comma-list borrowing; conditional "and" split | lexicon.py, pipeline.py |
| Review sentiment | TF-IDF word 1-2g + char 2-5g + lexicon features -> class-balanced logistic regression | classifier.py, train.py |
| Tags | deterministic per-aspect templates (pos/neg/neutral); unknown -> no tag | tags.py |
| Aggregation | one vote per review per aspect, support >= 3, score = ln(1+n) * conf * abs(pos-neg) | aggregate.py |

Safety: model load verifies a SHA-256 sidecar and the lexicon fingerprint and **fails closed**
(HTTP 503 `MODEL_UNAVAILABLE`); the lexicon-only mode exists only via explicit `allow_lexicon_only=True`.

## API (all errors use `{"error":{code,message,details,retryable},"request_id"}`)
GET /health, /ready | POST /api/reviews/analyze (ephemeral preview) | POST /api/products/{id}/reviews
(Idempotency-Key UUID required) | GET /api/products, /api/products/{id}/insights, /tags?top_k=, /reviews |
GET /api/reviews/{id} | GET /api/models/metrics. FastAPI: `uvicorn --factory reviewradar.app:create_fastapi_app`.

## Measured results (models/metrics.json, out-of-fold, 500 reviews: 398 pos / 102 neg)
Review-level sentiment only. Macro-F1: majority 0.443 | lexicon rule 0.859 | TF-IDF LR 0.819 | **hybrid 0.877**
(accuracy 0.919, negative recall 0.82, ROC-AUC 0.970). Leave-one-category-out hybrid macro-F1 0.879; the eBooks
fold is weakest (0.66).

## Honest limitations
- The lexicon came from your HTML explorer and was probably tuned on these same reviews; lexicon-based scores are
  likely optimistic for unseen data. The unseen-category protocol only partly mitigates this.
- Labels are binary. There is **no neutral class and no aspect-level gold data**, so aspect extraction and aspect
  sentiment are NOT measured against gold; only 10 hand-written fixtures test them (tests/test_nlp.py).
- Alias matching is not a learned extractor: unseen nouns are missed (~45% of seed reviews yield no aspect).
  The planned DistilBERT checkpoints, MiniLM normalization and PostgreSQL were not available in this workspace.
- Aspect probabilities are a fixed softmax of a clause score: they sum to 1 but are not calibrated.
- Persistence is SQLite with synchronous analysis (no separate worker/queue); schema mirrors the planned PostgreSQL one.
- Almost every item has one review, so the seed imports one collection per category and aggregates per collection.
- The FastAPI adapter is a thin wrapper over the tested router; FastAPI was not installable offline, so it was not run.
- The UI was syntax-checked and served over HTTP in tests but not viewed in a browser here.
