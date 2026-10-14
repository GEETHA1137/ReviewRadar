import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
MODEL = os.path.join(ROOT, "models", "review_sentiment.joblib")
RAW = os.path.join(ROOT, "data", "raw")
