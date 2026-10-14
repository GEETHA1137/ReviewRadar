"""Wiring: build the Service, then serve it with FastAPI (if installed) or the stdlib server.
Binds to 127.0.0.1 by default: this is a local demo with no authentication."""
from __future__ import annotations
import argparse, os, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .classifier import ModelUnavailable
from .pipeline import ReviewAnalyzer
from .service import MAX_BODY_BYTES, Service
from .store import Store

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX = os.path.join(ROOT, "static", "index.html")


def build_service(db=None, model=None, metrics=None, allow_lexicon_only=False) -> Service:
    db = db or os.environ.get("RR_DB", os.path.join(ROOT, "data", "reviewradar.sqlite3"))
    model = model or os.environ.get("RR_MODEL", os.path.join(ROOT, "models", "review_sentiment.joblib"))
    metrics = metrics or os.environ.get("RR_METRICS", os.path.join(ROOT, "models", "metrics.json"))
    try:  # fail closed: no silent rule fallback unless explicitly requested
        analyzer = ReviewAnalyzer.from_path(model, allow_lexicon_only=allow_lexicon_only)
    except ModelUnavailable:
        analyzer = None  # /ready and analysis endpoints answer 503 MODEL_UNAVAILABLE
    return Service(Store(db), analyzer, metrics)


def _index_bytes() -> bytes:
    with open(INDEX, "rb") as fh:
        return fh.read()


def make_handler(svc: Service):
    class H(BaseHTTPRequestHandler):
        def _send(self, status, body: bytes, ctype, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _handle(self):
            path, _, qs = self.path.partition("?")
            if self.command == "GET" and path == "/":
                return self._send(200, _index_bytes(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"})
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                status, payload, h = 411, {"error": {"code": "VALIDATION_ERROR", "message": "Content-Length required.", "details": [], "retryable": False}, "request_id": ""}, {}
            else:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY_BYTES:  # reject before buffering the body
                    status, payload, h = svc.dispatch(self.command, path, qs, b"x" * (MAX_BODY_BYTES + 1), dict(self.headers))
                else:
                    status, payload, h = svc.dispatch(self.command, path, qs, self.rfile.read(n) if n else b"", dict(self.headers))
            self._send(status, json.dumps(payload).encode(), "application/json", h)

        do_GET = do_POST = _handle

        def log_message(self, *a):  # never log request bodies
            pass
    return H


def serve(svc: Service, host="127.0.0.1", port=8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(svc))


def create_fastapi_app(svc: Service | None = None):
    """Thin FastAPI adapter around Service.dispatch (all logic/validation lives in Service)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    svc = svc or build_service()
    app = FastAPI(title="ReviewRadar API", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(_index_bytes().decode("utf-8"))

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def route(full_path: str, request: Request):
        status, payload, headers = svc.dispatch(request.method, "/" + full_path, request.url.query,
                                                await request.body(), dict(request.headers))
        return JSONResponse(payload, status_code=status, headers=headers)
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="ReviewRadar local server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", action="store_true", help="import data/raw CSVs first")
    a = ap.parse_args(argv)
    svc = build_service()
    if a.seed:
        from .seed import seed
        print(seed(svc, os.path.join(ROOT, "data", "raw")))
    print(f"serving on http://{a.host}:{a.port}  backend={svc.analyzer.backend if svc.analyzer else 'UNAVAILABLE'}")
    serve(svc, a.host, a.port).serve_forever()


if __name__ == "__main__":
    main()
