#!/usr/bin/env python
"""Local web server for the interactive demo (frontend/demo.html) and the animated showcase (index.html).

Standard library only, bound to 127.0.0.1, no internet needed. The demo page calls two endpoints:
  GET  /api/setup    market scenarios, preset headlines and the options the page offers
  POST /api/analyze  run the agent with and without the defense layer on the posted headlines
                     (rhal.demo.Demo.analyze); errors come back as {"error": "..."} with status 400
Usage: make demo, or .venv/bin/python scripts/demo_server.py [--port 8765] [--no-browser]
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from _common import ROOT, base_parser, parse_overrides

from rhal.config import load_config
from rhal.demo import BIASES, PRESETS, READERS, RISK_MODES, Demo, DemoError

FRONTEND = ROOT / "frontend"
MAX_BODY = 64_000  # bytes; a request is a few headlines


class Handler(SimpleHTTPRequestHandler):
    demo: Demo  # set by main() before serving
    lock = threading.Lock()  # one analysis at a time (the LLM reader keeps an in-process memo)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def log_request(self, code="-", size="-") -> None:  # keep the console quiet apart from API calls and errors
        if self.path.startswith("/api/") or str(code)[:1] in ("4", "5"):
            super().log_request(code, size)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # always serve the current page and data
        super().end_headers()

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/setup":
            self._json(
                200,
                {
                    "scenarios": self.demo.scenario_payload(),
                    "presets": PRESETS,
                    "readers": list(READERS),
                    "biases": list(BIASES),
                    "risk_modes": list(RISK_MODES),
                    "llm_model": self.demo.llm_model,
                },
            )
            return
        if self.path in ("/", "/index"):
            self.path = "/demo.html"
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self._json(413, {"error": "request too large"})
            return
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(req, dict):
                raise ValueError
        except ValueError:
            self._json(400, {"error": "the request is not a JSON object"})
            return
        with self.lock:
            try:
                result = self.demo.analyze(req)
            except DemoError as e:
                self._json(400, {"error": str(e)})
                return
        self._json(200, result)


def main() -> None:
    p = base_parser("Interactive demo server")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=None, help="which simulated market to use (default: config seed)")
    p.add_argument("--llm-model", default="qwen2.5:7b", help="Ollama model for the LLM reader (make llm-up)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    print("preparing the simulated market ...", flush=True)
    Handler.demo = Demo(cfg, seed=a.seed, llm_model=a.llm_model)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError as e:
        raise SystemExit(f"cannot listen on port {a.port} ({e}); try --port 8766") from e
    url = f"http://127.0.0.1:{a.port}/"
    print(f"demo: {url}   (showcase: {url}index.html)   Ctrl+C to stop", flush=True)
    if not a.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
