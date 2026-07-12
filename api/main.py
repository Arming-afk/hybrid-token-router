"""Single Vercel entrypoint (pyproject.toml [tool.vercel]): serves the demo UI
at / and dispatches /api/status, /api/ask, /api/batch — same surface as
scripts/demo.py's Handler. The per-file api/*.py handlers stay for the classic
per-route functions model; this file covers the single-entrypoint model."""
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import vercel_demo  # noqa: E402

try:
    PAGE = (ROOT / "public" / "index.html").read_text(encoding="utf-8")
except FileNotFoundError:
    # Vercel strips public/ from the Python function bundle; the page also
    # ships embedded as code (regenerated from public/index.html by
    # scratchpad gen_page.py whenever the UI changes).
    from api._page import PAGE


class handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict) -> None:
        self._send(json.dumps(payload, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send_json(vercel_demo.status_payload())
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/ask":
                prompt = body.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    self._send_json({"error": "empty prompt"})
                    return
                self._send_json(vercel_demo.answer_question(prompt.strip()))
            elif path == "/api/batch":
                raw = body.get("prompts", [])
                prompt_list = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
                if not prompt_list:
                    self._send_json({"error": "no questions — one per line"})
                    return
                self._send_json(vercel_demo.answer_batch(prompt_list))
            else:
                self.send_error(404)
        except Exception as error:  # the UI must always get JSON, never a 500 page
            self._send_json({"error": f"{type(error).__name__}: {str(error)[:300]}"})
