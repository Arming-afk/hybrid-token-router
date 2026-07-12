"""POST /api/batch {"prompts": [str]} — Phase E batching demo over several questions."""
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import vercel_demo  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        try:
            raw = body.get("prompts", [])
            prompt_list = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
            if not prompt_list:
                payload = {"error": "no questions — one per line"}
            else:
                payload = vercel_demo.answer_batch(prompt_list)
        except Exception as error:  # the UI must always get JSON, never a 500 page
            payload = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
