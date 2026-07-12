"""GET /api/status — tier map + which backends this deployment can reach."""
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import vercel_demo  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            payload = vercel_demo.status_payload()
        except Exception as error:
            payload = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
