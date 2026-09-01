import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "swe-pruner" / "src"))

from swe_pruner.prune_wrapper import PruneRequest, SwePrunerForCodePruning


MODEL_PATH = os.environ.get(
    "SWEPRUNER_MODEL_PATH",
    str(REPO_ROOT / "runtime_models" / "swe-pruner-qwen-local"),
)
PORT = int(os.environ.get("SWEPRUNER_SMOKE_PORT", "8011"))

MODEL = SwePrunerForCodePruning.from_pretrained(MODEL_PATH)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "healthy", "model_loaded": True})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/prune":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            request = PruneRequest(**payload)
            response = MODEL.prune(request)
            self._send_json(200, response.model_dump())
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format, *args):  # noqa: A003
        return


def main() -> None:
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
