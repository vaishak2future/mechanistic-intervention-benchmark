"""Keep a warmed vLLM engine alive; sleep between requests to free training VRAM."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from eval import evaluate, load_tasks, result_report
from prepare import warm_engine

ADDRESS = "http://127.0.0.1:8765"


def serve():
    engine = warm_engine()
    engine.sleep(level=1)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ready": true}')

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                engine.wake_up()
                try:
                    options = argparse.Namespace(n=request.get("n", 4), max_tokens=3072)
                    results = evaluate(load_tasks(request["tasks"]), options, model=engine)
                    report = result_report(results)
                finally:
                    engine.sleep(level=1)
                body = json.dumps(report).encode()
                self.send_response(200)
            except Exception as error:
                body = json.dumps({"error": str(error)}).encode()
                self.send_response(500)
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=["/task/tasks.json"])
    parser.add_argument("--report", type=Path, default=Path("/workspace/baseline.json"))
    options = parser.parse_args()
    if options.serve:
        serve()
    elif options.check:
        with urlopen(ADDRESS, timeout=5) as response:
            assert json.load(response)["ready"]
    else:
        request = Request(ADDRESS, data=json.dumps({"tasks": options.tasks}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=3600) as response:
            report = json.load(response)
        options.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Score: {report['score']:.3f}. Full report: {options.report}")


if __name__ == "__main__":
    main()
