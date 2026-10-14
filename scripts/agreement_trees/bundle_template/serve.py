"""Serve the site in ./site on http://localhost:__PORT__/ (Ctrl+C to stop).

Opens straight to decision_trees/, the overview of every condition.

    python3 serve.py            # default port __PORT__
    python3 serve.py 9000       # another port
"""
import functools
import http.server
import os
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else __PORT__


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    handler = functools.partial(Handler, directory=os.path.join(HERE, "site"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    url = f"http://localhost:{PORT}/decision_trees/"
    print(f"Serving {url}  (Ctrl+C to stop)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
