"""Start the project UI: python serve.py  ->  http://localhost:8000"""
import functools, http.server, os, webbrowser
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PORT = int(os.environ.get("PORT", 8000))
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
with http.server.ThreadingHTTPServer(("", PORT), handler) as srv:
    print(f"Warehouse Distribution Optimization running at http://localhost:{PORT}  (Ctrl+C to stop)")
    webbrowser.open(f"http://localhost:{PORT}")
    srv.serve_forever()
