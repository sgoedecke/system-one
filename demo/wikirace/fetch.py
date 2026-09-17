"""Direct Wikipedia fetching, or an optional loopback-only fetch bridge."""
import argparse
import datetime
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def normalized(title):
    return urllib.parse.unquote(title).replace("_", " ").strip()


def article_url(title):
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="()")


def eligible_link(href, source_url, source_title):
    if not href or "#" in href:
        return None
    parsed = urllib.parse.urlsplit(urllib.parse.urljoin(source_url, href))
    if parsed.scheme != "https" or parsed.netloc != "en.wikipedia.org":
        return None
    if not parsed.path.startswith("/wiki/") or parsed.query:
        return None
    title = normalized(parsed.path[len("/wiki/"):])
    if not title or ":" in title or title == source_title:
        return None
    return title, article_url(title)


def valid_article_url(url):
    parsed = urllib.parse.urlsplit(url)
    return (parsed.scheme == "https" and parsed.netloc == "en.wikipedia.org"
            and parsed.path.startswith("/wiki/") and not parsed.query and not parsed.fragment)


class WikipediaRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not valid_article_url(newurl):
            raise ValueError("Redirect left the permitted Wikipedia article URLs")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_article(url):
    if not valid_article_url(url):
        raise ValueError("Only HTTPS en.wikipedia.org/wiki/ article URLs are allowed")
    attempts, result = [], None
    opener = urllib.request.build_opener(WikipediaRedirects())
    for attempt in range(2):
        start = time.perf_counter()
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": "SystemOneWikiraceDemo/1.0 educational research",
                "Accept": "text/html",
            })
            with opener.open(request, timeout=25) as response:
                result = {
                    "html": response.read().decode("utf-8"), "url": response.url,
                    "status": response.status, "headers": dict(response.headers),
                    "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
            attempts.append({"attempt": attempt + 1, "seconds": time.perf_counter() - start,
                             "status": result["status"]})
            break
        except Exception as exc:
            attempts.append({"attempt": attempt + 1, "seconds": time.perf_counter() - start,
                             "error": repr(exc)})
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 429):
                break
            if attempt == 0:
                time.sleep(1)
    return {"result": result, "attempts": attempts}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"wikirace-fetch-ready")

    def do_POST(self):
        if self.path != "/fetch":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            url = request["url"]
            if not isinstance(url, str) or not valid_article_url(url):
                raise ValueError("Invalid article URL")
        except (ValueError, KeyError, TypeError):
            self.send_error(400, "Expected a Wikipedia article URL")
            return
        body = json.dumps(fetch_article(url)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Bridge ready on http://127.0.0.1:{args.port}/fetch", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
