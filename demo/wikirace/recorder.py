"""Shared live-article recording and paused-clock support for label experiments."""
import datetime
import hashlib
import json
import re
import time
import urllib.parse

import requests
from bs4 import BeautifulSoup

from .fetch import article_url, eligible_link, normalized, fetch_article


class RaceRecorder:
    def __init__(self, out, metadata, fetch_proxy=None):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=False)
        self.fetch_proxy = fetch_proxy
        (self.out / "pages").mkdir()
        self.events, self.pages, self.hops, self.route = [], [], [], []
        self.loading_intervals = []
        self.loading_total = 0.0
        self.loading_start = None
        self.visited = set()
        self.url = article_url("Baseball")
        self.metadata = {
            **metadata, "success": False, "model_seconds": 0.0,
            "timer_label": "Race time · page loads excluded",
            "loading_definition": "Each actual fetch including retries, parsing, page snapshot save and page_loaded event is paused, including initial and terminal page loads.",
            "wall_definition": "Immediately before fetching Baseball until the terminal page load and success/hop-cap decision. Final aggregate serialization, auditing and transfer occur after timing.",
            "model_time_definition": "Synchronized SystemOne API calls, including adapter prompt construction, tokenization, prefix/suffix model forwards and answer processing.",
            "load_and_warmup_excluded": True,
            "fetch_transport": "Loopback fetch bridge" if fetch_proxy else "Direct public Wikipedia HTTPS",
            "filter_policy": "Identical prior full #mw-content-text .mw-parser-output filter, including infobox, references and embedded navboxes; distinct article destinations in document order; no external hosts, fragments, queries/edit/red links, selflinks, colon namespaces or known visited canonical/requested titles.",
            "license": "Wikipedia text: CC BY-SA 4.0; https://creativecommons.org/licenses/by-sa/4.0/ ; each article/history URL provides attribution and contributor history.",
        }

    def start(self):
        self.metadata["started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.started = time.perf_counter()

    def event(self, phase, **extra):
        t = time.perf_counter() - self.started
        loading = self.loading_total + (t - self.loading_start if self.loading_start is not None else 0.0)
        item = {
            "t": t, "active_t": t - loading, "loading_seconds": loading,
            "paused": self.loading_start is not None, "phase": phase,
            "current_page": self.route[-1] if self.route else "Baseball",
            "hop": len(self.hops), "route": list(self.route),
            "model_seconds": self.metadata["model_seconds"], **extra,
        }
        self.events.append(item)
        with (self.out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)
        return item

    def close_loading(self):
        if self.loading_start is not None:
            end = time.perf_counter() - self.started
            item = {"start_t": self.loading_start, "end_t": end,
                    "seconds": end - self.loading_start, "requested_url": self.url}
            self.loading_total += item["seconds"]
            self.loading_intervals.append(item)
            self.loading_start = None

    def fetch_page(self):
        self.loading_start = time.perf_counter() - self.started
        self.event("fetching", requested_url=self.url)
        fetch_start = time.perf_counter()
        if self.fetch_proxy:
            response = requests.post(self.fetch_proxy, json={"url": self.url}, timeout=65)
            response.raise_for_status()
            fetched = response.json()
        else:
            fetched = fetch_article(self.url)
        fetch_seconds = time.perf_counter() - fetch_start
        if fetched["result"] is None:
            self.event("fetch_failed", attempts=fetched["attempts"])
            raise RuntimeError("Article fetch failed: " + repr(fetched["attempts"]))
        document = fetched["result"]
        parse_start = time.perf_counter()
        html = document["html"]
        soup = BeautifulSoup(html, "html.parser")
        canonical_node = soup.find("link", rel="canonical")
        canonical_url = canonical_node["href"] if canonical_node else document["url"]
        canonical = urllib.parse.urlsplit(canonical_url)
        if canonical.netloc != "en.wikipedia.org" or not canonical.path.startswith("/wiki/"):
            raise RuntimeError("Unexpected canonical article URL: " + canonical_url)
        title = normalized(canonical.path[len("/wiki/"):])
        heading = soup.find(id="firstHeading")
        body = soup.select_one("#mw-content-text .mw-parser-output")
        if body is None:
            raise RuntimeError("Full article body missing")
        self.route.append(title)
        self.visited.update((title, normalized(urllib.parse.urlsplit(self.url).path[len("/wiki/"):])))
        links, seen = [], set()
        for anchor in body.find_all("a", href=True):
            edge = eligible_link(anchor["href"], document["url"], title)
            if edge is None or edge[0] in seen:
                continue
            seen.add(edge[0])
            links.append({"title": edge[0], "url": edge[1], "href": anchor["href"],
                          "anchor": anchor.get_text(" ", strip=True)[:120],
                          "document_index": len(links)})
        eligible = [link for link in links if link["title"] not in self.visited]
        paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
        paragraphs = [p for p in paragraphs if p]
        digest = hashlib.sha256(html.encode()).hexdigest()
        page_id = f"{len(self.pages):02d}-{digest[:12]}"
        revision_match = re.search(r'"wgRevisionId"\s*:\s*(\d+)', html)
        page = {
            "id": page_id, "title": title,
            "display_title": heading.get_text(" ", strip=True) if heading else title,
            "url": canonical_url, "requested_url": self.url, "source_url": document["url"],
            "html_sha256": digest, "html_path": f"pages/{page_id}.html",
            "revision": int(revision_match.group(1)) if revision_match else None,
            "history_url": "https://en.wikipedia.org/w/index.php?title=" + urllib.parse.quote(title) + "&action=history",
            "leadtext": "\n\n".join(paragraphs)[:1500], "paragraphs": paragraphs,
            "all_links": links, "eligible_links": eligible,
            "fetch_seconds": fetch_seconds, "fetch_attempts": fetched["attempts"],
            "fetched_utc": document["fetched_utc"], "http_headers": document["headers"],
            "parse_seconds": time.perf_counter() - parse_start,
            "loading_interval_index": len(self.loading_intervals),
            "attribution": self.metadata["license"],
        }
        (self.out / page["html_path"]).write_text(html)
        (self.out / "pages" / f"{page_id}.json").write_text(json.dumps(page, ensure_ascii=False, indent=2))
        self.pages.append(page)
        self.event("page_loaded", current_page=title, page_id=page_id, eligible_count=len(eligible),
                   fetch_seconds=fetch_seconds, parse_seconds=page["parse_seconds"])
        self.close_loading()
        self.event("loading_complete", page_id=page_id)
        return page

    def validate_edge(self, page, chosen):
        body = BeautifulSoup((self.out / page["html_path"]).read_text(), "html.parser").select_one(
            "#mw-content-text .mw-parser-output"
        )
        if not any(a.get("href") == chosen["href"] and eligible_link(
            a["href"], page["source_url"], page["title"]
        ) == (chosen["title"], chosen["url"]) for a in body.find_all("a", href=True)):
            raise RuntimeError("Chosen edge failed source-HTML validation")

    def save(self):
        self.metadata.update(route=list(self.route), hops=len(self.hops),
                             loading_seconds=self.loading_total, loading_intervals=self.loading_intervals)
        (self.out / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        (self.out / "trace.json").write_text(json.dumps({
            "metadata": self.metadata, "events": self.events, "pages": self.pages, "hops": self.hops,
        }, ensure_ascii=False, indent=2))

    def finish(self):
        self.metadata["elapsed_seconds"] = time.perf_counter() - self.started
        self.metadata["loading_seconds"] = self.loading_total
        self.metadata["active_elapsed_seconds"] = self.metadata["elapsed_seconds"] - self.loading_total
        self.metadata["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.event("finished", success=self.metadata["success"],
                   elapsed_seconds=self.metadata["elapsed_seconds"],
                   active_elapsed_seconds=self.metadata["active_elapsed_seconds"])
        self.save()
        print("RESULT " + json.dumps(self.metadata), flush=True)
