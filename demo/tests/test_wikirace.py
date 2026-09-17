import contextlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid
from types import SimpleNamespace

from demo.wikirace.fetch import article_url, eligible_link, valid_article_url
from demo.wikirace.recorder import RaceRecorder
from demo.wikirace.render import Tournament100Renderer
from demo.wikirace.run import run_race
from demo.wikirace.audit import audit
from demo.labels import LabelSystemOne
from demo.tests.test_labels import Model, Tokenizer


class WikiTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("demo") / f".test-{uuid.uuid4().hex}"
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_article_link_filter(self):
        source = article_url("Baseball")
        self.assertEqual(eligible_link("/wiki/Amateur_astronomy", source, "Baseball"),
                         ("Amateur astronomy", article_url("Amateur astronomy")))
        for href in ("#Sun", "/wiki/Sun#History", "/wiki/Sun?oldid=1",
                     "/wiki/File:Sun.jpg", "/wiki/Baseball", "https://example.org/wiki/Sun",
                     "http://en.wikipedia.org/wiki/Sun", "/w/index.php?title=Sun"):
            with self.subTest(href=href):
                self.assertIsNone(eligible_link(href, source, "Baseball"))
        self.assertFalse(valid_article_url("https://en.wikipedia.org.evil.test/wiki/Sun"))
        self.assertFalse(valid_article_url("https://en.wikipedia.org/wiki/Sun?x=1"))

    def test_direct_fetch_records_html_edges_and_pauses(self):
        recorder = RaceRecorder(self.root, {"start": "Baseball", "target": "Sun"})
        html = ('<link rel="canonical" href="https://en.wikipedia.org/wiki/Baseball">'
                '<h1 id="firstHeading">Baseball</h1><div id="mw-content-text">'
                '<div class="mw-parser-output"><p>Example article with '
                '<a href="/wiki/Sun">Sun</a> and a duplicate '
                '<a href="/wiki/Sun">star</a><a href="/wiki/Baseball">self</a>'
                '</p></div></div>')
        fetched = {"result": {"html": html, "url": article_url("Baseball"),
                             "fetched_utc": "2026-09-17T00:00:00+00:00", "headers": {}},
                   "attempts": []}
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("demo.wikirace.recorder.time.perf_counter", side_effect=range(20)), \
                patch("demo.wikirace.recorder.fetch_article", return_value=fetched) as fetch:
            recorder.start()
            page = recorder.fetch_page()
            recorder.validate_edge(page, page["eligible_links"][0])
            recorder.finish()
        fetch.assert_called_once_with(article_url("Baseball"))
        self.assertEqual([link["title"] for link in page["eligible_links"]], ["Sun"])
        self.assertEqual(recorder.loading_total, 7)
        self.assertEqual(recorder.metadata["elapsed_seconds"], 10)
        self.assertEqual(recorder.metadata["active_elapsed_seconds"], 3)
        self.assertTrue((self.root / page["html_path"]).is_file())
        self.assertEqual(recorder.events[0]["phase"], "fetching")
        self.assertTrue(recorder.events[0]["paused"])
        self.assertFalse(recorder.events[-1]["paused"])

    def test_failed_load_closes_timer(self):
        recorder = RaceRecorder(self.root, {})
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("demo.wikirace.recorder.time.perf_counter", side_effect=range(20)), \
                patch("demo.wikirace.recorder.fetch_article",
                      return_value={"result": None, "attempts": [{"error": "denied"}]}):
            recorder.start()
            with self.assertRaisesRegex(RuntimeError, "Article fetch failed"):
                recorder.fetch_page()
            recorder.close_loading()
            recorder.finish()
        self.assertIsNone(recorder.loading_start)
        self.assertGreater(recorder.loading_total, 0)

    def test_clock_boundaries_and_offline_frames(self):
        self.root.mkdir()
        meta = {
            "model": "Qwen/Qwen3-8B", "start": "Baseball", "target": "Sun",
            "success": False, "hops": 0, "route": ["Baseball"], "group_size": 100,
            "elapsed_seconds": 10, "active_elapsed_seconds": 5,
            "loading_seconds": 5, "model_seconds": 0,
            "loading_intervals": [{"start_t": 0, "end_t": 3}, {"start_t": 6, "end_t": 8}],
        }
        events = [
            {"t": 0, "active_t": 0, "paused": True, "phase": "fetching"},
            {"t": 3, "active_t": 0, "paused": False, "phase": "loading_complete"},
            {"t": 6, "active_t": 3, "paused": True, "phase": "fetching"},
            {"t": 8, "active_t": 3, "paused": False, "phase": "loading_complete"},
            {"t": 10, "active_t": 5, "paused": False, "phase": "finished"},
        ]
        trace = {"metadata": meta, "events": events, "pages": [], "hops": []}
        (self.root / "trace.json").write_text(json.dumps(trace))
        renderer = Tournament100Renderer(self.root)
        for t, expected in ((0, (0, True)), (2, (0, True)), (3, (0, False)),
                            (5, (2, False)), (6, (3, True)), (8, (3, False)),
                            (10, (5, False)), (16, (5, False))):
            self.assertEqual(renderer.active_clock(t), expected)
        for t in (0, 5, 7, 10, 16):
            self.assertEqual(renderer.frame(t).size, (1920, 1080))
        self.assertEqual(renderer.frame(10).tobytes(), renderer.frame(16).tobytes())

    def test_full_tournament_capture_audit_and_group_display(self):
        engine = LabelSystemOne(Model(), Tokenizer())
        records = []

        def record(module, args, kwargs, output):
            records.append({"input_shape": list(kwargs["input_ids"].shape),
                            "logits_shape": list(output.logits.shape),
                            "use_cache": False, "past_key_values": False})

        hook = engine.model.register_forward_hook(record, with_kwargs=True)
        self.addCleanup(hook.remove)

        def fetch(url):
            article = "Baseball" if url == article_url("Baseball") else "Sun"
            links = "".join(f'<a href="/wiki/Candidate{i}">Candidate{i}</a>' for i in range(100))
            html = (f'<link rel="canonical" href="{article_url(article)}">'
                    f'<h1 id="firstHeading">{article}</h1><div id="mw-content-text">'
                    f'<div class="mw-parser-output"><p>Saved synthetic test article.</p>'
                    f'{links}<a href="/wiki/Sun">Sun</a></div></div>')
            return {"result": {"html": html, "url": url, "fetched_utc": "test", "headers": {}},
                    "attempts": []}

        # This fake model exercises the inherited numeric-to-name mapping path;
        # cache support itself is tested by the unchanged core's real tiny models.
        original = engine.system_one
        with contextlib.redirect_stdout(io.StringIO()), \
                patch("demo.wikirace.run.capacity_test", return_value=(2, [])), \
                patch("demo.wikirace.run.torch.cuda.synchronize"), \
                patch("demo.wikirace.recorder.fetch_article", side_effect=fetch), \
                patch.object(engine, "system_one",
                             side_effect=lambda state, questions, **kwargs: original(state, questions)):
            result = run_race(engine, records, SimpleNamespace(
                output=self.root, model="Qwen/Qwen3-8B", revision="synthetic",
                max_hops=25, fetch_proxy=None,
            ))
            audit(self.root, engine.tokenizer)
        self.assertTrue(result["success"])
        self.assertEqual(result["route"], ["Baseball", "Sun"])
        trace = json.loads((self.root / "trace.json").read_text())
        rounds = trace["hops"][0]["rounds"]
        self.assertEqual([r["candidate_count"] for r in rounds], [101, 2])
        self.assertEqual([len(g["candidates"]) for g in rounds[0]["groups"]], [100, 1])
        self.assertTrue(json.loads((self.root / "audit.json").read_text())["verified"])
        renderer = Tournament100Renderer(self.root)
        for event in trace["events"]:
            self.assertEqual(renderer.frame(event["t"]).size, (1920, 1080))


if __name__ == "__main__":
    unittest.main()
