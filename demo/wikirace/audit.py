"""Audit exact token labels, complete tournaments, HTML edges and paused clocks."""
import hashlib
import json
import math
import argparse
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from .fetch import eligible_link, normalized


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-6), (a, b)


def audit(directory, tokenizer):
    root = Path(directory)
    trace = json.loads((root / "trace.json").read_text())
    metadata, pages, hops, events = (trace[key] for key in ("metadata", "pages", "hops", "events"))
    mapping = json.loads((root / "labelmap.json").read_text())
    labels = mapping["labels"]
    assert len(labels) == len({item["token_id"] for item in labels}) == len({item["label"] for item in labels}) == 100
    prefix = tokenizer.encode(mapping["prefix"], add_special_tokens=False)
    assert prefix == mapping["prefix_token_ids"]
    for index, item in enumerate(labels):
        assert item["index"] == index
        assert tokenizer.encode(item["label"], add_special_tokens=False) == [item["token_id"]]
        assert item["token_id"] not in tokenizer.all_special_ids
        assert tokenizer.decode(prefix + [item["token_id"]], clean_up_tokenization_spaces=False) == mapping["prefix"] + item["label"]
    visited = set()
    for page in pages:
        html = (root / page["html_path"]).read_text()
        assert hashlib.sha256(html.encode()).hexdigest() == page["html_sha256"]
        soup = BeautifulSoup(html, "html.parser")
        assert soup.find("link", rel="canonical")["href"] == page["url"]
        body = soup.select_one("#mw-content-text .mw-parser-output")
        assert body is not None
        found, seen = [], set()
        for anchor in body.find_all("a", href=True):
            edge = eligible_link(anchor["href"], page["source_url"], page["title"])
            if edge and edge[0] not in seen:
                seen.add(edge[0])
                found.append(edge)
        assert found == [(link["title"], link["url"]) for link in page["all_links"]]
        visited.update((page["title"], normalized(urlsplit(page["requested_url"]).path[len("/wiki/"):])))
        assert [title for title, url in found if title not in visited] == [
            link["title"] for link in page["eligible_links"]
        ]
    model_total = 0.0
    questions_total = 0
    calls_total = 0
    forwards_total = 0
    for index, hop in enumerate(hops):
        page = pages[index]
        assert hop["source_page_id"] == page["id"]
        assert hop["eligible_count"] == len(page["eligible_links"])
        expected = page["eligible_links"]
        hop_model = 0.0
        for round_data in hop["rounds"]:
            assert round_data["candidate_count"] == len(expected)
            assert round_data["group_count"] == len(round_data["groups"])
            expected_groups = [expected[i:i + metadata["group_size"]]
                               for i in range(0, len(expected), metadata["group_size"])]
            assert [group["candidates"] for group in round_data["groups"]] == expected_groups
            for group in round_data["groups"]:
                assert 1 <= len(group["candidates"]) <= metadata["group_size"]
                answer = group["answer"]
                names = [link["title"] for link in group["candidates"]]
                assert list(answer["probabilities"]) == names
                close(sum(answer["probabilities"].values()), 1.0)
                assert all(0 <= p <= 1 for p in answer["probabilities"].values())
                assert max(answer["probabilities"], key=answer["probabilities"].get) == group["winner"] == answer["choice"]
                selected = names.index(group["winner"])
                assert selected == group["selected_index"]
                assert group["selected_label"] == labels[selected]["label"]
                assert group["selected_token_id"] == labels[selected]["token_id"]
            covered = []
            for batch in round_data["batches"]:
                covered.extend(range(batch["batch_start"], batch["batch_start"] + batch["questions"]))
                assert batch["usage"]["output_tokens"] == batch["questions"]
                assert 1 <= len(batch["forwards"]) <= 2
                assert all(record["logits_shape"][1] == 1 for record in batch["forwards"])
                assert batch["forwards"][-1]["logits_shape"][0] == batch["questions"]
                hop_model += batch["seconds"]
                questions_total += batch["questions"]
                calls_total += 1
                forwards_total += len(batch["forwards"])
            assert covered == list(range(len(round_data["groups"])))
            expected = [group["candidates"][group["selected_index"]] for group in round_data["groups"]]
        assert expected == [hop["selected_link"]]
        assert hop["selected_link"] in page["eligible_links"]
        assert hop["edge_validated"]
        assert hop["source_html_sha256"] == page["html_sha256"]
        assert pages[index + 1]["requested_url"] == hop["destination_url"]
        html = (root / page["html_path"]).read_text()
        body = BeautifulSoup(html, "html.parser").select_one("#mw-content-text .mw-parser-output")
        chosen = hop["selected_link"]
        assert any(a.get("href") == chosen["href"] and eligible_link(
            a["href"], page["source_url"], page["title"]
        ) == (chosen["title"], chosen["url"]) for a in body.find_all("a", href=True))
        hop_model += sum(event["call_seconds"] for event in events
                         if event["phase"] == "model_oom" and event["hop"] == index)
        close(hop_model, hop["model_seconds"])
        model_total += hop_model
    close(model_total, metadata["model_seconds"])
    intervals = metadata["loading_intervals"]
    assert all(0 <= item["start_t"] <= item["end_t"] for item in intervals)
    assert all(a["end_t"] <= b["start_t"] for a, b in zip(intervals, intervals[1:]))
    for item in intervals:
        close(item["seconds"], item["end_t"] - item["start_t"])
    close(sum(item["seconds"] for item in intervals), metadata["loading_seconds"])
    close(metadata["active_elapsed_seconds"], metadata["elapsed_seconds"] - metadata["loading_seconds"])
    assert all(a["t"] <= b["t"] for a, b in zip(events, events[1:]))
    assert all(a["active_t"] <= b["active_t"] + 1e-6 for a, b in zip(events, events[1:]))
    for event in events:
        loading = sum(max(0, min(event["t"], item["end_t"]) - item["start_t"]) for item in intervals)
        close(event["loading_seconds"], loading)
        close(event["active_t"], event["t"] - loading)
        assert event["paused"] == any(item["start_t"] <= event["t"] < item["end_t"] for item in intervals)
    for page in pages:
        loaded = next(event for event in events if event["phase"] == "page_loaded" and event["page_id"] == page["id"])
        interval = intervals[page["loading_interval_index"]]
        assert interval["start_t"] <= loaded["t"] <= interval["end_t"]
        assert interval["requested_url"] == page["requested_url"]
    assert metadata["route"] == [page["title"] for page in pages]
    assert metadata["hops"] == len(hops)
    if metadata["success"]:
        assert pages[-1]["title"] == "Sun" and pages[-1]["url"] == "https://en.wikipedia.org/wiki/Sun"
    result = {
        "verified": True, "group_size": metadata["group_size"], "success": metadata["success"],
        "hops": len(hops), "pages": len(pages), "single_token_label_count": len(labels),
        "questions": questions_total, "model_api_calls": calls_total, "model_forwards": forwards_total,
        "model_seconds": metadata["model_seconds"], "elapsed_seconds": metadata["elapsed_seconds"],
        "active_elapsed_seconds": metadata["active_elapsed_seconds"],
        "loading_seconds": metadata["loading_seconds"], "route": metadata["route"],
        "first_page_round_candidates": [r["candidate_count"] for r in hops[0]["rounds"]] if hops else [],
        "checks": "100 unique exact-context single-token labels; all original links and contiguous groups covered; every argmax/index/label/token mapping valid; one output token per Choice verified by hooks and usage; source HTML edges and all paused-clock events verified.",
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((args.directory / "metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["model"], revision=metadata["model_revision"],
        local_files_only=args.local_files_only,
    )
    audit(args.directory, tokenizer)
